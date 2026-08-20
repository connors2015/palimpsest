"""DiPaCo — distributed path composition, no pipeline (§3.1, §6; DeepMind 2403.10616).

This is the shard design a volunteer network actually wants, and it needs NO
activation exchange between machines. The model is L levels, each level a set of
M interchangeable MODULES. A PATH picks one module per level; different sequences
route to different paths. The routing is COARSE — a whole sequence is assigned to
one path up front (here by its domain, exactly as DiPaCo pre-buckets data
offline) — so a worker that holds one path can run that sequence's ENTIRE
forward and backward locally. Nothing crosses the network mid-pass; the only
thing shipped is the infrequent DiLoCo pseudo-gradient. That is the whole point:
pipeline parallelism passes activations every micro-batch (latency-bound,
synchronous, non-deterministic); DiPaCo replaces it with coarse routing +
infrequent sync, which suits poorly-connected islands of compute.

Why this is not client/moe.py: that routes per TOKEN, top-k, so a batch's tokens
scatter across all experts and a worker would need every expert to take one
step. Coarse per-sequence routing makes a path self-contained — the property
that lets a node hold and train only its slice.

How it reuses what we built:
  * modules are FFN experts (client/moe.Expert), the shard unit;
  * a worker masks its pseudo-gradient to the pages it holds (backbone + its
    path's modules) and the chain averages each coordinate over its HOLDERS
    (client/moe.shard_aggregate) — module-level DiLoCo. A module used by several
    paths is averaged across those workers; a private module comes from its one
    owner; the backbone is averaged over everyone.

So no worker holds the whole model, no worker exchanges activations, and the
composed model (route each query to its path) beats any single path — paths
specialize, composition wins.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import apply_genesis, pick_device
from .moe import Expert


@dataclass
class DiPaCoConfig:
    vocab_size: int = 256          # byte-level
    n_layer: int = 2               # L levels
    n_head: int = 2
    n_embd: int = 32
    block_size: int = 16
    n_modules: int = 4             # M interchangeable modules per level
    dropout: float = 0.0

    @property
    def d_ff(self):
        return 4 * self.n_embd


class DiPaCoBlock(nn.Module):
    """Shared attention backbone + M interchangeable FFN modules. A forward pass
    uses exactly ONE module (the one the sequence's path selects at this level)."""

    def __init__(self, cfg: DiPaCoConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mods = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_modules)])
        self.register_buffer("mask", torch.triu(
            torch.ones(cfg.block_size, cfg.block_size) * float("-inf"), diagonal=1))

    def forward(self, x, m: int):
        h = self.ln1(x)
        T = x.size(1)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        x = x + self.mods[m](self.ln2(x))          # only module m runs — local, sparse
        return x


class DiPaCoGPT(nn.Module):
    """Byte-level GPT whose per-layer FFN is chosen by the sequence's PATH. A path
    is a list of length n_layer giving the module index at each level."""

    def __init__(self, cfg: DiPaCoConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([DiPaCoBlock(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None, path=None):
        assert path is not None, "DiPaCo forward needs a path (one module per level)"
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for l, b in enumerate(self.blocks):
            x = b(x, path[l])                       # one module per level = the path
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_dipaco(cfg: DiPaCoConfig = None, device: str = None, seed: int = None):
    cfg = cfg or DiPaCoConfig()
    device = device or pick_device()
    model = DiPaCoGPT(cfg).to(device)
    if seed is not None:
        apply_genesis(model, seed)                  # shared genesis, as ever
    return model, device


def make_path(path_id: int, n_layer: int, n_modules: int):
    """A deterministic path assignment. Path p uses module (p+level) mod M at each
    level, so paths OVERLAP (module reuse across paths) — the combinatorial reuse
    that lets M modules serve many paths, and that shard_aggregate averages."""
    return [(path_id + l) % n_modules for l in range(n_layer)]


def coarse_route(domain: int, n_paths: int) -> int:
    """Coarse, per-sequence routing: a whole sequence's domain picks its path,
    decided once (DiPaCo pre-buckets data offline the same way). This is what
    keeps a path self-contained — no per-token scatter, no activation exchange."""
    return domain % n_paths


# --------------------------------------------------------------------------
# PathMap — which flat-vector pages a worker HOLDS for a given path
# --------------------------------------------------------------------------
class PathMap:
    """Maps a DiPaCoGPT's flat parameters into a backbone page + one page per
    (level, module), so a worker can mask its pseudo-gradient to the pages it
    holds (backbone + its path's modules). Mirrors client/moe.PageMap."""

    def __init__(self, model: DiPaCoGPT):
        self.n = model.num_params()
        self.mod_span = {}                          # (layer, module) -> (start, end)
        i = 0
        cur, start = None, None
        for name, p in model.named_parameters():
            key = self._mod_key(name)
            if key != cur:
                if cur is not None:
                    self.mod_span[cur] = (start, i)
                cur, start = key, (i if key is not None else None)
            i += p.numel()
        if cur is not None:
            self.mod_span[cur] = (start, i)
        covered = np.zeros(self.n, dtype=bool)
        for s, e in self.mod_span.values():
            covered[s:e] = True
        self.backbone_idx = np.nonzero(~covered)[0]

    @staticmethod
    def _mod_key(name: str):
        parts = name.split(".")
        if "mods" in parts and "blocks" in parts:
            return (int(parts[parts.index("blocks") + 1]),
                    int(parts[parts.index("mods") + 1]))
        return None

    def path_modules(self, path):
        """The (level, module) pages a path touches."""
        return [(l, m) for l, m in enumerate(path)]

    def mask(self, path, include_backbone=True) -> np.ndarray:
        """Pages a worker on `path` holds: the backbone (needed to run any
        forward) + the one module per level the path selects."""
        m = np.zeros(self.n, dtype=bool)
        if include_backbone:
            m[self.backbone_idx] = True
        for key in self.path_modules(path):
            s, e = self.mod_span[key]
            m[s:e] = True
        return m

    def hold_fraction(self, path) -> float:
        return float(self.mask(path).sum()) / self.n
