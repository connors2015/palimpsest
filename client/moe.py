"""Real (PyTorch) mixture-of-experts GPT with page-sharded hold/train/serve.

The data-parallel client (client/gpt.py) makes every node hold the WHOLE model —
so the biggest model the network can train is the biggest that fits on its
smallest GPU. This lifts that ceiling: most parameters live in EXPERTS, the
model's weights are addressed as PAGES, and a node may hold only a SUBSET of the
pages (a shard). It trains the experts it holds and serves the tokens routed to
them; it never needs the whole model in memory.

Two things make this fit the chain unchanged:

  * flat_params / set_flat_params already flatten an nn.Module in parameter
    order, so the MoE model speaks the same flat vector the chain aggregates. A
    node that holds only some pages simply MASKS its pseudo-gradient to those
    pages (PageMap.mask) — every other coordinate is zero, so the chain's
    trimmed-mean aggregation combines disjoint contributions with no special
    case. Consensus is untouched; sharding is purely who-computes-which-page.

  * the weights are paged exactly as rig/moe.py commits them (a backbone page +
    one page per (layer, expert)), so sparse serving loads backbone + the routed
    experts and can prove those pages against the committed Merkle root.

"Split even smaller if required": PageMap's granularity is a knob. At
granularity="expert" a page is a whole expert; subdivide(max_page=s) chops every
page into ≤ s-parameter sub-pages, down to s=1 — one page per individual weight.
So the same machinery shards experts across big GPUs OR lets one node train a
handful of individual weights at ultimate fidelity. Holding a slice for TRAINING
still needs the backbone to run the forward/backward for those weights (gradients
aren't local); true hold-only-a-slice across the depth is pipeline parallelism,
the next step up. MoE experts are the self-contained, internet-friendly middle.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import pick_device


@dataclass
class MoEGPTConfig:
    vocab_size: int = 256          # byte-level, no tokenizer
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    n_experts: int = 8
    top_k: int = 2
    dropout: float = 0.0

    @property
    def d_ff(self):
        return 4 * self.n_embd


class Expert(nn.Module):
    """One FFN expert: the unit of sharding. Self-contained (no cross-expert
    state), so a node can hold, train, and serve any subset of experts."""

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of experts. Training computes all experts but gates
    every token to its top-k (others get zero weight), so the result is IDENTICAL
    to sparse serving that evaluates only the selected experts — which is what
    makes skipping them at inference sound (rig/moe_transformer.py)."""

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])

    def _gates(self, x):
        """Top-k softmax gates per token; non-selected experts are exactly 0."""
        logits = self.router(x)                                   # (..., E)
        topv, topi = logits.topk(self.cfg.top_k, dim=-1)
        w = torch.zeros_like(logits).scatter_(-1, topi, F.softmax(topv, dim=-1))
        return w                                                  # (..., E), k non-zeros

    def forward(self, x):
        w = self._gates(x)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                # dense in training…
            we = w[..., e:e + 1]
            if torch.any(we > 0):
                out = out + we * expert(x)
        return out

    @torch.no_grad()
    def serve(self, x, held=None):
        """Sparse serving: evaluate ONLY the experts a token routes to (and that
        this node holds, if `held` is given). Returns (output, experts_touched)."""
        w = self._gates(x)
        out = torch.zeros_like(x)
        touched = set()
        for e, expert in enumerate(self.experts):
            we = w[..., e:e + 1]
            sel = torch.any(we > 0)
            if sel and (held is None or e in held):
                out = out + we * expert(x)
                touched.add(e)
        return out, touched


class MoEBlock(nn.Module):
    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.moe = MoEFeedForward(cfg)                            # experts live here
        self.register_buffer("mask", torch.triu(
            torch.ones(cfg.block_size, cfg.block_size) * float("-inf"), diagonal=1))

    def forward(self, x):
        h = self.ln1(x)
        T = x.size(1)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        x = x + self.moe(self.ln2(x))
        return x


class MoEGPT(nn.Module):
    """A byte-level GPT whose FFNs are mixtures of experts. Ordinary nn.Module, so
    it plugs into client/trainer.py (flat_params/DiLoCo) and the chain unchanged."""

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_moe(cfg: MoEGPTConfig = None, device: str = None, seed: int = None):
    from .gpt import apply_genesis
    cfg = cfg or MoEGPTConfig()
    device = device or pick_device()
    model = MoEGPT(cfg).to(device)
    if seed is not None:                                          # shared genesis
        apply_genesis(model, seed)
    return model, device


# --------------------------------------------------------------------------
# PageMap — address the flat parameter vector as pages, at any granularity
# --------------------------------------------------------------------------
class PageMap:
    """Maps an MoEGPT's flat parameter vector (client/trainer.flat_params order)
    into PAGES so a node can hold/train a subset. A page is a contiguous slice of
    the flat vector; experts fall on contiguous slices because each expert's
    parameters are registered together. Two page kinds:

        ("backbone",)              — embeddings, attention, router, norms, head
        ("expert", layer, expert)  — one FFN expert (the shard unit)

    subdivide(max_page) chops every page into ≤ max_page-parameter sub-pages, so
    the same shard machinery scales from whole-expert down to per-weight pages.
    """

    def __init__(self, model: MoEGPT):
        self.n = model.num_params()
        self.pages = {}                       # page-key -> (start, end)
        self.expert_of = {}                   # (layer, e) -> page-key
        i = 0
        cur_expert, cur_start = None, None
        for name, p in model.named_parameters():
            sz = p.numel()
            ex = self._expert_key(name)       # (layer, e) or None
            if ex != cur_expert:
                if cur_expert is not None:
                    self.pages[("expert", *cur_expert)] = (cur_start, i)
                cur_expert = ex
                cur_start = i if ex is not None else None
            i += sz
        if cur_expert is not None:
            self.pages[("expert", *cur_expert)] = (cur_start, i)
        # backbone = every index not covered by an expert page
        covered = np.zeros(self.n, dtype=bool)
        for (start, end) in self.pages.values():
            covered[start:end] = True
        self.backbone_idx = np.nonzero(~covered)[0]
        for key in list(self.pages):
            if key[0] == "expert":
                self.expert_of[(key[1], key[2])] = key

    @staticmethod
    def _expert_key(name: str):
        # names like "blocks.0.moe.experts.3.fc.weight"
        parts = name.split(".")
        if "experts" in parts and "blocks" in parts:
            layer = int(parts[parts.index("blocks") + 1])
            e = int(parts[parts.index("experts") + 1])
            return (layer, e)
        return None

    @property
    def experts(self):
        return sorted(self.expert_of)                 # list of (layer, e)

    def expert_indices(self, expert_keys) -> np.ndarray:
        """Flat indices covered by the given (layer, e) experts."""
        out = []
        for k in expert_keys:
            s, e = self.pages[self.expert_of[k]]
            out.append(np.arange(s, e))
        return np.concatenate(out) if out else np.array([], dtype=int)

    def mask(self, expert_keys, include_backbone=True) -> np.ndarray:
        """Boolean mask over the flat vector for a node that HOLDS these experts
        (plus the backbone, which every trainer needs to run the forward pass)."""
        m = np.zeros(self.n, dtype=bool)
        if include_backbone:
            m[self.backbone_idx] = True
        idx = self.expert_indices(expert_keys)
        if idx.size:
            m[idx] = True
        return m

    def subdivide(self, max_page: int):
        """Every page split into contiguous sub-pages of ≤ max_page params. At
        max_page=1 each parameter is its own page — the finest possible shard,
        the 'train individual weights' granularity. Returns list of (start, end)."""
        spans = [(s, e) for (s, e) in self.pages.values()]
        if self.backbone_idx.size:                    # backbone as contiguous runs
            b = self.backbone_idx
            brk = np.nonzero(np.diff(b) > 1)[0]
            starts = np.concatenate([[0], brk + 1])
            ends = np.concatenate([brk + 1, [b.size]])
            spans += [(int(b[s]), int(b[e - 1]) + 1) for s, e in zip(starts, ends)]
        out = []
        for s, e in sorted(spans):
            for a in range(s, e, max_page):
                out.append((a, min(a + max_page, e)))
        return out


def mask_delta(delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only the components a node is responsible for; zero the rest. The
    chain aggregates the result unchanged — untouched pages simply get no
    contribution from this node."""
    out = delta.copy()
    out[~mask] = 0
    return out


def shard_aggregate(base_int: np.ndarray, deltas_int, masks) -> np.ndarray:
    """The sharded DiLoCo outer step: each coordinate is averaged over the nodes
    that HOLD it, not over all nodes. A page trained by one node is applied in
    full (not halved by absent nodes); the shared backbone is averaged over
    everyone. Deterministic integer math, like the chain's trimmed_mean_int — so
    this is the page-aware generalization of trainer.outer_apply."""
    stacked = np.stack([mask_delta(d, m) for d, m in zip(deltas_int, masks)])
    holders = np.stack(masks).sum(axis=0)               # per-coordinate holder count
    summed = stacked.sum(axis=0)
    out = base_int.copy()
    nz = holders > 0
    out[nz] = base_int[nz] + summed[nz] // holders[nz]  # integer mean over holders
    return out


def load_fraction(pagemap: PageMap, held_experts) -> float:
    """Fraction of parameters a node loads if it holds `held_experts` + backbone —
    the memory win of sharding vs holding the whole model."""
    m = pagemap.mask(held_experts, include_backbone=True)
    return float(m.sum()) / pagemap.n
