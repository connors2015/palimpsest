"""A real (small) GPT — the model the network actually trains (WHITEPAPER §3, §6).

This replaces the numpy TinyTransformer of the mechanism-proof rig with an actual
PyTorch language model, trained on real text on whatever GPU the volunteer has
(CUDA / Apple MPS / CPU, auto-detected). It is a standard nanoGPT-style
decoder — token + positional embeddings, pre-norm transformer blocks with causal
self-attention and a GeLU MLP, a tied-free LM head.

The chain never sees this model's internals; it only ever aggregates the flat
parameter DELTA a client submits (client/trainer.py). So the model can be
ordinary non-deterministic float GPU training (§6.3, the inner loop is
unconstrained) while consensus stays bit-exact — the two meet only at the
quantised delta.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GPTConfig:
    vocab_size: int = 256          # byte-level LM — no tokenizer to ship
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))
        self.register_buffer("mask", torch.triu(
            torch.ones(cfg.block_size, cfg.block_size) * float("-inf"), diagonal=1))

    def forward(self, x):
        h = self.ln1(x)
        T = x.size(1)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

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

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0):
        for _ in range(n_new):
            idx_c = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_c)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def apply_genesis(model: nn.Module, seed: int):
    """Overwrite every parameter with a deterministic, version-independent draw so
    the genesis weights are BIT-IDENTICAL on every node — the network constant.

    torch's own init RNG differs across torch versions and devices (MPS vs CUDA),
    which would fork the chain at genesis. numpy's RNG is byte-stable across
    platforms/versions, so we seed the genesis from it. We still respect the init
    scheme by parameter name (weights ~N(0,0.02); LayerNorm scale =1; biases =0),
    and numpy draws are consumed in named_parameters() order — identical on all
    nodes. set_flat_params-style .double() keeps it device-agnostic."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():        # deterministic order
            if name.endswith("bias") or "in_proj_bias" in name:
                vals = np.zeros(tuple(p.shape))
            elif name.endswith("weight") and (".ln" in name or name.startswith("ln")):
                vals = np.ones(tuple(p.shape))          # LayerNorm scale
            else:
                vals = rng.standard_normal(tuple(p.shape)) * 0.02
            p.copy_(torch.from_numpy(vals).to(dtype=p.dtype, device=p.device))


def build(cfg: GPTConfig = None, device: str = None, seed: int = None):
    cfg = cfg or GPTConfig()
    device = device or pick_device()
    model = GPT(cfg).to(device)
    if seed is not None:                                 # shared genesis across all nodes
        apply_genesis(model, seed)
    return model, device
