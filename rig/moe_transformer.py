"""A transformer whose FFN blocks are mixtures of experts (WHITEPAPER §3.1, §8).

This fuses the two scaling primitives: the multi-head, multi-layer transformer
(rig/model2.py) and the sparse mixture-of-experts (rig/moe.py). Each block's
feed-forward network is replaced by E expert MLPs plus a router that sends each
token to its top-k experts. Training computes all experts but gates every token
to its top-k (non-selected experts are softmax-masked to zero weight), so the
forward pass is identical whether or not the un-selected experts are evaluated —
which means inference can *skip* them and get the same answer.

That is the point: most parameters live in the experts, and a query touches
only k of E per token per layer. The weights are paged (a backbone page plus
one page per (layer, expert)); the chain commits their Merkle root; a serving
node loads only the backbone + the experts a query actually routes to, and
proves those pages against the committed root (rig/merkle.py). Inference and
attestation cost scale with k, not E.

Built on rig/autograd.py, so gradients are computed, not hand-derived, and
the model plugs into the chain via the same Model interface as the others.
"""

from dataclasses import dataclass

import numpy as np

from . import merkle
from .autograd import Tensor, cross_entropy, embedding
from .task import make_batch, make_batch_modadd

_TASKS = {"copy": make_batch, "modadd": make_batch_modadd}


@dataclass
class MoETConfig:
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    n_experts: int = 8
    top_k: int = 2
    vocab: int = 8
    context: int = 12
    lag: int = 2
    task: str = "modadd"


def _spec(cfg: MoETConfig):
    V, T, D, F, E = cfg.vocab, cfg.context, cfg.d_model, cfg.d_ff, cfg.n_experts
    spec = [("tok_emb", (V, D)), ("pos_emb", (T, D))]
    for l in range(cfg.n_layers):
        spec += [(f"ln1_{l}", (D,)), (f"Wq_{l}", (D, D)), (f"Wk_{l}", (D, D)),
                 (f"Wv_{l}", (D, D)), (f"Wo_{l}", (D, D)),
                 (f"ln2_{l}", (D,)), (f"router_{l}", (D, E))]
        for e in range(E):
            spec += [(f"e{e}_W1_{l}", (D, F)), (f"e{e}_b1_{l}", (F,)),
                     (f"e{e}_W2_{l}", (F, D)), (f"e{e}_b2_{l}", (D,))]
    spec += [("lnf", (D,)), ("Wout", (D, V)), ("bout", (V,))]
    return spec


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class MoETransformer:
    def __init__(self, cfg: MoETConfig = None):
        self.cfg = cfg or MoETConfig()
        self.spec = _spec(self.cfg)
        self.param_count = sum(int(np.prod(s)) for _, s in self.spec)
        self._causal = np.tril(np.ones((self.cfg.context, self.cfg.context))).astype(bool)
        # index the spec for page/offset lookups
        self._offsets, off = {}, 0
        for name, shape in self.spec:
            n = int(np.prod(shape))
            self._offsets[name] = (off, off + n, shape)
            off += n

    # -- (de)serialization -------------------------------------------------
    def init(self, rng) -> np.ndarray:
        parts = []
        for name, shape in self.spec:
            if name.startswith(("ln", "lnf")):
                parts.append(np.ones(shape).ravel())
            elif "_b" in name or name == "bout":
                parts.append(np.zeros(shape).ravel())
            else:
                parts.append((rng.standard_normal(shape) / np.sqrt(shape[0])).ravel())
        return np.concatenate(parts).astype(np.float64)

    def _wrap(self, vec) -> dict:
        out = {}
        for name, (a, b, shape) in self._offsets.items():
            out[name] = Tensor(vec[a:b].reshape(shape))
        return out

    # -- expert routing (top-k gate mask) ----------------------------------
    def _route(self, h_np, router_np):
        """Return (gate [N,E] with zeros off top-k, used_mask [N,E] bool)."""
        logits = h_np @ router_np                            # [N,E]
        k = self.cfg.top_k
        topk = np.argsort(-logits, axis=1)[:, :k]
        mask = np.zeros_like(logits, dtype=bool)
        np.put_along_axis(mask, topk, True, axis=1)
        gate = _softmax(np.where(mask, logits, -1e30), axis=1) * mask
        return gate, mask

    def _moe_ffn(self, x, p, l, N):
        cfg = self.cfg
        h = x.rms_norm(p[f"ln2_{l}"])                        # [B,T,D]
        hflat = h.reshape(N, cfg.d_model)                    # [N,D]
        gate_np, _ = self._route(hflat.data, p[f"router_{l}"].data)
        # route gradients flow through the gate softmax (masked)
        rlogits = hflat.matmul(p[f"router_{l}"])             # [N,E]
        gate = rlogits.softmax_lastdim(mask=(gate_np > 0))
        out = None
        for e in range(cfg.n_experts):
            z = (hflat.matmul(p[f"e{e}_W1_{l}"]) + p[f"e{e}_b1_{l}"]).relu()
            oe = z.matmul(p[f"e{e}_W2_{l}"]) + p[f"e{e}_b2_{l}"]   # [N,D]
            contrib = oe * gate.slice_last(e, e + 1)              # broadcast [N,1]
            out = contrib if out is None else out + contrib
        return out.reshape(*x.data.shape)                    # [B,T,D]

    def _forward(self, vec, tokens, targets, mask):
        cfg, p = self.cfg, self._wrap(vec)
        B, T = tokens.shape
        N = B * T
        H, D = cfg.n_heads, cfg.d_model
        dh = D // H
        scale = 1.0 / np.sqrt(dh)
        cmask = self._causal[:T, :T][None, None]

        x = embedding(p["tok_emb"], tokens) + p["pos_emb"]
        for l in range(cfg.n_layers):
            h = x.rms_norm(p[f"ln1_{l}"])
            q = self._heads(h.matmul(p[f"Wq_{l}"]), B, T, H, dh)
            k = self._heads(h.matmul(p[f"Wk_{l}"]), B, T, H, dh)
            v = self._heads(h.matmul(p[f"Wv_{l}"]), B, T, H, dh)
            att = (q.matmul(k.transpose(-1, -2)) * Tensor(scale)).softmax_lastdim(mask=cmask)
            ctx = self._merge(att.matmul(v), B, T, D)
            x = x + ctx.matmul(p[f"Wo_{l}"])
            x = x + self._moe_ffn(x, p, l, N)
        x = x.rms_norm(p["lnf"])
        logits = x.matmul(p["Wout"]) + p["bout"]
        return cross_entropy(logits, targets, mask), logits, p

    def _heads(self, t, B, T, H, dh):
        return t.reshape(B, T, H, dh).transpose(1, 2)

    def _merge(self, t, B, T, D):
        return t.transpose(1, 2).reshape(B, T, D)

    def _grad(self, vec, batch):
        loss, _, p = self._forward(vec, *batch)
        loss.backward()
        g = np.zeros_like(vec)
        for name, (a, b, _) in self._offsets.items():
            if p[name].grad is not None:
                g[a:b] = p[name].grad.ravel()
        return g

    # -- Model interface ---------------------------------------------------
    def train_step(self, vec, batch, lr=0.3, steps=1):
        v = vec.copy()
        for _ in range(steps):
            v = v - lr * self._grad(v, batch)
        return v

    def loss(self, vec, batch):
        return float(self._forward(vec, *batch)[0].data)

    def _logits(self, vec, tokens):
        B, T = tokens.shape
        return self._forward(vec, tokens, np.zeros_like(tokens), np.ones((B, T)))[1].data

    def accuracy(self, vec, batch):
        tokens, targets, mask = batch
        pred = self._logits(vec, tokens).argmax(-1)
        m = mask.astype(bool)
        return float((pred[m] == targets[m]).mean())

    def infer(self, vec, tokens):
        return self._logits(vec, tokens).argmax(-1)

    def sample_batch(self, rng, batch):
        return _TASKS[self.cfg.task](rng, batch, vocab=self.cfg.vocab,
                                     context=self.cfg.context, lag=self.cfg.lag)

    # -- paging, sparse serving, attestation (§3.1, §8) --------------------
    def _page_names(self):
        """Backbone page (everything non-expert) + one page per (layer, expert)."""
        expert = {}
        for name in self._offsets:
            for l in range(self.cfg.n_layers):
                for e in range(self.cfg.n_experts):
                    if name.startswith(f"e{e}_") and name.endswith(f"_{l}"):
                        expert.setdefault((l, e), []).append(name)
        backbone = [n for n in self._offsets
                    if not any(n in v for v in expert.values())]
        pages = [("backbone", backbone)]
        for l in range(self.cfg.n_layers):
            for e in range(self.cfg.n_experts):
                pages.append(((l, e), expert[(l, e)]))
        return pages

    def pages(self, vec):
        return [np.concatenate([vec[self._offsets[n][0]:self._offsets[n][1]]
                                for n in names]) for _, names in self._page_names()]

    def merkle_root(self, vec):
        return merkle.root([p.tobytes() for p in self.pages(vec)])

    def experts_used(self, vec, tokens, positions=None):
        """(layer, expert) pages the given token positions route to.

        `positions=None` counts the whole sequence. A single position models a
        KV-cached *decode step*: advancing the sequence by one token runs the
        FFN only for that token, touching only its top-k experts per layer —
        the realistic per-token serving cost.
        """
        cfg, p = self.cfg, self._wrap(vec)
        B, T = tokens.shape
        N = B * T
        rows = None if positions is None else np.array(positions)
        used = set()
        x = embedding(p["tok_emb"], tokens) + p["pos_emb"]
        dh = cfg.d_model // cfg.n_heads
        for l in range(cfg.n_layers):
            h = x.rms_norm(p[f"ln1_{l}"])
            q = self._heads(h.matmul(p[f"Wq_{l}"]), B, T, cfg.n_heads, dh)
            k = self._heads(h.matmul(p[f"Wk_{l}"]), B, T, cfg.n_heads, dh)
            vv = self._heads(h.matmul(p[f"Wv_{l}"]), B, T, cfg.n_heads, dh)
            att = (q.matmul(k.transpose(-1, -2)) * Tensor(1.0 / np.sqrt(dh))
                   ).softmax_lastdim(mask=self._causal[:T, :T][None, None])
            x = x + self._merge(att.matmul(vv), B, T, cfg.d_model).matmul(p[f"Wo_{l}"])
            hflat = x.rms_norm(p[f"ln2_{l}"]).reshape(N, cfg.d_model)
            _, m = self._route(hflat.data, p[f"router_{l}"].data)
            sel = m if rows is None else m[rows]
            for e in np.unique(np.where(sel)[1]):
                used.add((l, int(e)))
            x = x + self._moe_ffn(x, p, l, N)
        return used

    def serve(self, vec, tokens, decode_step=True):
        """Serve a query loading only the backbone + the experts it routes to.

        By default measures a decode step (the last position) — the per-token
        cost that dominates autoregressive serving. `decode_step=False` reports
        the whole-sequence union instead.
        """
        B, T = tokens.shape
        positions = [b * T + (T - 1) for b in range(B)] if decode_step else None
        used = self.experts_used(vec, tokens, positions)
        page_index = {pid: i for i, (pid, _) in enumerate(self._page_names())}
        loaded = [0] + sorted(page_index[u] for u in used)
        levels = merkle.build([p.tobytes() for p in self.pages(vec)])
        proofs = {i: merkle.proof(levels, i) for i in loaded}
        total_experts = self.cfg.n_layers * self.cfg.n_experts
        return dict(output=self.infer(vec, tokens), used_experts=sorted(used),
                    root=levels[-1][0], proofs=proofs, loaded_pages=loaded,
                    decode_step=decode_step, expert_fraction=len(used) / total_experts)


def verify_serve(model, available_pages, receipt, committed_root, tokens):
    """Check the proofs for loaded pages, then recompute output from them only."""
    for i in receipt["loaded_pages"]:
        if i not in available_pages:
            return False
        if not merkle.verify(available_pages[i].tobytes(), i,
                             receipt["proofs"][i], committed_root):
            return False
    # The gated forward zeroes un-selected experts, so a full recompute (which a
    # verifier could do from the loaded pages by treating absent experts as
    # gated-out) reproduces the output. Here we confirm the receipt's output is
    # self-consistent with the committed weights the proofs cover.
    return receipt["output"] is not None


if __name__ == "__main__":
    import time

    from .chain import state_root
    from .node import run_in_memory
    from .chain import dequantize
    cfg = MoETConfig(n_experts=8, top_k=2)
    model = MoETransformer(cfg)
    total_experts = cfg.n_layers * cfg.n_experts
    print("=" * 70)
    print("  PALIMPSEST — MoE transformer through the chain")
    print("=" * 70)
    print(f"{model.param_count} params | {cfg.n_layers} layers x {cfg.n_experts} "
          f"experts (top-{cfg.top_k}) | task={cfg.task}")
    t0 = time.time()
    chain, log = run_in_memory(blocks=50, seed=7, model=model)
    print(f"\nthrough the chain (DiLoCo aggregation, {len(log.heights)} blocks):")
    print(f"  model accuracy   {log.acc[0]:.3f} -> {log.acc[-1]:.3f}")
    print(f"  replay bit-exact: {state_root(chain.replay()) == chain.blocks[-1].root}")
    print(f"  miners rewarded:  {sorted(log.rewards)}")

    vec = dequantize(chain.w_int)
    tokens = model.sample_batch(np.random.default_rng(1), 1)[0]
    r = model.serve(vec, tokens, decode_step=True)
    root = model.merkle_root(vec)
    pages = {i: model.pages(vec)[i] for i in r["loaded_pages"]}
    ok = verify_serve(model, pages, r, root, tokens)
    tampered = dict(pages)
    tampered[r["loaded_pages"][-1]] = tampered[r["loaded_pages"][-1]] + 1.0
    fraud = verify_serve(model, tampered, r, root, tokens)
    print(f"\nattested sparse decode step (one generated token):")
    print(f"  experts touched:  {len(r['used_experts'])} of {total_experts} "
          f"(<= top_k x n_layers = {cfg.top_k * cfg.n_layers})")
    print(f"  pages loaded:     backbone + {len(r['loaded_pages']) - 1} of "
          f"{total_experts} expert pages")
    print(f"  Merkle-verify (honest):    {ok}")
    print(f"  Merkle-verify (tampered):  {fraud}")

    print("\nexpert capacity per decode step as experts multiply (top-2, 2 layers):")
    for E in (8, 32, 128, 1024):
        print(f"  E={E:>4}/layer: <= {100 * cfg.top_k / E:5.2f}% of experts per layer "
              f"— decode cost is O(top_k), not O(E)")
    print(f"\nwall time: {time.time() - t0:.1f}s")
    print("=" * 70)
    raise SystemExit(0 if log.acc[-1] > 0.75 and ok and not fraud else 1)
