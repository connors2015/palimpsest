"""A bigger, configurable transformer on the autograd engine (rig/autograd.py).

Multi-head attention, stacked pre-norm blocks (RMSNorm), a ReLU MLP, and tied
or untied output — all expressed with autograd Tensors, so the gradient is
computed, not hand-derived. Defaults are still toy scale but meaningfully
larger and deeper than rig/model.py (the single-layer hand-backprop model),
which is what stresses DiLoCo aggregation at depth.

Exposes the same Model interface (init / train_step / loss / accuracy /
infer / sample_batch) so the chain, e2e, node, and async node use it
unchanged. Parameters serialize to one flat float vector, matching the
chain's quantize/aggregate contract (§3.4).
"""

from dataclasses import dataclass, field

import numpy as np

from .autograd import Tensor, cross_entropy, embedding
from .task import make_batch, make_batch_modadd

_TASKS = {"copy": make_batch, "modadd": make_batch_modadd}


@dataclass
class Config:
    # Model size (the "bigger model"): 2 layers, 4 heads, d=64 -> ~68k params.
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    # Task dimensions (validated to converge through DiLoCo aggregation).
    vocab: int = 8
    context: int = 12
    lag: int = 2
    task: str = "modadd"     # "copy" | "modadd" — what miners train on


def _spec(cfg: Config):
    V, T, D, F = cfg.vocab, cfg.context, cfg.d_model, cfg.d_ff
    spec = [("tok_emb", (V, D)), ("pos_emb", (T, D))]
    for l in range(cfg.n_layers):
        spec += [
            (f"ln1_{l}", (D,)), (f"Wq_{l}", (D, D)), (f"Wk_{l}", (D, D)),
            (f"Wv_{l}", (D, D)), (f"Wo_{l}", (D, D)),
            (f"ln2_{l}", (D,)), (f"W1_{l}", (D, F)), (f"b1_{l}", (F,)),
            (f"W2_{l}", (F, D)), (f"b2_{l}", (D,)),
        ]
    spec += [("lnf", (D,)), ("Wout", (D, V)), ("bout", (V,))]
    return spec


class BigTransformer:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.spec = _spec(self.cfg)
        self.param_count = sum(int(np.prod(s)) for _, s in self.spec)
        self._causal = np.tril(np.ones((self.cfg.context, self.cfg.context))).astype(bool)

    # -- (de)serialization -------------------------------------------------
    def init(self, rng) -> np.ndarray:
        parts = []
        for name, shape in self.spec:
            if name.startswith(("ln", "lnf")):
                parts.append(np.ones(shape).ravel())          # norm gains -> 1
            elif name.startswith("b"):
                parts.append(np.zeros(shape).ravel())
            else:
                parts.append((rng.standard_normal(shape)
                              / np.sqrt(shape[0])).ravel())
        return np.concatenate(parts).astype(np.float64)

    def _unflatten(self, vec, wrap=True) -> dict:
        out, i = {}, 0
        for name, shape in self.spec:
            n = int(np.prod(shape))
            arr = vec[i:i + n].reshape(shape)
            out[name] = Tensor(arr) if wrap else arr
            i += n
        return out

    # -- forward -----------------------------------------------------------
    def _forward(self, vec, tokens, targets, mask):
        cfg, p = self.cfg, self._unflatten(vec)
        B, T = tokens.shape
        H, D = cfg.n_heads, cfg.d_model
        dh = D // H
        scale = 1.0 / np.sqrt(dh)

        # Batches always span the full context, so pos_emb ([T,D]) broadcast-adds
        # over the batch axis while staying attached to the graph.
        x = embedding(p["tok_emb"], tokens) + p["pos_emb"]
        cmask = self._causal[:T, :T][None, None]              # [1,1,T,T]

        for l in range(cfg.n_layers):
            h = x.rms_norm(p[f"ln1_{l}"])
            q = h.matmul(p[f"Wq_{l}"]); k = h.matmul(p[f"Wk_{l}"])
            v = h.matmul(p[f"Wv_{l}"])
            q = self._split_heads(q, B, T, H, dh)             # [B,H,T,dh]
            k = self._split_heads(k, B, T, H, dh)
            v = self._split_heads(v, B, T, H, dh)
            scores = q.matmul(k.transpose(-1, -2)) * Tensor(scale)
            attn = scores.softmax_lastdim(mask=cmask)
            ctx = attn.matmul(v)                              # [B,H,T,dh]
            ctx = self._merge_heads(ctx, B, T, D)
            x = x + ctx.matmul(p[f"Wo_{l}"])                  # residual
            h2 = x.rms_norm(p[f"ln2_{l}"])
            ff = (h2.matmul(p[f"W1_{l}"]) + p[f"b1_{l}"]).relu()
            x = x + (ff.matmul(p[f"W2_{l}"]) + p[f"b2_{l}"])  # residual

        x = x.rms_norm(p["lnf"])
        logits = x.matmul(p["Wout"]) + p["bout"]
        loss = cross_entropy(logits, targets, mask)
        return loss, logits, p

    def _split_heads(self, t: Tensor, B, T, H, dh) -> Tensor:
        # [B,T,D] -> [B,H,T,dh] via reshape+transpose expressed as raw ops.
        r = Tensor(t.data.reshape(B, T, H, dh), (t,))
        def _bw():
            t._accum(r.grad.reshape(B, T, H * dh))
        r._backward = _bw
        return r.transpose(1, 2)

    def _merge_heads(self, t: Tensor, B, T, D) -> Tensor:
        m = t.transpose(1, 2)                                 # [B,T,H,dh]
        r = Tensor(m.data.reshape(B, T, D), (m,))
        def _bw():
            m._accum(r.grad.reshape(m.data.shape))
        r._backward = _bw
        return r

    def _grad(self, vec, batch) -> np.ndarray:
        tokens, targets, mask = batch
        loss, _, p = self._forward(vec, tokens, targets, mask)
        loss.backward()
        return np.concatenate([p[name].grad.ravel() for name, _ in self.spec])

    # -- Model interface ---------------------------------------------------
    def train_step(self, vec, batch, lr=0.3, steps=1) -> np.ndarray:
        v = vec.copy()
        for _ in range(steps):
            v = v - lr * self._grad(v, batch)
        return v

    def loss(self, vec, batch) -> float:
        return float(self._forward(vec, *batch)[0].data)

    def _logits(self, vec, tokens):
        B, T = tokens.shape
        dummy = np.zeros_like(tokens)
        return self._forward(vec, tokens, dummy, np.ones((B, T)))[1].data

    def accuracy(self, vec, batch) -> float:
        tokens, targets, mask = batch
        pred = self._logits(vec, tokens).argmax(axis=-1)
        m = mask.astype(bool)
        return float((pred[m] == targets[m]).mean())

    def infer(self, vec, tokens) -> np.ndarray:
        return self._logits(vec, tokens).argmax(axis=-1)

    def sample_batch(self, rng, batch):
        return _TASKS[self.cfg.task](rng, batch, vocab=self.cfg.vocab,
                                     context=self.cfg.context, lag=self.cfg.lag)


if __name__ == "__main__":
    import time

    from .chain import state_root
    from .node import run_in_memory
    cfg = Config()
    model = BigTransformer(cfg)
    print(f"BigTransformer: {model.param_count} params "
          f"({cfg.n_layers} layers, {cfg.n_heads} heads, d={cfg.d_model}, "
          f"task={cfg.task})")
    t0 = time.time()
    chain, log = run_in_memory(blocks=25, seed=7, model=model)
    print(f"through the chain (DiLoCo aggregation, {len(log.heights)} blocks):")
    print(f"  model accuracy {log.acc[0]:.3f} -> {log.acc[-1]:.3f}")
    print(f"  replay bit-exact: {state_root(chain.replay()) == chain.blocks[-1].root}")
    print(f"  miners rewarded: {sorted(log.rewards)}")
    print(f"  wall time: {time.time() - t0:.1f}s")
    raise SystemExit(0 if log.acc[-1] > 0.85 else 1)
