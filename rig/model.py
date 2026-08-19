"""A tiny decoder-only transformer in numpy, with manual backprop.

One layer, one head, causal mask, residual connections, ReLU MLP, no
LayerNorm (kept out to keep the backward pass small and auditable). ~2.6k
parameters. Everything a miner does with this model — the inner training loop
— is ordinary float math (WHITEPAPER §6.3: the inner loop is unconstrained).
The chain only ever quantizes and aggregates the flat parameter vector, which
is where determinism is required (§3.4).

The `Model` protocol at the bottom is what the chain, e2e demo, and node all
program against, so the model behind the flywheel is swappable.
"""

from dataclasses import dataclass

import numpy as np

from .task import CONTEXT, LAG, VOCAB, make_batch

# Fixed architecture. Param order here defines the flat-vector layout.
D_MODEL = 16
D_FF = 32

# (name, shape) in a fixed order — the serialization contract for flatten().
_SPEC = [
    ("tok_emb", (VOCAB, D_MODEL)),
    ("pos_emb", (CONTEXT, D_MODEL)),
    ("Wq", (D_MODEL, D_MODEL)),
    ("Wk", (D_MODEL, D_MODEL)),
    ("Wv", (D_MODEL, D_MODEL)),
    ("Wo", (D_MODEL, D_MODEL)),
    ("W1", (D_MODEL, D_FF)),
    ("b1", (D_FF,)),
    ("W2", (D_FF, D_MODEL)),
    ("b2", (D_MODEL,)),
    ("Wout", (D_MODEL, VOCAB)),
    ("bout", (VOCAB,)),
]
PARAM_COUNT = sum(int(np.prod(s)) for _, s in _SPEC)
_CAUSAL = np.tril(np.ones((CONTEXT, CONTEXT)))


def init_vec(rng: np.random.Generator) -> np.ndarray:
    parts = []
    for name, shape in _SPEC:
        fan_in = shape[0]
        scale = 0.0 if name.startswith("b") else 1.0 / np.sqrt(fan_in)
        parts.append((rng.standard_normal(shape) * scale).ravel())
    return np.concatenate(parts).astype(np.float64)


def unflatten(vec: np.ndarray) -> dict:
    out, i = {}, 0
    for name, shape in _SPEC:
        n = int(np.prod(shape))
        out[name] = vec[i:i + n].reshape(shape)
        i += n
    return out


def flatten(params: dict) -> np.ndarray:
    return np.concatenate([params[name].ravel() for name, _ in _SPEC])


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(vec, tokens, targets, mask):
    """Returns (mean_loss, cache). Shapes: tokens/targets/mask [B, T]."""
    p = unflatten(vec)
    B, T = tokens.shape
    scale = 1.0 / np.sqrt(D_MODEL)

    X = p["tok_emb"][tokens] + p["pos_emb"][None, :T, :]        # [B,T,D]
    Q, K, V = X @ p["Wq"], X @ p["Wk"], X @ p["Wv"]            # [B,T,D]
    scores = np.einsum("btd,bsd->bts", Q, K) * scale           # [B,T,T]
    scores = np.where(_CAUSAL[:T, :T][None] > 0, scores, -1e30)
    A = _softmax(scores, axis=-1)                              # [B,T,T]
    Ctx = np.einsum("bts,bsd->btd", A, V)                      # [B,T,D]
    attn = Ctx @ p["Wo"]
    H1 = X + attn                                              # residual
    Z = H1 @ p["W1"] + p["b1"]
    Hr = np.maximum(Z, 0.0)
    M = Hr @ p["W2"] + p["b2"]
    H2 = H1 + M                                                # residual
    logits = H2 @ p["Wout"] + p["bout"]                        # [B,T,V]
    probs = _softmax(logits, axis=-1)

    idx = (np.arange(B)[:, None], np.arange(T)[None, :], targets)
    tok_loss = -np.log(probs[idx] + 1e-12)
    m = mask.astype(np.float64)
    loss = float((tok_loss * m).sum() / m.sum())

    cache = dict(p=p, tokens=tokens, targets=targets, mask=m, B=B, T=T,
                 X=X, Q=Q, K=K, V=V, A=A, Ctx=Ctx, H1=H1, Z=Z, Hr=Hr,
                 H2=H2, probs=probs, scale=scale)
    return loss, cache


def backward(cache) -> np.ndarray:
    """Analytic gradient of the masked mean cross-entropy wrt the flat vec."""
    p, B, T = cache["p"], cache["B"], cache["T"]
    m, scale = cache["mask"], cache["scale"]
    g = {name: np.zeros_like(p[name]) for name, _ in _SPEC}

    # d loss / d logits
    dlogits = cache["probs"].copy()
    idx = (np.arange(B)[:, None], np.arange(T)[None, :], cache["targets"])
    dlogits[idx] -= 1.0
    dlogits *= (m / m.sum())[:, :, None]                       # [B,T,V]

    g["Wout"] = np.einsum("btd,btv->dv", cache["H2"], dlogits)
    g["bout"] = dlogits.sum(axis=(0, 1))
    dH2 = dlogits @ p["Wout"].T                                # [B,T,D]

    # MLP block (H2 = H1 + M)
    dH1 = dH2.copy()
    dM = dH2
    g["W2"] = np.einsum("btf,btd->fd", cache["Hr"], dM)
    g["b2"] = dM.sum(axis=(0, 1))
    dHr = dM @ p["W2"].T
    dZ = dHr * (cache["Z"] > 0)
    g["W1"] = np.einsum("btd,btf->df", cache["H1"], dZ)
    g["b1"] = dZ.sum(axis=(0, 1))
    dH1 += dZ @ p["W1"].T

    # Attention block (H1 = X + attn), attn = Ctx @ Wo
    dX = dH1.copy()
    dattn = dH1
    g["Wo"] = np.einsum("btd,bte->de", cache["Ctx"], dattn)
    dCtx = dattn @ p["Wo"].T                                   # [B,T,D]

    # Ctx = A @ V
    dA = np.einsum("btd,bsd->bts", dCtx, cache["V"])
    dV = np.einsum("bts,btd->bsd", cache["A"], dCtx)
    # softmax backward (row-wise over last axis)
    dscores = cache["A"] * (dA - (dA * cache["A"]).sum(axis=-1, keepdims=True))
    dscores *= scale
    dQ = np.einsum("bts,bsd->btd", dscores, cache["K"])
    dK = np.einsum("bts,btd->bsd", dscores, cache["Q"])

    g["Wq"] = np.einsum("btd,bte->de", cache["X"], dQ)
    g["Wk"] = np.einsum("btd,bte->de", cache["X"], dK)
    g["Wv"] = np.einsum("btd,bte->de", cache["X"], dV)
    dX += dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T

    # Embeddings: X = tok_emb[tokens] + pos_emb[:T]
    np.add.at(g["tok_emb"], cache["tokens"], dX)
    g["pos_emb"][:T] += dX.sum(axis=0)

    return flatten(g)


# --------------------------------------------------------------------------
# Model interface (chain / e2e / node program against this)
# --------------------------------------------------------------------------
@dataclass
class TinyTransformer:
    lag: int = LAG
    param_count: int = PARAM_COUNT

    def init(self, rng) -> np.ndarray:
        return init_vec(rng)

    def train_step(self, vec, batch, lr=0.3, steps=1) -> np.ndarray:
        tokens, targets, mask = batch
        v = vec.copy()
        for _ in range(steps):
            _, cache = forward(v, tokens, targets, mask)
            v = v - lr * backward(cache)
        return v

    def loss(self, vec, batch) -> float:
        return forward(vec, *batch)[0]

    def accuracy(self, vec, batch) -> float:
        tokens, targets, mask = batch
        _, cache = forward(vec, tokens, targets, mask)
        pred = cache["probs"].argmax(axis=-1)
        m = mask.astype(bool)
        return float((pred[m] == targets[m]).mean())

    def infer(self, vec, tokens) -> np.ndarray:
        """Greedy next-token prediction per position (the served output)."""
        B, T = tokens.shape
        dummy = np.zeros_like(tokens)
        _, cache = forward(vec, tokens, dummy, np.ones((B, T)))
        return cache["probs"].argmax(axis=-1)

    def sample_batch(self, rng, batch):
        return make_batch(rng, batch, lag=self.lag)
