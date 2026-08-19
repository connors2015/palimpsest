"""A minimal numpy reverse-mode autograd engine.

Just enough operators to express the transformer (and, next, a mixture-of-
experts layer) without hand-deriving each gradient. A `Tensor` wraps a numpy
array and records a backward closure per op; `.backward()` walks the tape in
reverse topological order. Correctness is pinned by a numerical gradient check
in the test suite.

This lives entirely in miners' inner training loop and in scoring's forward
pass (WHITEPAPER §6.3: the inner loop is unconstrained float math). It never
touches the chain's consensus arithmetic, which stays integer fixed-point
(rig/chain.py) — so the autograd engine cannot affect replay determinism.
"""

import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_parents")

    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._backward = lambda: None
        self._parents = parents

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum a gradient back to `shape` after numpy broadcasting."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def _accum(self, g):
        self.grad = g if self.grad is None else self.grad + g

    # -- ops ---------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _bw():
            self._accum(self._unbroadcast(out.grad, self.data.shape))
            other._accum(other._unbroadcast(out.grad, other.data.shape))
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _bw():
            self._accum(self._unbroadcast(out.grad * other.data, self.data.shape))
            other._accum(other._unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _bw
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _bw():
            a, b, g = self.data, other.data, out.grad
            self._accum(self._unbroadcast(g @ np.swapaxes(b, -1, -2), a.shape))
            other._accum(other._unbroadcast(np.swapaxes(a, -1, -2) @ g, b.shape))
        out._backward = _bw
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def _bw():
            self._accum(out.grad * (self.data > 0))
        out._backward = _bw
        return out

    def transpose(self, ax1, ax2):
        out = Tensor(np.swapaxes(self.data, ax1, ax2), (self,))

        def _bw():
            self._accum(np.swapaxes(out.grad, ax1, ax2))
        out._backward = _bw
        return out

    def reshape(self, *shape):
        old = self.data.shape
        out = Tensor(self.data.reshape(*shape), (self,))

        def _bw():
            self._accum(out.grad.reshape(old))
        out._backward = _bw
        return out

    def slice_last(self, start, end):
        """Select columns [start:end] of the last axis; scatter grad back."""
        out = Tensor(self.data[..., start:end], (self,))

        def _bw():
            g = np.zeros_like(self.data)
            g[..., start:end] = out.grad
            self._accum(g)
        out._backward = _bw
        return out

    def rms_norm(self, gain, eps=1e-5):
        """RMSNorm over the last axis: x / sqrt(mean(x^2)+eps) * gain."""
        x = self.data
        ms = np.mean(x * x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(ms + eps)
        norm = x * inv
        out = Tensor(norm * gain.data, (self, gain))

        def _bw():
            g = out.grad
            gg = g * gain.data
            d = g.shape[-1]
            # d(norm)/dx backprop for RMSNorm
            dot = np.sum(gg * x, axis=-1, keepdims=True)
            dx = inv * gg - (inv ** 3) * x * dot / d
            self._accum(dx)
            gain._accum(Tensor._unbroadcast(g * norm, gain.data.shape))
        out._backward = _bw
        return out

    def softmax_lastdim(self, mask=None):
        z = self.data
        if mask is not None:
            z = np.where(mask, z, -1e30)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(p, (self,))

        def _bw():
            g = out.grad
            dz = p * (g - np.sum(g * p, axis=-1, keepdims=True))
            if mask is not None:
                dz = np.where(mask, dz, 0.0)
            self._accum(dz)
        out._backward = _bw
        return out

    # -- autodiff ----------------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()


def embedding(table: Tensor, idx: np.ndarray) -> Tensor:
    """Gather rows table[idx]; scatter-add on the backward pass."""
    out = Tensor(table.data[idx], (table,))

    def _bw():
        g = np.zeros_like(table.data)
        np.add.at(g, idx, out.grad)
        table._accum(g)
    out._backward = _bw
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray, mask: np.ndarray) -> Tensor:
    """Masked mean next-token cross-entropy. logits [B,T,V]."""
    z = logits.data
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    B, T, V = p.shape
    idx = (np.arange(B)[:, None], np.arange(T)[None, :], targets)
    m = mask.astype(np.float64)
    loss_val = -np.sum(np.log(p[idx] + 1e-12) * m) / m.sum()
    out = Tensor(loss_val, (logits,))

    def _bw():
        d = p.copy()
        d[idx] -= 1.0
        d *= (m / m.sum())[:, :, None]
        logits._accum(d * out.grad)
    out._backward = _bw
    return out
