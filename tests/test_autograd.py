"""The autograd engine itself: op gradients match numerical differentiation."""

import numpy as np

from rig.autograd import Tensor, cross_entropy, embedding


def _num_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
        it.iternext()
    return g


def test_matmul_add_mul_relu_chain():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 2))
    C = rng.standard_normal((3, 2))

    def f(a):
        ta, tb, tc = Tensor(a), Tensor(B), Tensor(C)
        out = (ta.matmul(tb) + tc).relu()
        return (out * out).data.sum()

    ta = Tensor(A)
    out = (ta.matmul(Tensor(B)) + Tensor(C)).relu()
    loss = out * out
    s = Tensor(loss.data.sum(), (loss,))
    s._backward = lambda: loss._accum(np.ones_like(loss.data))
    s.backward()
    assert np.allclose(ta.grad, _num_grad(f, A), atol=1e-5)


def test_rms_norm_grad():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 5, 8))
    gain = rng.standard_normal(8)

    def f(x):
        return (Tensor(x).rms_norm(Tensor(gain)).data ** 2).sum()

    tx = Tensor(X)
    out = tx.rms_norm(Tensor(gain))
    sq = out * out
    s = Tensor(sq.data.sum(), (sq,))
    s._backward = lambda: sq._accum(np.ones_like(sq.data))
    s.backward()
    assert np.allclose(tx.grad, _num_grad(f, X), atol=1e-4)


def test_softmax_and_cross_entropy_grad():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((2, 4, 6))
    targets = rng.integers(0, 6, size=(2, 4))
    mask = np.ones((2, 4), dtype=bool)

    def f(z):
        return cross_entropy(Tensor(z), targets, mask).data

    tz = Tensor(logits)
    cross_entropy(tz, targets, mask).backward()
    assert np.allclose(tz.grad, _num_grad(f, logits), atol=1e-5)


def test_embedding_scatter_grad():
    rng = np.random.default_rng(3)
    table = rng.standard_normal((5, 3))
    idx = np.array([[0, 2, 2], [1, 4, 0]])

    def f(t):
        return (embedding(Tensor(t), idx).data ** 2).sum()

    tt = Tensor(table)
    out = embedding(tt, idx)
    sq = out * out
    s = Tensor(sq.data.sum(), (sq,))
    s._backward = lambda: sq._accum(np.ones_like(sq.data))
    s.backward()
    assert np.allclose(tt.grad, _num_grad(f, table), atol=1e-5)
