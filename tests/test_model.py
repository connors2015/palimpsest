"""Transformer correctness: analytic gradient == numerical, and it learns."""

import numpy as np
import pytest

from rig.model import (PARAM_COUNT, TinyTransformer, backward, forward,
                       init_vec)
from rig.task import make_batch


def test_param_count_matches_spec():
    assert PARAM_COUNT == len(init_vec(np.random.default_rng(0)))


def test_analytic_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    vec = init_vec(rng)
    batch = make_batch(rng, 4)
    _, cache = forward(vec, *batch)
    g = backward(cache)
    eps = 1e-5
    for i in rng.choice(PARAM_COUNT, size=40, replace=False):
        vp = vec.copy(); vp[i] += eps
        vm = vec.copy(); vm[i] -= eps
        num = (forward(vp, *batch)[0] - forward(vm, *batch)[0]) / (2 * eps)
        rel = abs(num - g[i]) / (abs(num) + abs(g[i]) + 1e-9)
        assert rel < 1e-4, f"grad mismatch at {i}: {rel:.2e}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_model_learns_delayed_copy(seed):
    model = TinyTransformer()
    rng = np.random.default_rng(seed)
    vec = model.init(rng)
    test = model.sample_batch(np.random.default_rng(seed + 500), 200)
    start = model.accuracy(vec, test)
    for _ in range(300):
        vec = model.train_step(vec, model.sample_batch(rng, 32), lr=0.3, steps=1)
    end = model.accuracy(vec, test)
    assert start < 0.4 and end > 0.95, f"{start:.2f} -> {end:.2f}"


def test_train_step_is_deterministic():
    model = TinyTransformer()
    vec = model.init(np.random.default_rng(3))
    batch = model.sample_batch(np.random.default_rng(4), 16)
    a = model.train_step(vec, batch, lr=0.3, steps=2)
    b = model.train_step(vec, batch, lr=0.3, steps=2)
    assert np.array_equal(a, b)
