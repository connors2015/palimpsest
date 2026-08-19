"""Chain primitives: fixed-point determinism, robust aggregation, replay (§3)."""

import numpy as np
import pytest

from rig.chain import (Chain, dequantize, quantize, state_root,
                       trimmed_mean_int)


def test_quantize_roundtrip_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    a, b = quantize(x), quantize(x)
    assert np.array_equal(a, b)
    # round-trip within one quantization step
    assert np.max(np.abs(dequantize(quantize(x)) - x)) <= 1.0 / (1 << 16)


def test_trimmed_mean_is_deterministic_and_order_independent():
    rng = np.random.default_rng(1)
    deltas = [quantize(rng.standard_normal(50)) for _ in range(7)]
    r1 = trimmed_mean_int(deltas)
    r2 = trimmed_mean_int(list(reversed(deltas)))
    assert np.array_equal(r1, r2)  # elementwise sort => order independent


def test_trimmed_mean_bounds_outlier_influence():
    base = [quantize(np.zeros(10)) for _ in range(6)]
    poisoned = base + [quantize(np.full(10, 1e6))]  # one wild outlier
    agg = trimmed_mean_int(poisoned)
    assert np.max(np.abs(agg)) < 1000  # outlier trimmed away


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_apply_then_replay_is_bit_exact(seed):
    rng = np.random.default_rng(seed)
    chain = Chain(quantize(np.zeros(64)))
    for _ in range(30):
        deltas = [quantize(rng.standard_normal(64) * 0.1) for _ in range(5)]
        chain.apply_block(deltas, list(range(5)))
    assert state_root(chain.replay()) == chain.blocks[-1].root


def test_excision_removes_targeted_miner_contributions():
    rng = np.random.default_rng(7)
    chain = Chain(quantize(np.zeros(32)))
    for _ in range(20):
        deltas = [quantize(rng.standard_normal(32) * 0.05) for _ in range(4)]
        chain.apply_block(deltas, [0, 1, 2, 3])
    full = chain.replay()
    excised = chain.replay(exclude_miner_ids={3})
    assert not np.array_equal(full, excised)  # removing a miner changes state
