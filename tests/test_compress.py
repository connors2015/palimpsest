"""Delta compression (§6): ratio, exact top-k, error feedback, determinism."""

import numpy as np

from client.compress import Compressor, compress, decompress, ratio
from rig.chain import dequantize, quantize


def test_topk_roundtrip_keeps_largest_exactly():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal(100_000) * 0.01
    payload = compress(delta, keep_frac=0.05)
    dense = decompress(payload)
    q = quantize(delta)
    # the kept components equal the quantised delta exactly; the rest are zero
    kept = np.nonzero(dense)[0]
    assert np.array_equal(dense[kept], q[kept])
    assert len(kept) <= int(0.05 * delta.size) + 1


def test_compression_ratio_is_large():
    rng = np.random.default_rng(1)
    delta = rng.standard_normal(500_000) * 0.01
    payload = compress(delta, keep_frac=0.02)
    assert ratio(delta, payload) > 40         # ~50x at 2% keep


def test_deterministic():
    rng = np.random.default_rng(2)
    delta = rng.standard_normal(50_000) * 0.01
    a = compress(delta, 0.02)
    b = compress(delta, 0.02)
    assert a["idx"] == b["idx"] and a["val"] == b["val"]


def test_error_feedback_tracks_and_beats_plain_topk():
    """On training-like (correlated) deltas, error feedback carries the dropped
    components forward, so cumulative sent tracks cumulative true — and clearly
    beats plain top-k, which permanently drops signal."""
    rng = np.random.default_rng(3)
    n = 2000
    direction = rng.standard_normal(n) * 0.01           # a consistent pull (like training)
    comp = Compressor(keep_frac=0.05)
    true_sum = np.zeros(n)
    ef_sum = np.zeros(n)
    plain_sum = np.zeros(n)
    for _ in range(60):
        delta = direction + rng.standard_normal(n) * 0.003
        true_sum += dequantize(quantize(delta))
        ef_sum += dequantize(decompress(comp.compress(delta)))
        plain_sum += dequantize(decompress(compress(delta, 0.05)))
    ef_err = np.linalg.norm(true_sum - ef_sum) / np.linalg.norm(true_sum)
    plain_err = np.linalg.norm(true_sum - plain_sum) / np.linalg.norm(true_sum)
    assert ef_err < 0.25                                 # EF tracks the true signal well
    assert ef_err < 0.5 * plain_err                      # …and dramatically beats plain top-k


def test_decompress_shape_and_dtype():
    delta = np.random.default_rng(4).standard_normal(1000) * 0.01
    dense = decompress(compress(delta, 0.1))
    assert dense.shape == (1000,) and dense.dtype == np.int64
