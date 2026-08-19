"""Data-availability: erasure coding + sampling (§3.3)."""

import numpy as np
import pytest

from rig.da import (detection_probability, disperse, encode, reconstruct,
                    sample_available, verify_shard)


def _body(n=4096, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=n, dtype=np.uint8).tobytes()


@pytest.mark.parametrize("k,n", [(4, 12), (3, 6), (8, 16)])
def test_any_k_of_n_reconstructs(k, n):
    body = _body(1000)
    shards = encode(body, k, n)
    rng = np.random.default_rng(1)
    for _ in range(5):                                # several random k-subsets
        keep = sorted(rng.choice(n, size=k, replace=False).tolist())
        got = reconstruct({i: shards[i] for i in keep}, k, len(body))
        assert got == body


def test_fewer_than_k_cannot_reconstruct():
    body = _body(512)
    shards = encode(body, k := 4, n := 8)
    with pytest.raises(ValueError):
        reconstruct({i: shards[i] for i in range(k - 1)}, k, len(body))


def test_sampling_passes_when_available():
    blob = disperse(_body(2048), 4, 12)
    avail = {i: blob.shards[i] for i in range(blob.n)}
    assert sample_available(avail, blob, num_samples=4, rng=np.random.default_rng(0))


def test_sampling_detects_withholding():
    blob = disperse(_body(2048), k := 4, 12)
    withheld = {i: blob.shards[i] for i in range(k - 1)}   # unrecoverable
    caught = sum(
        not sample_available(withheld, blob, num_samples=4,
                             rng=np.random.default_rng(s))
        for s in range(200))
    assert caught == 200                              # every sample hits a hole


def test_merkle_proof_rejects_forged_shard():
    blob = disperse(_body(1024), 4, 8)
    good = blob.shards[2]
    assert verify_shard(good, 2, blob.proof(2), blob.root)
    assert not verify_shard(bytes(len(good)), 2, blob.proof(2), blob.root)


def test_detection_probability_is_high_for_small_samples():
    # unrecoverable (k-1 available of n); a few samples should almost surely catch it
    assert detection_probability(available_count=3, n=12, k=4, num_samples=4) == 1.0
    assert detection_probability(available_count=7, n=12, k=4, num_samples=3) > 0.7
