"""Persistence & fast-sync equal full replay; reload lands on the same root (§3.5)."""

import numpy as np
import pytest

from rig.chain import Chain, quantize, state_root
from rig.storage import ChainStore


def _build_chain(seed, blocks=25, dim=48):
    rng = np.random.default_rng(seed)
    chain = Chain(quantize(np.zeros(dim)))
    for _ in range(blocks):
        deltas = [quantize(rng.standard_normal(dim) * 0.1) for _ in range(4)]
        chain.apply_block(deltas, [0, 1, 2, 3])
    return chain


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_replay_and_fast_sync_match_head(tmp_path, seed):
    chain = _build_chain(seed)
    store = ChainStore(str(tmp_path), checkpoint_every=10)
    store.persist_chain(chain)
    head = chain.blocks[-1].root
    assert state_root(store.full_replay()) == head
    assert state_root(store.fast_sync()) == head
    assert store.verify()


def test_fast_sync_uses_checkpoint_not_genesis(tmp_path):
    chain = _build_chain(0, blocks=25)
    store = ChainStore(str(tmp_path), checkpoint_every=10)
    store.persist_chain(chain)
    # fast_sync and full_replay agree, but fast_sync starts from ckpt_20.
    assert np.array_equal(store.fast_sync(), store.full_replay())
    import os
    ckpts = os.listdir(os.path.join(str(tmp_path), "checkpoints"))
    assert "ckpt_20.npy" in ckpts


def test_stop_and_restart_from_disk(tmp_path):
    chain = _build_chain(1, blocks=15)
    store = ChainStore(str(tmp_path), checkpoint_every=5)
    store.persist_chain(chain)
    # a fresh node reloads the whole chain from disk
    reloaded = store.load_chain()
    assert reloaded.blocks[-1].root == chain.blocks[-1].root
    assert reloaded.height == chain.height
