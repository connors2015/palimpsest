"""Multiprocess node: transports agree bit-for-bit; sockets train; persists (§3, §6)."""

import numpy as np
import pytest

from rig.chain import state_root
from rig.node import SocketCoordinator, run_in_memory
from rig.storage import ChainStore


@pytest.mark.parametrize("seed", [7, 11])
def test_in_memory_node_trains(seed):
    chain, log = run_in_memory(blocks=30, seed=seed)
    assert log.acc[-1] > 0.9
    assert state_root(chain.replay()) == chain.blocks[-1].root


def test_socket_and_in_memory_produce_identical_chain():
    mem_chain, mem_log = run_in_memory(blocks=15, seed=7)
    sock_chain, sock_log = SocketCoordinator(blocks=15, seed=7).run()
    # Synchronous barrier + seeded beacon => real processes, identical state.
    assert sock_chain.blocks[-1].root == mem_chain.blocks[-1].root
    assert sock_log.acc == mem_log.acc


def test_node_persists_and_fast_syncs(tmp_path):
    store = ChainStore(str(tmp_path), checkpoint_every=10)
    chain, _ = run_in_memory(blocks=25, seed=7, store=store)
    head = chain.blocks[-1].root
    assert state_root(store.full_replay()) == head
    assert state_root(store.fast_sync()) == head


def test_all_miners_earn_rewards():
    _, log = run_in_memory(blocks=30, seed=7)
    from rig.node import N_MINERS
    assert set(log.rewards) == set(range(N_MINERS))
    assert all(v > 0 for v in log.rewards.values())
