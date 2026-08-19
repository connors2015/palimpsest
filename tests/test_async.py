"""Async miners with staleness (§4.1): trains, drops stale, reproducible."""

import numpy as np
import pytest

from rig.async_node import FAST_MINERS, GRACE_G, run_async_sim
from rig.chain import state_root


@pytest.mark.parametrize("seed", [7, 8])
def test_async_trains_despite_staleness(seed):
    chain, log = run_async_sim(ticks=120, seed=seed)
    assert log.acc[-1] > 0.9
    assert state_root(chain.replay()) == chain.blocks[-1].root


def test_async_is_reproducible():
    c1, _ = run_async_sim(ticks=80, seed=7)
    c2, _ = run_async_sim(ticks=80, seed=7)
    assert c1.blocks[-1].root == c2.blocks[-1].root


def test_staleness_is_real_not_barriered():
    _, log = run_async_sim(ticks=120, seed=7)
    # Some deltas are dropped for exceeding the grace window, and included
    # deltas carry nonzero lag — i.e., this is genuinely asynchronous.
    assert log.stale_dropped > 0
    assert max(log.included_lags) >= 1
    assert all(lag <= GRACE_G for lag in log.included_lags)  # never over grace


def test_staleness_discount_favors_fast_miners():
    _, log = run_async_sim(ticks=120, seed=7)
    fast = sum(v for m, v in log.rewards.items() if m < FAST_MINERS)
    slow = sum(v for m, v in log.rewards.items() if m >= FAST_MINERS)
    assert fast > slow  # slow miners' stale work is discounted/dropped (§4.1)
