"""The observability harness: converges when healthy, aborts fast when not."""

from rig.moe_transformer import MoETConfig, MoETransformer
from rig.observe import Monitor, run_observed

SMALL = MoETConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_experts=4, top_k=2)


def test_healthy_run_converges_without_false_abort():
    """A slow/grokking run (long early zero-inclusion stretch) must NOT be killed."""
    m = MoETransformer(SMALL)
    r = run_observed(m, blocks=70, seed=7, stream=False)
    assert not r["aborted"]
    assert r["acc1"] > 0.7 and r["acc1"] > r["acc0"]
    assert r["replay_ok"]
    assert 0.0 < r["verify_overhead"] < 1.0        # a real overhead fraction
    assert r["da_ratio"] > 1                        # delta bodies dominate on-chain bytes


def test_stuck_run_aborts_early():
    """A wrong step size that never lands a delta is caught and aborted well
    before the run would finish."""
    m = MoETransformer(SMALL)
    r = run_observed(m, blocks=80, lr=1000.0, seed=7, stream=False, abort=True)
    assert r["aborted"]                             # fail-fast fired
    assert r["blocks"] < 40                         # well before the end


def test_monitor_flags_nan_and_explosion():
    from rig.observe import BlockStat
    mon = Monitor()
    mon.update(BlockStat(1, 0.5, float("nan"), 1.0, 4, 4, 0.1, 0.02, 0, 0.1))
    ok, why = mon.health()
    assert not ok and "divergence" in why

    mon2 = Monitor()
    mon2.update(BlockStat(1, 0.5, 1.0, 1e5, 4, 4, 0.1, 0.02, 0, 0.1))
    ok2, why2 = mon2.health()
    assert not ok2 and "exploding" in why2
