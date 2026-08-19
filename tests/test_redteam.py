"""§12.3 red-team: the honest findings must hold, or the security story is wrong."""

from rig.redteam import (TH_CLEAN, TH_DELTA_Z, TH_ORACLE, TH_RANDOM, detected_by,
                         experiment_A, experiment_B_drip)


def test_stealthy_backdoor_beats_blind_input_probes():
    """A stealthy OOD-triggered backdoor is invisible to clean-loss and
    in-distribution probing, while the oracle (known trigger) always catches it."""
    A = experiment_A(seed=0)
    for a in A["results"]:
        assert a.backdoor_success > 0.8                 # the backdoor works
        assert a.p_oracle > TH_ORACLE                   # known trigger -> caught
        if a.strategy in ("stealthy", "minimal"):
            assert a.p_clean < TH_CLEAN                  # clean-loss probe misses
            assert a.p_random < TH_RANDOM                # in-distribution probe misses


def test_naive_backdoor_is_caught():
    """The crude attack that a real defender would obviously catch, is caught."""
    A = experiment_A(seed=0)
    naive = next(a for a in A["results"] if a.strategy == "naive")
    d = detected_by(naive)
    assert d["delta"] or d["oracle"]


def test_slow_drip_evades_anomaly_and_accumulates():
    """Drip coalition: each delta is no more conspicuous than honest work, yet
    the backdoor accumulates across blocks."""
    B = experiment_B_drip(seed=0)
    assert B["poisoned_backdoor"] > 0.5                  # backdoor accumulated
    assert B["max_coal_z"] < TH_DELTA_Z                  # per-delta anomaly misses it
    assert B["curve"][-1] > B["curve"][0]                # it grew over blocks


def test_excision_recovers_from_poisoning():
    """The design's durable, detection-independent guarantee: replay-excision
    removes a discovered backdoor while preserving clean accuracy (§10.4)."""
    B = experiment_B_drip(seed=0)
    assert B["excised_backdoor"] < 0.1
    assert B["excised_clean"] > B["poisoned_clean"] - 0.15
