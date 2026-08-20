"""Capacity retarget (§9.4a): the fast knob holds the delta rate as compute
grows, growth fires only on SUSTAINED surplus (never a transient spike), the
total ratchets monotonically while the active set breathes, and the whole
controller is deterministic."""

from rig.capacity import CapacityRetarget, simulate


def test_fast_knob_absorbs_fleet_growth():
    c = CapacityRetarget()
    # fleet doubles: quota should rise (harder deltas), not the model (yet)
    simulate(c, [1.0] * 6 + [2.0] * 6)
    assert c.log[-1]["quota"] > c.log[5]["quota"]
    assert c.total_modules == 4                     # no growth without saturation


def test_growth_only_on_sustained_surplus():
    c = CapacityRetarget()
    # transient one-window spike: must NOT grow
    simulate(c, [1.0] * 5 + [50.0] + [1.0] * 5)
    assert c.total_modules == 4
    # sustained large fleet: quota pins at ceiling -> growth event fires,
    # bounded and after the announcement lead
    c2 = CapacityRetarget()
    log = simulate(c2, [1.0] * 3 + [50.0] * 20)
    assert c2.total_modules > 4
    grew_windows = [e["window"] for e in log if e["grew"]]
    assert grew_windows, "sustained surplus must grow the model"
    # bounded: each event adds at most growth_bound
    assert all(e["grew"] <= c2.growth_bound for e in log)


def test_ratchet_and_elastic_active_set():
    c = CapacityRetarget()
    simulate(c, [50.0] * 25)                        # grow under a big fleet
    grown_total = c.total_modules
    assert grown_total > 4
    grown_total_after_pending = grown_total + len(c.pending_growth) * c.growth_bound
    simulate(c, [0.05] * 30)                        # fleet collapses
    # TOTAL is monotone (ratchet): never shrinks — it may still tick up once
    # from a growth event announced during the boom (announcement lead)
    assert grown_total <= c.total_modules <= grown_total_after_pending
    assert c.active_modules < c.total_modules       # ACTIVE froze instead
    assert c.active_modules >= c.min_active
    frozen = c.active_modules
    simulate(c, [50.0] * 10)                        # fleet returns
    assert c.active_modules > frozen                # thaw before new growth


def test_deterministic():
    trace = [1.0] * 4 + [30.0] * 12 + [0.2] * 8 + [10.0] * 6
    a = CapacityRetarget()
    b = CapacityRetarget()
    la, lb = simulate(a, trace), simulate(b, trace)
    assert la == lb                                 # same trace -> same decisions
    assert (a.total_modules, a.active_modules, a.quota) == \
        (b.total_modules, b.active_modules, b.quota)
