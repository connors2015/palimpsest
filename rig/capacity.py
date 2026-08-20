"""The capacity retarget (§9.4a) — model size as difficulty, mechanism-proof.

Two-timescale controller over chain-observable signals only:

  * FAST knob: the per-delta work quota (inner steps × shard size a valid delta
    must embody). Damped adjustment per retarget window toward holding accepted
    delta count at target — the literal Bitcoin-difficulty analog, continuous
    and reversible.
  * SLOW knob: parameter growth events. Only when the fast knob has been pinned
    at its ceiling for k_sustain consecutive windows (a SUSTAINED surplus) does
    the model grow — bounded modules per event, ratcheted (total capacity never
    shrinks). When compute leaves, ACTIVE modules freeze (stop training, keep
    serving) instead of shrinking the model — MoE sparsity's graceful
    degradation.

Deterministic: the same observation trace yields the same decision trace on
every node (this is consensus code in spirit; the real trigger reads the same
numbers from chain history).
"""

from dataclasses import dataclass, field


@dataclass
class CapacityRetarget:
    # fast knob
    quota: float = 1.0             # work units a valid delta must embody
    quota_min: float = 0.25
    quota_max: float = 8.0
    target_deltas: int = 8         # accepted deltas per window we steer toward
    damp: float = 0.25             # fraction of the raw correction applied
    stale_ceiling: float = 0.2     # above this staleness, never count as surplus
    # slow knob
    k_sustain: int = 3             # windows pinned at ceiling before growth
    growth_bound: int = 1          # max modules added per event
    announce_lead: int = 2         # windows between decision and activation
    total_modules: int = 4
    active_modules: int = 4
    min_active: int = 1
    # internal
    pinned_streak: int = 0
    slack_streak: int = 0
    pending_growth: list = field(default_factory=list)   # activation window ids
    window_id: int = 0
    log: list = field(default_factory=list)

    def observe_window(self, accepted: int, staleness: float) -> dict:
        """Feed one retarget window's chain-observable signals; returns the
        decisions taken this window (all deterministic)."""
        self.window_id += 1
        events = {"window": self.window_id, "grew": 0, "froze": 0, "thawed": 0}

        # activate any growth event whose announcement lead has elapsed
        due = [w for w in self.pending_growth if w <= self.window_id]
        for _ in due:
            self.total_modules += self.growth_bound
            self.active_modules += self.growth_bound
            events["grew"] += self.growth_bound
        self.pending_growth = [w for w in self.pending_growth if w > self.window_id]

        # FAST: damped quota correction toward the target delta rate
        if accepted > 0:
            raw = self.quota * (accepted / self.target_deltas)
        else:
            raw = self.quota_min
        self.quota += self.damp * (raw - self.quota)
        self.quota = min(self.quota_max, max(self.quota_min, self.quota))

        # SLOW: surplus = ceiling-pinned AND healthy staleness AND target met.
        # Band tolerances are wide (5%) because the damped quota asymptotes to
        # its bounds without ever exactly reaching them.
        surplus = (self.quota >= self.quota_max * 0.95
                   and staleness <= self.stale_ceiling
                   and accepted >= self.target_deltas)
        deficit = (self.quota <= self.quota_min * 1.05
                   and accepted < self.target_deltas)
        self.pinned_streak = self.pinned_streak + 1 if surplus else 0
        self.slack_streak = self.slack_streak + 1 if deficit else 0

        if self.pinned_streak >= self.k_sustain and not self.pending_growth:
            self.pending_growth.append(self.window_id + self.announce_lead)
            self.pinned_streak = 0
            # growth resets the fast knob to mid-band: the bigger model absorbs
            # the surplus the quota ceiling could not
            self.quota = (self.quota_min + self.quota_max) / 2

        # decline: freeze active modules (total NEVER shrinks — the ratchet)
        if self.slack_streak >= self.k_sustain and self.active_modules > self.min_active:
            self.active_modules -= 1
            events["froze"] = 1
            self.slack_streak = 0
        # recovery: thaw frozen modules before any new growth is considered
        elif surplus and self.active_modules < self.total_modules:
            self.active_modules += 1
            events["thawed"] = 1
            self.pinned_streak = 0        # thawing consumes the surplus signal

        events.update(quota=round(self.quota, 4), total=self.total_modules,
                      active=self.active_modules)
        self.log.append(events)
        return events


def simulate(controller: CapacityRetarget, fleet_trace: list[float],
             per_unit: float = 8.0) -> list[dict]:
    """Drive the controller with a synthetic fleet: each window the fleet's
    compute produces ~ fleet/quota deltas (capped by staleness effects when the
    ACTIVE model outweighs the fleet). Deterministic — no randomness."""
    out = []
    modules_per_fleet_unit = 4.0        # a fleet of 1.0 comfortably trains 4 modules
    for fleet in fleet_trace:
        capacity = fleet * per_unit
        accepted = int(capacity / max(controller.quota, 1e-9))
        # if the active model outweighs the fleet, deltas arrive late
        load = controller.active_modules / (modules_per_fleet_unit * max(fleet, 1e-9))
        staleness = max(0.0, min(1.0, load - 1.0))
        accepted = int(accepted * (1.0 - staleness))
        out.append(controller.observe_window(accepted, staleness))
    return out
