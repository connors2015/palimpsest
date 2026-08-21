//! The capacity retarget (§9.4a) — model size as difficulty.
//!
//! A faithful port of `rig/capacity.py`, pinned by golden vectors. Two-timescale
//! controller over chain-observable signals only:
//!
//!   * FAST knob — the per-delta work quota, damped toward holding accepted
//!     delta count at target (the Bitcoin-difficulty analog: continuous,
//!     reversible).
//!   * SLOW knob — parameter growth. Only a SUSTAINED ceiling-pinned surplus
//!     (fast knob maxed for `k_sustain` windows) grows the model, bounded per
//!     event and ratcheted (total never shrinks). When compute leaves, ACTIVE
//!     modules freeze (stop training, keep serving) rather than the model
//!     shrinking — MoE sparsity's graceful degradation.
//!
//! Deterministic: only +,-,*,/,min,max and comparisons in a fixed order, so
//! every node computes the identical decision trace (this is consensus code in
//! spirit; the real trigger reads the same counts from chain history).

/// The decision taken for one retarget window (mirrors the Python `events`).
#[derive(Clone, Debug, PartialEq)]
pub struct WindowDecision {
    pub window: u64,
    pub grew: u64,
    pub froze: u64,
    pub thawed: u64,
    pub quota: f64, // rounded to 4 dp, as the reference reports it
    pub total: u64,
    pub active: u64,
}

pub struct CapacityRetarget {
    // fast knob
    pub quota: f64,
    quota_min: f64,
    quota_max: f64,
    target_deltas: u64,
    damp: f64,
    stale_ceiling: f64,
    // slow knob
    k_sustain: u64,
    growth_bound: u64,
    announce_lead: u64,
    pub total_modules: u64,
    pub active_modules: u64,
    min_active: u64,
    // internal
    pinned_streak: u64,
    slack_streak: u64,
    pending_growth: Vec<u64>,
    pub window_id: u64,
}

impl Default for CapacityRetarget {
    fn default() -> Self {
        CapacityRetarget {
            quota: 1.0,
            quota_min: 0.25,
            quota_max: 8.0,
            target_deltas: 8,
            damp: 0.25,
            stale_ceiling: 0.2,
            k_sustain: 3,
            growth_bound: 1,
            announce_lead: 2,
            total_modules: 4,
            active_modules: 4,
            min_active: 1,
            pinned_streak: 0,
            slack_streak: 0,
            pending_growth: Vec::new(),
            window_id: 0,
        }
    }
}

impl CapacityRetarget {
    /// Feed one window's chain-observable signals; returns the deterministic
    /// decisions taken. Line-for-line mirror of the reference `observe_window`.
    pub fn observe_window(&mut self, accepted: u64, staleness: f64) -> WindowDecision {
        self.window_id += 1;
        let mut grew = 0u64;
        let (mut froze, mut thawed) = (0u64, 0u64);

        // activate any growth event whose announcement lead has elapsed
        let due = self.pending_growth.iter().filter(|&&w| w <= self.window_id).count();
        for _ in 0..due {
            self.total_modules += self.growth_bound;
            self.active_modules += self.growth_bound;
            grew += self.growth_bound;
        }
        self.pending_growth.retain(|&w| w > self.window_id);

        // FAST: damped quota correction toward the target delta rate
        let raw = if accepted > 0 {
            self.quota * (accepted as f64 / self.target_deltas as f64)
        } else {
            self.quota_min
        };
        self.quota += self.damp * (raw - self.quota);
        self.quota = self.quota_max.min(self.quota_min.max(self.quota));

        // SLOW: surplus/deficit bands (wide, because the damped quota asymptotes
        // to its bounds without exactly reaching them)
        let surplus = self.quota >= self.quota_max * 0.95
            && staleness <= self.stale_ceiling
            && accepted >= self.target_deltas;
        let deficit = self.quota <= self.quota_min * 1.05 && accepted < self.target_deltas;
        self.pinned_streak = if surplus { self.pinned_streak + 1 } else { 0 };
        self.slack_streak = if deficit { self.slack_streak + 1 } else { 0 };

        if self.pinned_streak >= self.k_sustain && self.pending_growth.is_empty() {
            self.pending_growth.push(self.window_id + self.announce_lead);
            self.pinned_streak = 0;
            // growth resets the fast knob to mid-band
            self.quota = (self.quota_min + self.quota_max) / 2.0;
        }

        // decline: freeze active modules (total NEVER shrinks — the ratchet)
        if self.slack_streak >= self.k_sustain && self.active_modules > self.min_active {
            self.active_modules -= 1;
            froze = 1;
            self.slack_streak = 0;
        } else if surplus && self.active_modules < self.total_modules {
            // recovery: thaw a frozen module before any new growth is considered
            self.active_modules += 1;
            thawed = 1;
            self.pinned_streak = 0; // thawing consumes the surplus signal
        }

        WindowDecision {
            window: self.window_id,
            grew,
            froze,
            thawed,
            quota: round4(self.quota),
            total: self.total_modules,
            active: self.active_modules,
        }
    }
}

/// Round half-to-even to 4 decimal places, matching Python's `round(x, 4)`.
fn round4(x: f64) -> f64 {
    let scaled = x * 10_000.0;
    let r = scaled.round_ties_even();
    r / 10_000.0
}
