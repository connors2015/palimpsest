"""Write-price homeostat + stake/slash ledger (WHITEPAPER §9.4, §7, §9).

Two Bitcoin-inspired economic mechanisms the rig otherwise faked:

1. WritePriceController — a difficulty-adjustment controller. Bitcoin retargets
   difficulty every 2016 blocks by old × (target_timespan / actual_timespan),
   clamped to [0.25×, 4×], to hold block time at 10 minutes against changing
   hash power. Here the same math holds the *delta-admission rate* at target
   against changing submission pressure: when more deltas are offered than the
   committee can score well, the write price (stake bond to submit) rises and
   prices out the marginal — and the spam — first; when the network is idle it
   falls back toward the floor. Spam, sybil grinding, and low-effort floods are
   all the same phenomenon (interference) priced by the same controller.

2. StakeLedger — real stake accounting with slashing on *provable* faults (a
   bad signature, a withheld/forged DA body, a fraudulent score). Slashed stake
   pays the challenger who proved the fault; the rest burns. This is the
   economic backing that makes the challenge games of §5/§7 bite.
"""

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 1. Write-price homeostat (§9.4)
# --------------------------------------------------------------------------
class WritePriceController:
    """Bitcoin-difficulty-style retarget, with one deliberate difference. Bitcoin
    multiplies difficulty by (actual/target) clamped to [0.25×, 4×] because its
    plant — hash power — varies smoothly. Delta submission responds as a sharp
    threshold (a submitter is in or out), so an undamped proportional step
    oscillates; we damp the move (a fractional gain) and clamp tighter. Same
    controller shape as Bitcoin, tuned for a sharper plant (§9.4)."""

    def __init__(self, target_rate, window=5, clamp=2.0, gain=0.5, min_price=1.0):
        self.target = target_rate      # desired admitted deltas per block
        self.window = window           # retarget interval (Bitcoin: 2016 blocks)
        self.clamp = clamp             # max per-retarget move
        self.gain = gain               # damping exponent (Bitcoin ≈ 1.0)
        self.min_price = min_price
        self.price = float(min_price)
        self.loads = []
        self.price_history = [self.price]

    def observe(self, admitted: int):
        self.loads.append(admitted)

    def maybe_retarget(self):
        """Every `window` blocks, move price toward holding load at target."""
        if len(self.loads) < self.window or len(self.loads) % self.window != 0:
            return
        actual = sum(self.loads[-self.window:]) / self.window
        ratio = (actual / self.target if self.target else 1.0) ** self.gain
        ratio = max(1.0 / self.clamp, min(self.clamp, ratio))    # clamp like Bitcoin
        self.price = max(self.min_price, self.price * ratio)
        self.price_history.append(self.price)


def simulate_homeostat(target=8, blocks=80, honest_values=None, n_spam=40, seed=0):
    """Closed loop: submitters offer a delta if their expected reward exceeds the
    current write price. Honest submitters have a spread of value; spam has ≈0.
    The controller should drive admitted load to `target` and price out spam."""
    import numpy as np
    rng = np.random.default_rng(seed)
    if honest_values is None:
        honest_values = list(16.0 * rng.random(30) + 6.0)        # value 6..22
    spam_values = [3.0 * rng.random() for _ in range(n_spam)]    # low, some > floor
    ctrl = WritePriceController(target_rate=target)
    loads, spam_admitted = [], []
    for _ in range(blocks):
        admitted = [v for v in honest_values if v > ctrl.price]
        spam_in = [v for v in spam_values if v > ctrl.price]
        ctrl.observe(len(admitted) + len(spam_in))
        ctrl.maybe_retarget()
        loads.append(len(admitted) + len(spam_in))
        spam_admitted.append(len(spam_in))
    return dict(controller=ctrl, loads=loads, spam_admitted=spam_admitted,
                final_load=loads[-1], final_price=ctrl.price,
                final_spam=spam_admitted[-1])


# --------------------------------------------------------------------------
# 2. Stake / slash ledger (§7, §9)
# --------------------------------------------------------------------------
@dataclass
class SlashEvent:
    offender: str
    amount: float
    reason: str
    challenger: str


class StakeLedger:
    """Slashable stake per pubkey. Slashing on a proven fault pays the challenger
    a bounty and burns the rest — the economics behind the challenge games."""

    def __init__(self, bounty_share=0.5):
        self.staked = {}
        self.rewards = {}
        self.burned = 0.0
        self.bounty_share = bounty_share
        self.events = []

    def stake(self, pk, amount):
        self.staked[pk] = self.staked.get(pk, 0.0) + amount

    def unstake(self, pk, amount):
        if self.staked.get(pk, 0.0) < amount:
            raise ValueError("insufficient stake")
        self.staked[pk] -= amount

    def reward(self, pk, amount):
        self.rewards[pk] = self.rewards.get(pk, 0.0) + amount

    def slash(self, offender, reason, challenger, amount=None):
        """Slash `offender` for a proven fault; pay `challenger` a bounty, burn rest.
        Slashing the whole bond by default — provable faults are not fee events."""
        bond = self.staked.get(offender, 0.0)
        amount = bond if amount is None else min(amount, bond)
        if amount <= 0:
            return 0.0
        self.staked[offender] = bond - amount
        bounty = amount * self.bounty_share
        self.reward(challenger, bounty)
        self.burned += amount - bounty
        self.events.append(SlashEvent(offender, amount, reason, challenger))
        return bounty


# --------------------------------------------------------------------------
# Provable-fault detectors that trigger slashing (tie into §5/§7)
# --------------------------------------------------------------------------
def slash_on_invalid_tx(ledger: StakeLedger, tx, body, challenger: str) -> bool:
    """A watcher proves a submitted tx is invalid (bad signature or a DA body
    that doesn't match its committed hash) and slashes the submitter."""
    from .crypto import delta_hash
    if not tx.verify():
        ledger.slash(tx.miner, "invalid signature", challenger)
        return True
    if body is not None and delta_hash(body.tobytes()) != tx.delta_hash:
        ledger.slash(tx.miner, "DA body mismatch / withholding", challenger)
        return True
    return False


def slash_on_fraudulent_score(ledger: StakeLedger, validator: str, claimed: float,
                              recomputed: float, challenger: str, tol=1e-6) -> bool:
    """A validator's revealed score is recomputed (§5.3); a deviation beyond
    arithmetic tolerance is provable fraud and is slashed."""
    if abs(claimed - recomputed) > tol:
        ledger.slash(validator, "fraudulent score", challenger)
        return True
    return False


def main():
    print("=" * 68)
    print("  SESTRIAN — write-price homeostat + stake/slash")
    print("=" * 68)
    import numpy as np
    r = simulate_homeostat(target=8, blocks=120)
    settled = np.mean(r["loads"][-20:])
    print(f"\nwrite-price homeostat (target admitted/block = 8):")
    print(f"  load: {r['loads'][0]} (unpriced) -> {settled:.1f} (settled, last-20 mean)")
    print(f"  price: 1.0 (floor) -> {r['final_price']:.1f}")
    print(f"  spam admitted per block: {r['spam_admitted'][0]} -> {r['final_spam']} "
          f"(priced out)")

    print(f"\nstake / slash:")
    led = StakeLedger()
    for who in ("alice", "bob", "mallory"):
        led.stake(who, 100.0)
    led.reward("alice", 12.0)
    bounty = led.slash("mallory", "DA withholding (proven by watcher)", "watcher-carol")
    print(f"  alice stake {led.staked['alice']:.0f}, reward {led.rewards['alice']:.0f}")
    print(f"  mallory slashed to {led.staked['mallory']:.0f}; "
          f"watcher-carol earned bounty {bounty:.0f}; burned {led.burned:.0f}")
    print(f"  slash events: {[(e.offender, e.reason) for e in led.events]}")
    print("=" * 68)
    ok = (abs(settled - 8) <= 2 and r["final_spam"] == 0
          and led.staked["mallory"] == 0.0 and bounty > 0)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
