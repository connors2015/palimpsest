"""Scored-mempool consensus under attack (WHITEPAPER §5; falsifier §12.6 #1).

Monte Carlo over blocks. Per block, a committee of size M is sampled from
global stake. Deltas receive a protocol score = median of committee reveals.
Strategies:

  honest     — pays the evaluation cost, reveals the deterministic true score.
  freerider  — weight-copier/lazy validator: skips evaluation, guesses.
               Under commit-reveal there is no median to copy pre-reveal
               (§5.3), and any reveal deviating from the exactly-recomputable
               true score is provable fraud, caught with per-block audit
               probability `audit_rate` and slashed.
  colluder   — coalition holding global stake fraction g. If it captures the
               committee majority, it force-includes its own garbage delta and
               collects that block's miner reward (self-dealing). Colluders'
               scores are fraud-provable like any deviation; an included
               garbage delta is additionally evicted (rewards clawed back,
               scorers slashed) if at least one honest watcher challenges
               within the window (§5.4), which happens with prob `challenge_p`.

The sim asks one question per strategy: is deviating from honesty profitable,
and at what global stake fraction does anything break?
"""

from dataclasses import dataclass

import numpy as np

# Economic parameters, expressed in units of the per-block validator reward R.
R_VALIDATOR = 1.0        # validator payment per block served on a committee
C_EVAL = 0.10            # honest evaluation cost (forward passes are cheap, §6.3)
STAKE_SLASH = 25.0       # slashable bond, in R units (stake >> per-block reward)
MINER_REWARD = 8.0       # per-block training reward pool captured by self-dealing
COMMITTEE = 31           # committee size M (odd for clean medians)


@dataclass
class Verdict:
    strategy: str
    param: float
    ev_honest: float
    ev_attacker: float
    state_safety_failures: float  # fraction of blocks a bad delta became final
    challenge_p: float = 1.0      # watcher coverage used in this run (§5.4)


def freerider_ev(audit_rate: float, blocks: int = 20000, seed: int = 11) -> Verdict:
    """EV per committee-block: honest vs non-evaluating validator."""
    rng = np.random.default_rng(seed)
    ev_h = R_VALIDATOR - C_EVAL
    audited = rng.random(blocks) < audit_rate
    # A guessed reveal of a continuous deterministic quantity deviates ~always;
    # every audited block with a deviating reveal is a slash (§5.3).
    ev_f = float(np.mean(np.where(audited, R_VALIDATOR - STAKE_SLASH, R_VALIDATOR)))
    return Verdict("freerider", audit_rate, ev_h, ev_f, 0.0)


def colluder_ev(global_stake: float, audit_rate: float = 0.05,
                challenge_p: float = 0.95, blocks: int = 20000,
                seed: int = 13) -> Verdict:
    """EV per block for a coalition with global stake fraction g."""
    rng = np.random.default_rng(seed)
    # Committee capture: binomial sampling from global stake (§5.3).
    seats = rng.binomial(COMMITTEE, global_stake, size=blocks)
    captured = seats > COMMITTEE // 2

    profit = np.zeros(blocks)
    finalized_bad = 0

    # Non-captured blocks: colluders behave honestly (best case for them).
    profit += seats * (R_VALIDATOR - C_EVAL)

    # Captured blocks: force-include garbage delta for MINER_REWARD.
    challenged = rng.random(blocks) < challenge_p
    audited = rng.random(blocks) < audit_rate

    for i in np.nonzero(captured)[0]:
        gain = MINER_REWARD + seats[i] * R_VALIDATOR
        if challenged[i]:
            # Eviction: reward clawed back, majority of coalition seats slashed (§5.4).
            gain = -seats[i] * STAKE_SLASH
        elif audited[i]:
            # Even unchallenged, deviating scores are audit-provable fraud (§5.3).
            gain = MINER_REWARD - seats[i] * STAKE_SLASH
        else:
            finalized_bad += 1
        profit[i] = gain

    ev_h = float(np.mean(seats) * (R_VALIDATOR - C_EVAL))
    return Verdict("colluder", global_stake, ev_h, float(np.mean(profit)),
                   finalized_bad / blocks, challenge_p)


def run_consensus_suite() -> dict:
    freeriders = [freerider_ev(a) for a in (0.01, 0.02, 0.05, 0.10, 0.20)]
    # Break-even audit rate: freeriding profitable iff audit_rate * SLASH < C_EVAL.
    breakeven = C_EVAL / STAKE_SLASH

    colluders = [colluder_ev(g) for g in (0.05, 0.15, 0.30, 0.45, 0.51, 0.60, 0.70)]
    # Sensitivity: how much watcher coverage does state safety actually need?
    weak_watch = [colluder_ev(0.60, challenge_p=p) for p in (0.0, 0.25, 0.5, 0.75, 0.95)]

    return {
        "freeriders": freeriders,
        "audit_breakeven": breakeven,
        "colluders": colluders,
        "weak_watchers_at_60pct": weak_watch,
        "params": dict(R=R_VALIDATOR, c_eval=C_EVAL, slash=STAKE_SLASH,
                       miner_reward=MINER_REWARD, committee=COMMITTEE),
    }
