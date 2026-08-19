"""Bootstrap-tunnel economics (WHITEPAPER §9.3, §9.5; falsifier §12.6 #3).

Monthly simulation over 8 years. Emission funds bootstrap training, then the
milestone sunset steps it down (monotone, never up — §9.3). The question per
scenario: does fee-funded training (ARR x training share) cross emissions
("the crossover block") before the emission cap is exhausted, and what does
the cumulative subsidy peak at?
"""

from dataclasses import dataclass

import numpy as np

MONTHS = 96
TRAINING_SHARE = 0.25          # fee split to the training pool (§9.2)
STEADY_BURN_YR = 1.0e6         # continuous post-training budget at maturity (§9.5)
EMISSION_CAP = 20.0e6          # hard cap on total issuance value (§9.3)
SUNSET_STEP = 0.75             # emission *= 0.75 per milestone (§9.3)
MILESTONE_RATIO = 0.5          # milestone: trailing fees-to-training >= 50% of emission


@dataclass
class Scenario:
    name: str
    arr_cap: float       # ARR asymptote ($/yr)
    t50: float           # months to half of asymptote (logistic)


SCENARIOS = [
    Scenario("conservative", 1.5e6, 30.0),
    Scenario("base", 4.0e6, 24.0),
    Scenario("aggressive", 10.0e6, 18.0),
]


@dataclass
class TunnelResult:
    scenario: str
    bootstrap_cost: float
    crossover_month: int | None   # first month fees-to-training >= emission
    cumulative_emission: float    # total subsidy spent by month 96
    cap_hit: bool
    sunset_steps_taken: int


def arr(scenario: Scenario, month: int) -> float:
    return scenario.arr_cap / (1.0 + np.exp(-(month - scenario.t50) / 6.0))


def simulate(scenario: Scenario, bootstrap_cost: float,
             bootstrap_months: int = 15) -> TunnelResult:
    emission_rate = bootstrap_cost / bootstrap_months  # $/mo during bootstrap
    cumulative = 0.0
    crossover = None
    steps = 0

    fees_hist = []
    for m in range(MONTHS):
        fees_to_training = arr(scenario, m) / 12.0 * TRAINING_SHARE
        fees_hist.append(fees_to_training)

        # After bootstrap, emission targets covering the steady burn, then sunsets.
        if m == bootstrap_months:
            emission_rate = STEADY_BURN_YR / 12.0
        if m > bootstrap_months and m % 3 == 0:
            trailing = float(np.mean(fees_hist[-3:]))
            if trailing >= MILESTONE_RATIO * emission_rate:
                emission_rate *= SUNSET_STEP  # monotone step-down (§9.3)
                steps += 1

        if cumulative + emission_rate <= EMISSION_CAP:
            cumulative += emission_rate
        else:
            emission_rate = max(0.0, EMISSION_CAP - cumulative)
            cumulative = EMISSION_CAP

        # A genuine crossover requires *live* emission being outgrown by fees
        # while the cap is not yet exhausted — emission dying at the cap while
        # fees stay small is failure, not victory.
        if (crossover is None and m > bootstrap_months and emission_rate > 0
                and cumulative < EMISSION_CAP
                and fees_to_training >= emission_rate):
            crossover = m

    return TunnelResult(scenario.name, bootstrap_cost, crossover,
                        cumulative, cumulative >= EMISSION_CAP, steps)


def run_econ_suite() -> list[TunnelResult]:
    results = []
    for scenario in SCENARIOS:
        for cost in (5e6, 10e6, 15e6):
            results.append(simulate(scenario, cost))
    return results
