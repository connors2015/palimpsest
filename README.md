# Palimpsest

A blockchain whose state is the weights of a single public neural network. Transactions are the model's own computations: backprops transition the state and earn rewards; forward-props pay the fees that fund them. Replaying the chain reconstructs the model. The chain does not record a model — the chain *is* a model.

- **[WHITEPAPER.md](WHITEPAPER.md)** — the master design document (§1–12).
- **rig/** — the Phase 0 simulation rig (WHITEPAPER §11.3): runnable experiments that attack the design's load-bearing mechanisms and test its falsifiers (§12.6) before any real infrastructure is built.

## Quick start

```
scripts/run e2e        # the whole flywheel in one process (start here)
scripts/run run_all    # the falsifier suite -> results/report.md
```

`scripts/run` uses system `numpy` if present, else falls back to `uv run --with numpy`.
All randomness is seeded; every run is reproducible.

### `rig/e2e.py` — the end-to-end toy chain

One node that turns every block through the complete loop and shows the pieces
working *together*: **train** (beacon-assigned miners) → **score** (commit-reveal
loss impact) → **apply** (deterministic fixed-point aggregation, new state root)
→ **serve** (attested forward-prop inference) → **attest** (a verifier recomputes
a receipt and catches a fake-serving node) → **pay** (fees split, training pool +
emission → miner rewards). Over 40 blocks the toy model climbs from chance to
~0.97 accuracy, replays bit-exact from genesis, distributes rewards across
honest miners, and slashes the fake node every block.

### `rig/run_all.py` — the falsifier suite

| Experiment | Whitepaper mechanism under test | Falsifier |
|---|---|---|
| `chain.py` + `training_sim.py` | Model-as-chain-state: fixed-point deterministic apply, bit-exact replay from genesis (§3, §6.3) | — (correctness gate) |
| `training_sim.py` backdoor runs | Lock 2: stealthy poison is invisible to loss-scoring alone, caught by canary probes, reversible by replay-excision (§7.2) | #5 |
| `consensus_sim.py` | Scored mempool under attack: weight-copiers, lazy validators, colluding committees vs commit-reveal + audits + challenge window (§5) | #1 |
| `econ_sim.py` | Bootstrap tunnel: emission-funded training to revenue crossover under demand scenarios (§9.3, §9.5) | #3 |

No external dependencies beyond numpy.

*Founding doctrine: stop citing, start seeing. If the rig kills the thesis, it dies for a few GPU-days.*
