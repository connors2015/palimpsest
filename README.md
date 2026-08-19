# Palimpsest

A blockchain whose state is the weights of a single public neural network. Transactions are the model's own computations: backprops transition the state and earn rewards; forward-props pay the fees that fund them. Replaying the chain reconstructs the model. The chain does not record a model — the chain *is* a model.

- **[WHITEPAPER.md](WHITEPAPER.md)** — the master design document (§1–12).
- **rig/** — the Phase 0 simulation rig (WHITEPAPER §11.3): runnable experiments that attack the design's load-bearing mechanisms and test its falsifiers (§12.6) before any real infrastructure is built.

## The rig

```
python3 -m rig.run_all        # runs all experiments, writes results/report.md
```

| Experiment | Whitepaper mechanism under test | Falsifier |
|---|---|---|
| `rig/chain.py` + `rig/training_sim.py` | Model-as-chain-state: fixed-point deterministic apply, bit-exact replay from genesis (§3, §6.3) | — (correctness gate) |
| `rig/training_sim.py` backdoor runs | Lock 2: poisoning is invisible to loss-scoring alone, caught by canary probes, and reversible by replay-excision (§7.2) | #5 |
| `rig/consensus_sim.py` | Scored mempool under attack: weight-copiers, lazy validators, colluding committees vs commit-reveal + audits + challenge window (§5) | #1 |
| `rig/econ_sim.py` | Bootstrap tunnel: emission-funded training to revenue crossover under demand scenarios (§9.3, §9.5) | #3 |

No external dependencies beyond numpy. All randomness is seeded; every run is reproducible.

*Founding doctrine: stop citing, start seeing. If the rig kills the thesis, it dies for a few GPU-days.*
