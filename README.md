# Palimpsest

A blockchain whose state is the weights of a single public neural network. Transactions are the model's own computations: backprops transition the state and earn rewards; forward-props pay the fees that fund them. Replaying the chain reconstructs the model. The chain does not record a model — the chain *is* a model.

- **[WHITEPAPER.md](WHITEPAPER.md)** — the master design document (§1–12).
- **rig/** — the Phase 0 simulation rig (WHITEPAPER §11.3): runnable experiments that attack the design's load-bearing mechanisms and test its falsifiers (§12.6) before any real infrastructure is built.

## Quick start

```
scripts/run e2e        # the whole flywheel in one process (start here)
scripts/run node       # the same loop across real miner subprocesses / sockets
scripts/run run_all    # the falsifier suite -> results/report.md
scripts/run test       # the pytest suite (30 tests)
```

`scripts/run` uses system `numpy`/`pytest` if present, else falls back to
`uv run --with numpy`. All randomness is seeded; every run is reproducible.

### `rig/e2e.py` — the end-to-end toy chain

One node that turns every block through the complete loop and shows the pieces
working *together*: **train** (beacon-assigned miners run inner steps) →
**score** (loss impact on an unpredictable eval batch) → **apply** (deterministic
fixed-point aggregation, new state root) → **serve** (attested forward-prop
inference) → **attest** (a verifier recomputes a receipt and catches a
fake-serving node) → **pay** (fees split, training pool + emission → miner
rewards). The model behind the flywheel is a real tiny transformer
(`rig/model.py`, ~2.5k params, manual backprop, gradient-checked). Over 40
blocks it climbs from chance to 1.0 accuracy on a delayed-copy task, replays
bit-exact from genesis, distributes rewards across all miners, and slashes the
fake node every block.

### `rig/node.py` — multiprocess miners over sockets

The same block loop with miners as **separate OS processes** connected to a
coordinator over localhost TCP (length-prefixed pickle protocol,
`rig/protocol.py`). Rounds are synchronous — each block, the coordinator ships
weights + a beacon-assigned shard to every miner and waits for all deltas (the
DiLoCo outer-sync barrier). The socket run and the in-memory run produce the
**byte-identical chain**, so real multiprocess consensus stays fully
reproducible.

### `rig/async_node.py` — asynchronous miners with staleness

The synchronous barrier dropped. Heterogeneous fast/slow miners submit whenever
they finish; a delta computed against block N may not be scored until the head
is at N+lag. Staleness is handled per §4.1 — stale deltas are scored against the
*current* head (included only if still helpful), reward decays with lag, and
deltas older than `GRACE_G` are dropped. `--sim` runs a fully-seeded,
reproducible event-driven simulator; without it, real miner processes run over
sockets against a threaded, wall-clock coordinator. The model still trains to
100% under asynchrony, and fast miners out-earn slow ones as staleness discounts
late work.

### `rig/model2.py` + `rig/autograd.py` — a bigger model

A minimal reverse-mode autograd engine (`autograd.py`, gradient-checked op by
op) carries a configurable multi-layer, multi-head transformer with RMSNorm
(`model2.py`, ~68k params by default). `scripts/run model2` trains it through
the chain's DiLoCo aggregation (0.12 → 0.93 on an in-context modular-addition
task), replaying bit-exact. All rig components take the same `Model` interface,
so the chain runs either model unchanged.

### `rig/moe.py` + `rig/merkle.py` — sparse inference without loading the model

The path toward a very large model: most parameters live in **experts**, and
each query routes to only top-k of them, so serving never loads the whole
model. The chain commits a Merkle root over the weight pages (one per expert,
`rig/merkle.py`); a serving node answers a query by loading only the router +
the k selected expert pages, and returns a receipt with **Merkle inclusion
proofs for exactly those pages**. A verifier checks the proofs against the
committed root and recomputes the output touching only those pages — the other
experts are never loaded, so inference *and* attestation cost O(k), not O(E).
`scripts/run moe` trains an 8-expert model (sparse top-1 serving matches dense
accuracy), attests one query, catches a tampered page, and shows expert
capacity loaded per query falling to 0.1% at 1024 experts.

### `rig/moe_transformer.py` — the two fused, running on-chain

The deep transformer (piece above) with each FFN block replaced by a mixture of
experts, trained *through the chain*. `scripts/run moe_transformer` runs the
full loop: 6 miners train a 2-layer × 8-expert model to 1.0 accuracy via DiLoCo
aggregation, it replays bit-exact, and then a **decode step** (advancing one
token, KV-cache style) touches only top-k experts per layer — 4 of 16 here —
loading just the backbone + those 4 expert pages, attested by Merkle proofs
against the committed root (tampered pages and wrong roots rejected). Per-token
serving cost is O(top_k), not O(E): at 1024 experts/layer a decode step loads
0.2% of expert capacity. This is the concrete shape of serving a model far too
large to hold in memory.

### `rig/storage.py` — persistence & fast-sync

Blocks (delta bodies + periodic full checkpoints) persist to disk; a stopped
node restarts via **fast sync** (latest checkpoint + later deltas) and lands on
the identical state root as **full replay** from genesis.

### `rig/run_all.py` — the falsifier suite

| Experiment | Whitepaper mechanism under test | Falsifier |
|---|---|---|
| `chain.py` + `training_sim.py` | Model-as-chain-state: fixed-point deterministic apply, bit-exact replay from genesis (§3, §6.3) | — (correctness gate) |
| `training_sim.py` backdoor runs | Lock 2: stealthy poison is invisible to loss-scoring alone, caught by canary probes, reversible by replay-excision (§7.2) | #5 |
| `consensus_sim.py` | Scored mempool under attack: weight-copiers, lazy validators, colluding committees vs commit-reveal + audits + challenge window (§5) | #1 |
| `econ_sim.py` | Bootstrap tunnel: emission-funded training to revenue crossover under demand scenarios (§9.3, §9.5) | #3 |

No external dependencies beyond numpy.

*Founding doctrine: stop citing, start seeing. If the rig kills the thesis, it dies for a few GPU-days.*
