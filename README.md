# Palimpsest

A blockchain whose state is the weights of a single public neural network.
Transactions are the model's own computations: backprops transition the state
and earn rewards; forward-props pay the fees that fund them. Replaying the
chain reconstructs the model bit-for-bit. The chain does not record a model —
the chain *is* a model.

Data is a priced, owned input: contributors stake behind submissions, earn the
block data share, and face an on-chain challenge market. The token is native
chain state — fair-launched, emissions halving to a hard sunset, every grain
minted by verifiable work.

> **Status — honest map.** This is an active Phase-0/1 project. What is
> *enforced in the shipping node today* versus what is *designed or testnet-phase*
> is tracked precisely in **[docs/production-readiness.md](docs/production-readiness.md)**
> and **[docs/internal/threat-model.md](docs/internal/threat-model.md)**. In
> short: all consensus-safety and runtime hardening, verifiable VRF proposer
> sortition + non-forgeable work, Byzantine-robust aggregation, erasure-coded
> multi-node data availability, the stake-bond admission cost, and fee-bearing
> inference are implemented and pinned to the reference by golden vectors. Delta
> loss-scoring, cross-inclusion, and the threshold-BLS beacon remain testnet-phase
> (they need live compute + a running network). The network is **open** — anyone
> can join from one peer address + the published genesis id (see
> **[docs/joining.md](docs/joining.md)**). Launch is phased: a small monitored
> devnet → testnet → open mainnet.

## The map

| Path | What it is |
|---|---|
| **[WHITEPAPER.md](WHITEPAPER.md)** | the master design document (§1–12) — invariants: bytes-only interface, RoPE positions, from-scratch genesis, fair launch |
| **client/** | the Python client: real PyTorch GPT trained *through the chain* (gossip consensus, DiLoCo deltas, 50× compression), the chain watcher web UI (`watch.py`), the wallet, DiPaCo sharding, content-addressed storage |
| **rig/** | the reference implementation — consensus, token ledger + data lane, DA, beacon, economics; the SPEC the Rust node must match |
| **node/** | the Rust node: `palimpsest-core` (bit-exact consensus, pinned to the reference by golden vectors) + `palimpsest-node` (libp2p GossipSub/QUIC networking) |
| **docs/** | including **[genesis-ceremony.md](docs/genesis-ceremony.md)** — how the real network launches |
| **deploy/** | the bootstrap seed node (Kubernetes) |

## Quick starts

```bash
# watch a live local chain in your browser (blocks, loss falling, chat with the model):
uv run --with torch --with numpy --with pynacl python -m client.watch --demo

# a wallet (encrypted file + BIP39 mnemonic + pal1… checksummed address):
pip install pynacl mnemonic bech32 && python -m client.wallet new

# the Rust devnet — three nodes gossip, validate, and converge over libp2p:
scripts/devnet.sh 30

# the reference test suites:
uv run --with torch --with numpy --with pynacl --with py_ecc --with pytest \
    python -m pytest tests/ -q          # Python reference (163 tests)
cd node && cargo test                   # Rust vs golden vectors (13 families)
```

## Status

Phase 0/1 — mechanisms proven, pre-launch. Verified to date: real cross-GPU
training through the chain (Apple MPS + CUDA, no coordinator, byte-identical
heads); an 86M-parameter model trained from scratch *on-chain*; the token
economy live end-to-end (mining rewards, transfers, staked data submissions,
challenges — all committed by ledger roots in block headers); the Rust node
devnet converging bit-exactly with the Python reference. Before mainnet:
NAT traversal for public volunteers, external audit, legal counsel on the
token, and the genesis ceremony (docs/genesis-ceremony.md).

## License

MIT — see [LICENSE](LICENSE).
