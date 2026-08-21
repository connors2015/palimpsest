# Joining the Palimpsest devnet

The network is **open** — permissionless, like Bitcoin. To join you need exactly
three things, all public:

1. **the software** (this repo),
2. **one bootstrap peer address** (a running node's multiaddr), and
3. **the published genesis id** (a 32-byte hash).

Your node dials the peer, **fetches the genesis from it, and verifies it against
the published id** (so a bad peer can't seed you a wrong chain), then syncs. No
genesis file to download, no seed to reproduce.

> Live devnet parameters:
>
> | | value |
> |---|---|
> | bootstrap peer | `/ip4/169.58.211.248/udp/9800/quic-v1` |
> | genesis id | `30ea20da27f1da0c94512d50a6291370a63a426b77dc425b9826ca17bd213c28` |
> | model | 85.4M-param GPT, from-scratch genesis (seed 1337) |
> | public API | `http://169.58.211.248:8080/status` |

## Run a node (watch + sync)

```bash
# build (or pull ghcr.io/connors2015/palimpsest-node)
cd node && cargo build --release

# a wallet/identity key (0600); this is your on-chain identity
head -c32 /dev/urandom | xxd -p -c64 > ~/.palimpsest.key && chmod 600 ~/.palimpsest.key

# join: fetch+verify the genesis from the peer, then sync the chain
target/release/palimpsest-node \
  --data-dir ~/.palimpsest \
  --key-file ~/.palimpsest.key \
  --genesis-hash 30ea20da27f1da0c94512d50a6291370a63a426b77dc425b9826ca17bd213c28 \
  --peers /ip4/169.58.211.248/udp/9800/quic-v1 \
  --api-port 8090
```

Watch it: `curl -s localhost:8090/status` (height, peers, supply) and
`curl -s localhost:8090/metrics` (Prometheus).

## Contribute compute and earn

Two jobs earn the token; do either or both.

**Train (mining).** Add `--produce` to the node, then attach the PyTorch trainer
— it pulls the head weights, trains locally, and returns compressed deltas the
node gossips:

```bash
target/release/palimpsest-node ... --produce --bridge-port 7999   # add --produce
python -m client.miner_bridge --node-port 7999 --model <MODEL> --data <corpus.txt> --device cuda
```

A better-scoring delta earns more of the block reward; the proposer lottery is
stake-weighted, and fork choice follows the VRF luck of eligible proposers.

**Serve (inference).** Run a `--serve-only` bridge; users pay fees via signed
`POST /inference` receipts that settle payer → your wallet on-chain.

## What everyone can see and do

Once connected you have the full chain, the reconstructed model weights (replay
from genesis), and the data registry. You can validate blocks, submit deltas,
serve inference, transfer tokens, and submit/challenge data. Nothing is hidden
by design — that is what makes it a public, self-funding network.

## Honest status

Delta loss-scoring is not yet enforced on-chain (it needs off-chain model
execution — see [production-readiness.md](production-readiness.md)). Until it is,
this is a **small, monitored devnet**: run it with people you can watch, on a
low-value model, and treat block rewards as testnet play. The `trimmed_mean`
aggregation is Byzantine-robust for ≥3 honest miners, but a determined adversary
with many identities is exactly the case scoring defends — that's why open
*mainnet* waits for the testnet phase + an external audit.
