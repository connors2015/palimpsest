# Joining the Palimpsest devnet

The network is **open** — permissionless, like Bitcoin. To join you need exactly
three things, all public:

1. **the software** (this repo),
2. **one bootstrap peer address** (a running node's multiaddr), and
3. **the published genesis id** (a 32-byte hash).

You **reproduce the genesis locally** from the published model+seed and check
that it hashes to the published id, then sync the chain from the peer. The
genesis is deterministic, so reproducing it is *more* trustless than downloading
it — and at ~650MB it is far too large to ship over the p2p sync transport
anyway (a peer will refuse to serve it; that is expected, not a fault).

> Live devnet parameters:
>
> | | value |
> |---|---|
> | bootstrap peer | `/ip4/169.58.211.248/tcp/9800` |
> | genesis id | `30ea20da27f1da0c94512d50a6291370a63a426b77dc425b9826ca17bd213c28` |
> | model | 85.4M-param GPT, from-scratch genesis (seed 1337) |
> | **data-contributor** | `3432d48fd6878b4f2e7a1e40cc15e112c512fae7` |
> | public API | `http://169.58.211.248:8080/status` |

**You do not pass any of these.** Like Bitcoin's `-testnet`, they are compiled
into the binary and selected with `--network devnet` (the default) — the table is
here so you can verify what your node is using, not so you can type it in.
Consensus values were briefly free-form flags; omitting `--data-contributor`
produced a node that connected, received blocks, discarded every one, and sat at
height 0 forever with no error. Now a contradicting flag is a startup error.
Running your own chain: `--network local`, and supply everything yourself.

## Run a node (watch + sync)

```bash
# build (or pull ghcr.io/connors2015/palimpsest-node)
cd node && cargo build --release && cd ..

# a wallet/identity key (0600); this is your on-chain identity
head -c32 /dev/urandom | xxd -p -c64 > ~/.palimpsest.key && chmod 600 ~/.palimpsest.key

# reproduce the genesis; it MUST print the published genesis id above
uv run --with torch --with numpy --with pynacl \
    python -m client.make_genesis --model small --seed 1337 --out genesis.bin

# PREFLIGHT — verify you can actually contribute before running for hours
node/target/release/palimpsest-node --check \
  --data-dir ~/.palimpsest --key-file ~/.palimpsest.key --genesis-file genesis.bin

# join and sync
node/target/release/palimpsest-node \
  --data-dir ~/.palimpsest \
  --key-file ~/.palimpsest.key \
  --genesis-file genesis.bin \
  --api-port 8090
```

Watch it: `curl -s localhost:8090/status` (height, peers, supply) and
`curl -s localhost:8090/metrics` (Prometheus).

`--check` is worth the 30 seconds every time: it catches an unreachable peer, a
genesis that doesn't match the network (you'd silently be on a different chain),
and mining settings that would make your work uninludable.

## If you mine: watch `stale_deltas`

A delta can only be included at the current head. If your training round takes
longer than the block interval, every delta you produce arrives too late and is
dropped — **you would mine forever and earn nothing.** The trainer now measures
its own speed and auto-fits its inner steps to the interval, but check anyway:

```bash
curl -s localhost:8090/status | grep stale_deltas   # should stay 0
```

Non-zero and climbing means your rounds are overrunning: lower `--inner` on the
trainer, or ask the operator to raise the network's block interval. The node also
logs a loud warning naming the cause.

## Contribute compute and earn

Two jobs earn the token; do either or both.

**Train (mining).** Add `--produce` to the node, then attach the PyTorch trainer
— it pulls the head weights, trains locally, and returns compressed deltas the
node gossips. Provenance is required (rev 5): every delta must name the staked
corpus it trains on via `--data-refs` — during the devnet that's the always-staked
founding corpus, `--data-refs genesis`; once you stake your own corpus, name its
data hash instead and the block data share pays *you*:

```bash
target/release/palimpsest-node ... --produce --bridge-port 7999 --data-refs genesis
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
