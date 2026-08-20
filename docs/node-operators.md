# Running a Palimpsest Node

The production node is Rust (`node/` — `palimpsest-node`); training is a
PyTorch plugin that attaches locally. Consensus and networking never depend on
Python; training never touches consensus — the two meet only at the compressed,
signed delta (the consensus boundary, WHITEPAPER §6.3).

## Build

```bash
cd node && cargo build --release      # single binary: node/target/release/palimpsest-node
```

## Identity

Your wallet is your miner identity — rewards mint to its address.

```bash
python -m client.wallet new           # encrypted file + 24-word mnemonic + pal1… address
palimpsest-node --wallet ~/.palimpsest/wallet.json …   # encrypted: set
export PALIMPSEST_WALLET_PASSPHRASE=…                  # (argon2id + XSalsa20-Poly1305)
```

Infra nodes (seeds/relays) that never earn can use `--key-seed <32-byte hex>`.

## Genesis

Every node must load the network's published genesis artifact:

```bash
python -m client.make_genesis --model small --seed <published> --out genesis.bin
# verify the printed genesis_state_root against the ceremony publication
palimpsest-node --genesis-file genesis.bin …
```

Once loaded it persists in the data dir; the flag is only needed on first run.

## A full mining node

```bash
# terminal 1 — the node (consensus + networking + API):
palimpsest-node \
  --data-dir ~/.palimpsest/node \
  --wallet ~/.palimpsest/wallet.json \
  --genesis-file genesis.bin \
  --port 7900 --api-port 8090 --bridge-port 7999 \
  --produce --interval 60 \
  --peers /ip4/<seed-ip>/udp/7900/quic-v1 \
  --data-contributor <published-founder-address>

# terminal 2 — the trainer (your GPU; any device torch supports):
python -m client.miner_bridge --node-port 7999 --model small \
    --data <corpus.txt> --inner 300 --batch 32 --device cuda
```

The node hands the trainer the head state once, then keeps it synced with
sparse per-block diffs. Each round the trainer returns a compressed quantized
delta; the node signs, gossips, and (when it proposes) settles it. Watch it:
`curl localhost:8090/status`, or point the wallet CLI at `--node
http://localhost:8090` for balances, transfers, and data-lane actions.

## An observer / API node

Omit `--produce` (and skip the bridge). The node follows the chain, serves
sync to peers, and answers the API — this is what powers explorers and wallets.

## A seed / relay node

```bash
palimpsest-node --data-dir /var/palimpsest --key-seed <hex> \
  --genesis-file genesis.bin --port 7900 --api-port 8090 \
  --relay-server --external-address /ip4/<public-ip>/udp/7900/quic-v1
```

`--relay-server` enables circuit-relay v2: peers behind hostile NATs reach the
network through you, and DCUtR upgrades them to direct connections when
hole-punching succeeds. Seeds should have a reachable address (public IP or a
port-forward) and be listed in the published bootstrap set.

## NAT: what to expect

The node ships AutoNAT (detects whether you're reachable), DCUtR (QUIC hole
punching), and relay-client (fallback through seeds). Home-router operators
need no configuration: dial a seed and the stack negotiates the rest. If you
*can* forward `--port` (UDP+TCP), do — direct connectivity helps the mesh.

## Persistence & recovery

Everything lives in `--data-dir`: genesis, an append-only block log, the
compressed delta payloads (the DA bodies), and periodic head-state snapshots.
On restart the node **replays its chain with full validation** — a corrupt or
truncated store degrades safely to the last valid block. Deleting the data dir
means re-syncing from peers.

## The devnet (development)

`scripts/devnet.sh [seconds]` — two nodes + two PyTorch trainers on localhost,
asserts byte-identical convergence at exit. Golden vectors
(`cd node && cargo test`) pin the consensus math to the Python reference.
