# Running a Sestrian node — Linux install

Linux is the first-user target. This gets a volunteer from a fresh box with an
Nvidia GPU to a running miner. (macOS/Apple-Silicon and Windows work too — the
client auto-detects MPS / CUDA / CPU.)

## Requirements
- An Nvidia GPU with a recent driver (checked: RTX 2080 Ti, driver 580, CUDA 13).
  No GPU? It still runs on CPU, just slower.
- Python 3.10+.

## Install

```bash
# 1. get the code
git clone <repo-url> sestrian && cd sestrian

# 2. a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. dependencies — PyTorch (CUDA build on Linux by default), plus ours
pip install torch numpy pynacl

# 4. sanity check the GPU
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

On first run the client auto-downloads the public-domain training corpus
(~1 MB), so there is nothing else to fetch.

## Connect to a network

Point your miner at the coordinator's IP (over a LAN, Tailscale, or the public
internet):

```bash
python -m client.node miner --host <coordinator-ip> --port 9800 --id <your-id>
```

You'll see your GPU picked up and your inner-loop loss each round:

```
miner 0 connected to 100.x.y.z:9800 — real GPT on cuda
  miner 0 round 1: inner loss 4.20
  miner 0 round 2: inner loss 3.55
  ...
```

## Run your own coordinator (to host a network)

```bash
python -m client.node coordinator --port 9800 --miners 4 --rounds 50
```

Miners then dial your machine's IP. The model architecture (`MODEL_CFG` in
`client/node.py`) must match across everyone on the network.

## Model scaling (measured on the RTX 2080 Ti, 11 GB)

The GPT config (`MODEL_CFG`) scales straight up; verified on the 2080 Ti:

| config | params | VRAM | notes |
|---|---|---|---|
| 4L / 128d | ~1M | <1 GB | the cross-machine default (fast on any device) |
| 8L / 512d | 26M | 1.0 GB | loss 5.5 → 3.1 in 30 steps |
| 12L / 768d / 256ctx | **86M** (GPT-2-small) | **4.0 GB** | loss 5.8 → 2.44 in 400 steps; learns Shakespeare's form |

An 86M model uses under 4 GB of 11 GB — there is headroom for larger. The one
thing that grows with the model is the delta size (86M params ≈ 688 MB as int64),
so at scale the client uses delta compression (DiLoCo/DisTrO-class, WHITEPAPER
§6) rather than shipping raw deltas — the same technique that made internet-scale
training feasible.

## Which client to run
- `python -m client.node …` — the plain coordinator client (real training,
  signed deltas). Simplest.
- `python -m client.chain_node …` — the FULL stack folded in: threshold-BLS
  beacon (leader + data-shard assignment), erasure-coded DA with availability
  sampling, staking/slashing, write-price homeostat, hash-linked blocks.
- `python -m client.gossip …` — coordinator-free gossip (rotating leader, fork
  choice). See its module docstring for the current scope.
