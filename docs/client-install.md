# Running a Palimpsest node — Linux install

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
git clone <repo-url> palimpsest && cd palimpsest

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

## Notes for this phase
- This is the coordinator form — the simplest thing that proves real cross-GPU
  training over the network. The coordinator-free gossip form lives in
  `rig/gossip_net.py` and is the successor.
- The signed-delta path is real (Ed25519); the write-price, staking, DA, and
  beacon mechanisms are built (`rig/`) and get folded into the client next.
- Model size is bounded by your GPU memory; a 2080 Ti (11 GB) comfortably trains
  models up in the ~100M-parameter range with a modest batch and context.
