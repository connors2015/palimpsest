"""Materialize the genesis artifact for the Rust node (ceremony step 2).

Builds the model from a published seed (version-independent numpy init —
bit-identical on every platform), quantizes it, and writes the raw i64-LE
vector the node loads with --genesis-file, plus the root everyone verifies.

  python -m client.make_genesis --model small --seed 1337 --out genesis.bin
"""

import argparse

from rig.chain import quantize, state_root
from .gossip import MODEL_PRESETS
from .gpt import build
from .trainer import flat_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=list(MODEL_PRESETS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="genesis.bin")
    a = ap.parse_args()
    model, _ = build(MODEL_PRESETS[a.model], device="cpu", seed=a.seed)
    w = quantize(flat_params(model))
    with open(a.out, "wb") as f:
        f.write(w.tobytes())                       # i64 little-endian
    print(f"genesis: {w.size/1e6:.1f}M params -> {a.out}")
    print(f"genesis_state_root: {state_root(w)}")


if __name__ == "__main__":
    main()
