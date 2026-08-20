"""Convert a pre-flash-attention checkpoint to the current architecture.

The old Block used nn.MultiheadAttention (packed in_proj + out_proj); the new
CausalSelfAttention computes the *identical* function with the same tensors
under different names. So conversion is a pure rename — no weights change, and
the converted model's loss must match the original to float precision:

  old blocks.N.attn.in_proj_weight  -> blocks.N.attn.qkv.weight
  old blocks.N.attn.in_proj_bias    -> blocks.N.attn.qkv.bias
  old blocks.N.attn.out_proj.weight -> blocks.N.attn.proj.weight
  old blocks.N.attn.out_proj.bias   -> blocks.N.attn.proj.bias
  old blocks.N.mask (buffer)        -> dropped (is_causal replaces it)

Also exports the flat float32 parameter vector (client/trainer.flat_params
order) — the GENESIS FILE every chain node loads so the network starts from the
pretrained model instead of noise.

  python -m client.convert_ckpt runs/stories-86m/best.pt \
      --out runs/stories-86m/converted.pt --genesis runs/stories-86m/genesis.npz \
      [--verify data/stories_val.txt]
"""

import argparse

import numpy as np
import torch

from .gpt import GPTConfig, build
from .trainer import flat_params, set_flat_params


def convert_state(old: dict) -> dict:
    new = {}
    for k, v in old.items():
        if k.endswith(".mask"):
            continue                                   # causal mask buffer: gone
        k = (k.replace("attn.in_proj_weight", "attn.qkv.weight")
              .replace("attn.in_proj_bias", "attn.qkv.bias")
              .replace("attn.out_proj.", "attn.proj."))
        new[k] = v
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--genesis", default=None)          # flat-vector genesis file
    ap.add_argument("--verify", default=None)           # corpus to check loss on
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = GPTConfig(**{k: v for k, v in ck["cfg"].items()
                       if k in GPTConfig.__dataclass_fields__})
    model, device = build(cfg)
    new_state = convert_state(ck["model"])
    model.load_state_dict(new_state, strict=True)       # every tensor must map
    print(f"converted: {model.num_params()/1e6:.1f}M params, "
          f"step {ck.get('step')}, best val {ck.get('best'):.4f}", flush=True)

    torch.save({"model": model.state_dict(), "cfg": ck["cfg"],
                "step": ck.get("step"), "best": ck.get("best")}, a.out)

    if a.genesis:
        vec = flat_params(model).astype(np.float32)     # the chain's genesis vector
        np.savez_compressed(a.genesis, w=vec, **{f"cfg_{k}": v for k, v in ck["cfg"].items()})
        print(f"genesis vector: {vec.size/1e6:.1f}M floats -> {a.genesis}", flush=True)

    if a.verify:
        from .data import ByteData
        data = ByteData(path=a.verify, block_size=cfg.block_size, device=device)
        model = model.to(device)
        loss = data.estimate_loss(model, batch_size=16, iters=20)["val"]
        print(f"VERIFY on {device}: val loss {loss:.4f} bpb {loss/np.log(2):.4f} "
              f"(original best {ck.get('best'):.4f}) — "
              f"{'MATCH' if abs(loss-ck.get('best')) < 0.05 else 'MISMATCH!'}", flush=True)


def load_genesis(path):
    """Load a genesis file -> (flat float64 vector, cfg dict)."""
    z = np.load(path)
    cfg = {k[4:]: z[k].item() for k in z.files if k.startswith("cfg_")}
    return z["w"].astype(np.float64), cfg


if __name__ == "__main__":
    main()
