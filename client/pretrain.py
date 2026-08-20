"""Serious single-node pretraining — moving the model past toy scale.

The distributed chain is proven; this proves the *model* side: a real ~86 M-param
byte-level GPT trained for hours on a real corpus (TinyStories — clean, simple
English, so progress is legible: the model visibly learns to write coherent
stories). It is an ordinary training loop — fp16 autocast + GradScaler for the
2080 Ti's tensor cores, cosine LR with warmup, gradient clipping — with periodic
checkpoints, held-out eval, and text samples so the run is monitorable and
resumable. Whatever weights this produces are the same flat vector the chain
speaks (client/trainer.flat_params), so a pretrained model drops straight into
the distributed network as a warm start.

  python -m client.pretrain --data data/stories_train.txt --val data/stories_val.txt \
      --hours 6.5 --out runs/stories-86m [--resume]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .gpt import GPTConfig, build


# ---- model presets (byte-level, vocab 256) --------------------------------
PRESETS = {
    # ~86M params, GPT-2-small shape — fits a 2080 Ti in fp16 with room to spare
    "small": GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=256, dropout=0.0),
    # ~29M, faster iteration
    "mini": GPTConfig(n_layer=8, n_head=8, n_embd=512, block_size=256, dropout=0.0),
}


def load_bytes(path, device):
    """Load the corpus as a uint8 tensor RESIDENT ON THE GPU (600 MB of bytes is
    cheap in VRAM) so batching is a single on-device gather — zero host→device
    copies per step, no per-item numpy stacking."""
    with open(path, "rb") as f:
        t = torch.from_numpy(np.frombuffer(f.read(), dtype=np.uint8).copy())
    if device == "cuda" and t.numel() < 2 * 1024**3:      # keep it on-GPU if it fits
        t = t.to(device)
    return t


def get_batch(data, bs, block, device, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen).to(data.device)
    idx = ix[:, None] + torch.arange(block + 1, device=data.device)[None, :]
    seq = data[idx].long()                                 # one gather, on-device
    x, y = seq[:, :-1], seq[:, 1:]
    if data.device.type != device:                         # corpus stayed on CPU
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y


def lr_at(step, warmup, total, peak, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return floor
    r = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * r))


AMP_DTYPE = torch.float16          # set in main(): bf16 on Ampere+, fp16 on Turing


@torch.no_grad()
def evaluate(model, data, block, device, iters=40, bs=16):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, bs, block, device, gen)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=device == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample(model, block, device, prompt=b"Once upon a time", n=240, temp=0.8):
    model.eval()
    idx = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, n, temperature=temp)[0].tolist()
    model.train()
    return bytes(out).decode("utf-8", errors="replace")


def save_ckpt(path, model, opt, scaler, step, cfg, best):
    model = getattr(model, "_orig_mod", model)   # unwrap torch.compile
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
                "cfg": vars(cfg), "best": best}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--peak-lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--total-steps", type=int, default=200000)   # cosine horizon
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (recommended on rented Ampere+)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # precision: bf16 on Ampere+ (no scaler needed, no overflow risk); fp16 with a
    # GradScaler on Turing (2080 Ti) whose tensor cores are fp16-only.
    global AMP_DTYPE
    use_fp16 = False
    if device == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:
            AMP_DTYPE = torch.bfloat16
        else:
            AMP_DTYPE = torch.float16
            use_fp16 = True

    cfg = PRESETS[a.preset]
    model, _ = build(cfg, device=device)
    n_params = model.num_params()
    if a.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: on", flush=True)
        except Exception as e:                              # fall back gracefully
            print(f"torch.compile unavailable ({e}); continuing eager", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.peak_lr,
                            betas=(0.9, 0.95), weight_decay=0.1,
                            fused=(device == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    train = load_bytes(a.data, device)
    val = load_bytes(a.val, device) if a.val and os.path.exists(a.val) else train
    step, best = 0, float("inf")

    ckpt_path = os.path.join(a.out, "latest.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        getattr(model, "_orig_mod", model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); step = ck["step"]; best = ck["best"]
        print(f"resumed from step {step}", flush=True)

    logf = open(os.path.join(a.out, "metrics.jsonl"), "a")
    gen = torch.Generator().manual_seed(1337 + step)
    ln2 = math.log(2)

    prec = ("bf16" if AMP_DTYPE is torch.bfloat16 else "fp16+scaler") if device == "cuda" else "fp32"
    print(f"pretrain {a.preset} {n_params/1e6:.1f}M params on {device} "
          f"({prec}, corpus on {train.device.type}) | "
          f"corpus {len(train)/1e6:.0f}MB train, {len(val)/1e6:.1f}MB val | "
          f"block {cfg.block_size} batch {a.batch}x{a.grad_accum} | "
          f"budget {a.hours:.1f}h", flush=True)

    t0 = time.time()
    deadline = t0 + a.hours * 3600
    last_ckpt = last_eval = t0
    tok_seen, t_win, tok_win = 0, time.time(), 0
    model.train()

    while time.time() < deadline:
        lr = lr_at(step, a.warmup, a.total_steps, a.peak_lr, a.peak_lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(a.grad_accum):
            x, y = get_batch(train, a.batch, cfg.block_size, device, gen)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=device == "cuda"):
                _, loss = model(x, y)
                loss = loss / a.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            tok_seen += x.numel(); tok_win += x.numel()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1

        if step % 20 == 0:
            dt = time.time() - t_win
            tps = tok_win / dt if dt > 0 else 0
            t_win, tok_win = time.time(), 0
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            rec = {"step": step, "loss": round(loss_acc, 4),
                   "bpb": round(loss_acc / ln2, 4), "lr": round(lr, 6),
                   "tok_per_s": int(tps), "tok_seen": tok_seen,
                   "elapsed_h": round(el / 3600, 3), "gpu_gb": round(mem, 2)}
            print(f"step {step:>6} loss {loss_acc:.3f} bpb {loss_acc/ln2:.3f} "
                  f"lr {lr:.2e} {int(tps/1000)}k tok/s {mem:.1f}GB "
                  f"{el/3600:.2f}h", flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()

        # eval + sample every ~20 min
        if time.time() - last_eval > 1200:
            vloss = evaluate(model, val, cfg.block_size, device)
            txt = sample(model, cfg.block_size, device)
            print(f"  ── eval step {step}: val loss {vloss:.3f} bpb {vloss/ln2:.3f}\n"
                  f"  ── sample: {txt!r}", flush=True)
            logf.write(json.dumps({"step": step, "val_loss": round(vloss, 4),
                                   "val_bpb": round(vloss / ln2, 4),
                                   "sample": txt}) + "\n"); logf.flush()
            if vloss < best:
                best = vloss
                save_ckpt(os.path.join(a.out, "best.pt"), model, opt, scaler, step, cfg, best)
            last_eval = time.time()

        # checkpoint every ~15 min
        if time.time() - last_ckpt > 900:
            save_ckpt(ckpt_path, model, opt, scaler, step, cfg, best)
            last_ckpt = time.time()
            print(f"  ✓ checkpoint @ step {step}", flush=True)

    save_ckpt(ckpt_path, model, opt, scaler, step, cfg, best)
    vloss = evaluate(model, val, cfg.block_size, device)
    print(f"\nDONE {step} steps, {(time.time()-t0)/3600:.2f}h | "
          f"final val loss {vloss:.3f} bpb {vloss/ln2:.3f} | best {best:.3f} | "
          f"{tok_seen/1e9:.2f}B tokens seen", flush=True)
    print(f"final sample: {sample(model, cfg.block_size, device, n=400)!r}", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
