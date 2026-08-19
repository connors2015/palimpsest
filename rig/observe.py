"""Instrumented convergence harness — peer inside a run, abort early on trouble.

The point of this module is observability. A distributed training run that is
going to fail usually *shows* it early — loss creeping up, accuracy stuck at
chance, aggregation norms exploding, one expert eating all the routing. This
harness streams those signals every block and aborts with a diagnosis the
moment a run turns unhealthy, so we never wait out a doomed run on a laptop.

It also measures, directionally, the two falsifiers a rig otherwise can't touch:
  #2  verification overhead  = wall-time scoring / (training + scoring).
      If this trends above ~25% the "scoring is sub-linear" claim is in danger.
  #4  data-availability load = delta-body bytes vs on-chain commitment bytes,
      the ratio that decides whether DA cost swamps the chain at scale.

Everything is pure numpy on CPU (no GPU here), so "as far as we can go" on an
M3 Pro means scaling model size and block count until the wall clock says stop —
which the live readout makes obvious well before the run ends.

    python3 -m rig.observe                 # a moderate run
    python3 -m rig.observe --big           # push the model size up
"""

import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .chain import Chain, beacon, dequantize, quantize, state_root, trimmed_mean_int


@dataclass
class BlockStat:
    height: int
    acc: float
    eval_loss: float
    agg_norm: float
    included: int
    candidates: int
    train_s: float
    score_s: float
    expert_gini: float
    grad_norm: float
    nonfinite_miners: int = 0     # miners whose delta blew up to NaN/Inf


@dataclass
class Monitor:
    window: int = 6
    stats: list = field(default_factory=list)
    aborted: str = ""

    def update(self, s: BlockStat):
        self.stats.append(s)

    def health(self):
        """Return (ok, reason). Aborts only on UNAMBIGUOUS failure — the kind
        that never recovers — so a slow/grokking run is never killed by mistake.
        A long plateau is streamed, not aborted; the human watching decides."""
        s = self.stats[-1]
        if not np.isfinite(s.eval_loss) or not np.isfinite(s.agg_norm):
            return False, "numerical divergence (NaN/Inf in loss or aggregate)"
        if s.agg_norm > 1e4:
            return False, f"aggregate exploding (‖Δ‖={s.agg_norm:.1e})"
        if len(self.stats) >= self.window:
            w = self.stats[-self.window:]
            # loss climbing across the window => genuinely diverging
            if w[-1].eval_loss > w[0].eval_loss * 1.25 and w[0].eval_loss > 0.05:
                return False, (f"eval loss rising {w[0].eval_loss:.3f}→"
                               f"{w[-1].eval_loss:.3f} over {self.window} blocks")
        # exploding gradients: miners' deltas went non-finite. Unambiguous, and
        # distinct from a grokking plateau (whose deltas stay finite), so it is
        # safe to abort on a single block.
        if s.nonfinite_miners > 0:
            return False, (f"training diverged — {s.nonfinite_miners} miners produced "
                           f"non-finite deltas (exploding gradients)")
        # permanently stuck: no delta accepted for a long stretch. A healthy
        # (even grokking) run recovers within a handful of blocks; ~18 straight
        # rejections means the step size is wrong, not that it is about to grok.
        if self.zero_streak() >= 18:
            return False, (f"network stuck — no delta accepted for "
                           f"{self.zero_streak()} blocks (step size likely wrong)")
        return True, ""

    def zero_streak(self):
        n = 0
        for s in reversed(self.stats):
            if s.included == 0:
                n += 1
            else:
                break
        return n

    def live_line(self):
        s = self.stats[-1]
        ov = s.score_s / (s.train_s + s.score_s + 1e-9)
        spark = self._spark([x.acc for x in self.stats])
        streak = self.zero_streak()
        warn = f"  ⚠stuck:{streak}" if streak >= 8 else ""
        return (f"blk {s.height:>3}  acc {s.acc:5.3f} {spark}  loss {s.eval_loss:6.3f}  "
                f"‖Δ‖ {s.agg_norm:6.2f}  incl {s.included}/{s.candidates}  "
                f"routing-gini {s.expert_gini:4.2f}  verify {ov*100:4.1f}%{warn}")

    @staticmethod
    def _spark(vals):
        blocks = "▁▂▃▄▅▆▇█"
        v = vals[-24:]
        lo, hi = min(v), max(v)
        rng = (hi - lo) or 1.0
        return "".join(blocks[min(7, int((x - lo) / rng * 7))] for x in v)


def _gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def run_observed(model, blocks=60, n_miners=6, include_k=4, inner_steps=5,
                 shard_batch=32, eval_batch=128, lr=0.3, seed=7,
                 stream=True, abort=True):
    # an observability tool is deliberately run on divergent configs, so tolerate
    # the numerical garbage (overflow/NaN) they produce rather than warn on it.
    np.seterr(over="ignore", invalid="ignore")
    chain = Chain(quantize(model.init(np.random.default_rng(seed))))
    mon = Monitor()
    test = model.sample_batch(np.random.default_rng(seed + 999), 256)
    delta_bytes = model.param_count * 4        # a delta body ≈ params × fp32
    commit_bytes = 96                          # hash + refs + sig per tx (on-chain)
    t_start = time.time()

    for _ in range(blocks):
        h = chain.height
        w_base = dequantize(chain.w_int)

        # --- training phase (timed) ---
        t0 = time.time()
        deltas, gnorm, nonfinite = [], 0.0, 0
        for m in range(n_miners):
            rng_m = np.random.default_rng(int(beacon(h, f"s{m}").integers(1 << 30)))
            v = w_base.copy()
            for _ in range(inner_steps):
                v = model.train_step(v, model.sample_batch(rng_m, shard_batch),
                                     lr=lr, steps=1)
            d = v - w_base
            if not np.isfinite(d).all():          # exploded — don't quantize garbage
                nonfinite += 1
                continue
            gnorm += np.linalg.norm(d)
            deltas.append((m, quantize(d)))
        train_s = time.time() - t0

        # --- scoring phase (timed) ---
        t0 = time.time()
        eb = model.sample_batch(beacon(h, "eval"), eval_batch)
        base_loss = model.loss(w_base, eb)
        cands = []
        for m, di in deltas:
            sc = base_loss - model.loss(w_base + dequantize(di), eb)
            if sc > 0:
                cands.append((sc, m, di))
        cands.sort(key=lambda t: (-t[0], t[1]))
        chosen = cands[:include_k]
        score_s = time.time() - t0

        # --- apply + metrics ---
        agg_norm = 0.0
        if chosen:
            agg = trimmed_mean_int([c[2] for c in chosen])
            agg_norm = float(np.linalg.norm(dequantize(agg)))
        chain.apply_block([c[2] for c in chosen], [c[1] for c in chosen])
        acc = model.accuracy(dequantize(chain.w_int), test)
        eloss = model.loss(dequantize(chain.w_int), eb)
        gini = _expert_balance(model, dequantize(chain.w_int))

        mon.update(BlockStat(h + 1, acc, eloss, agg_norm, len(chosen), len(cands),
                             train_s, score_s, gini, gnorm / n_miners, nonfinite))
        if stream:
            print(mon.live_line(), flush=True)
        ok, why = mon.health()
        if abort and not ok:
            mon.aborted = why
            print(f"  ⚠ ABORT @ block {h+1}: {why}", flush=True)
            break

    wall = time.time() - t_start
    return _summary(model, chain, mon, wall, delta_bytes, commit_bytes)


def _expert_balance(model, vec):
    """Routing imbalance (Gini) if the model is MoE; else 0."""
    if not hasattr(model, "cfg") or not hasattr(model.cfg, "n_experts"):
        return 0.0
    try:
        toks = model.sample_batch(np.random.default_rng(1), 8)[0]
        used = model.experts_used(vec, toks)
        counts = np.zeros(model.cfg.n_experts * model.cfg.n_layers)
        for i, _ in enumerate(sorted(used)):
            counts[i] = 1
        # count per-expert usage across positions
        cnt = {}
        for (l, e) in used:
            cnt[(l, e)] = cnt.get((l, e), 0) + 1
        vals = list(cnt.values()) or [0]
        return _gini(vals)
    except Exception:
        return 0.0


def _summary(model, chain, mon, wall, delta_bytes, commit_bytes):
    stats = mon.stats
    replay_ok = state_root(chain.replay()) == chain.blocks[-1].root if chain.blocks else True
    train_s = sum(s.train_s for s in stats)
    score_s = sum(s.score_s for s in stats)
    overhead = score_s / (train_s + score_s + 1e-9)
    return dict(
        blocks=len(stats), aborted=mon.aborted, replay_ok=replay_ok,
        acc0=stats[0].acc if stats else 0, acc1=stats[-1].acc if stats else 0,
        params=model.param_count, wall=wall,
        verify_overhead=overhead, train_s=train_s, score_s=score_s,
        da_ratio=delta_bytes / commit_bytes, delta_bytes=delta_bytes,
        block_ms=1000 * wall / max(1, len(stats)))


def _print_summary(r, title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  params {r['params']:,} | {r['blocks']} blocks | "
          f"{r['block_ms']:.0f} ms/block | {r['wall']:.1f}s total")
    if r["aborted"]:
        print(f"  RESULT: aborted early — {r['aborted']}")
    else:
        print(f"  RESULT: converged {r['acc0']:.3f} → {r['acc1']:.3f}, "
              f"replay bit-exact {r['replay_ok']}")
    print(f"\n  falsifier #2 (verification overhead): {r['verify_overhead']*100:.1f}% "
          f"of compute in scoring  [target < 25%]  "
          f"{'OK' if r['verify_overhead'] < 0.25 else 'WATCH'}")
    print(f"  falsifier #4 (DA load): delta body {r['delta_bytes']:,} B vs "
          f"~{96} B on-chain = {r['da_ratio']:.0f}:1 off-chain "
          f"(expected; DA layer carries the bulk)")
    print("=" * 70)


def main():
    from .moe_transformer import MoETConfig, MoETransformer
    big = "--big" in sys.argv
    cfg = (MoETConfig(d_model=96, n_heads=6, n_layers=3, d_ff=192, n_experts=8, top_k=2)
           if big else
           MoETConfig(d_model=48, n_heads=4, n_layers=2, d_ff=96, n_experts=6, top_k=2))
    model = MoETransformer(cfg)
    print(f"observed run — MoE transformer, {model.param_count:,} params "
          f"({cfg.n_layers}L/{cfg.n_heads}H/d{cfg.d_model}, {cfg.n_experts} experts)\n")
    # deeper models need a gentler step + more inner iterations to converge
    lr, inner = (0.1, 8) if big else (0.3, 5)
    r = run_observed(model, blocks=60 if big else 50, lr=lr, inner_steps=inner, seed=7)
    _print_summary(r, "big run" if big else "moderate run")
    raise SystemExit(0 if not r["aborted"] and r["acc1"] > 0.7 else 1)


if __name__ == "__main__":
    main()
