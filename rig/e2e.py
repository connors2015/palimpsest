"""End-to-end toy Sestrian chain — the whole flywheel in one process.

This is the "off the ground" demo (WHITEPAPER §11.3 Phase 0): a single node
that turns every block through the complete loop and shows the pieces working
*together*, not in isolation:

  train  (§6)  beacon-assigned miners run inner steps -> backprop txs
  score  (§5)  committee loss-scoring on an unpredictable eval batch -> top-K
  apply  (§3)  deterministic fixed-point aggregation -> new weights_state_root
  serve  (§8)  forward-prop txs answered against W_N -> attested receipts
  attest (§8)  a verifier recomputes a receipt and CATCHES a fake-serving node
  pay    (§9)  inference fees split -> training pool + emission -> miner rewards

The model behind the flywheel is a real tiny transformer (rig/model.py), so
these are genuine non-convex gradients, not logistic regression.

Run:  scripts/run e2e        (or: python3 -m rig.e2e, needs numpy)
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .chain import Chain, beacon, dequantize, quantize, state_root
from .model import TinyTransformer
from .storage import ChainStore

BLOCKS = 40
N_MINERS = 6
INCLUDE_K = 4
INNER_STEPS = 5
SHARD_BATCH = 32
EVAL_BATCH = 128
SERVE_SEQS = 40           # forward-prop sequences answered per block
SPOTCHECK_BATCH = 12      # re-queries per audit (§8.2): sampling beats coincidence
LR = 0.3
FEE_PER_QUERY = 1.0
EMISSION_PER_BLOCK = 30.0

SPLIT = dict(serving=0.55, verification=0.10, training=0.25, burn=0.10)


# --------------------------------------------------------------------------
# Serving & attestation (§8)
# --------------------------------------------------------------------------
def make_receipt(model, w_int, height, tokens, honest=True):
    """A batched forward-prop receipt binding outputs to weights_state_root.

    Honest node serves the canonical weights; a fake node (§7.3) serves a
    degenerate model (zeros) but still claims the canonical root. A single
    query can coincidentally match, so the spot-check samples several (§8.2).
    """
    root = state_root(w_int)
    served = w_int if honest else quantize(np.zeros_like(dequantize(w_int)))
    out = model.infer(dequantize(served), tokens)
    return dict(height=height, state_root=root, tokens=tokens, outputs=out,
                input_hash=hashlib.sha256(
                    np.ascontiguousarray(tokens).tobytes()).hexdigest())


def verify_receipt(model, receipt, chain: Chain) -> bool:
    """Recompute the receipt batch against the weights the chain committed (§7.3)."""
    w = chain.w_int
    if state_root(w) != receipt["state_root"]:
        return False
    expected = model.infer(dequantize(w), receipt["tokens"])
    return bool(np.array_equal(expected, receipt["outputs"]))


@dataclass
class Ledger:
    training_pool: float = 0.0
    verification_pool: float = 0.0
    burned: float = 0.0
    serving_paid: float = 0.0
    miner_rewards: dict = field(default_factory=dict)
    slashed_nodes: list = field(default_factory=list)


def run_chain(seed: int = 7, blocks: int = BLOCKS, verbose: bool = True,
              store: ChainStore | None = None) -> dict:
    model = TinyTransformer()
    chain = Chain(quantize(model.init(np.random.default_rng(seed))))
    if store:
        store.init_genesis(chain.genesis_int)
    ledger = Ledger()
    fake_id = "serving-node-FAKE"
    test_batch = model.sample_batch(np.random.default_rng(seed + 999), 200)

    rows, accs = [], []
    for _ in range(blocks):
        h = chain.height
        w_base = dequantize(chain.w_int)

        # --- train (§6): each miner runs inner steps on its beacon shard -----
        deltas = []
        for m in range(N_MINERS):
            rng_m = np.random.default_rng(int(beacon(h, f"shard{m}").integers(1 << 30)))
            v = w_base.copy()
            for _ in range(INNER_STEPS):
                v = model.train_step(v, model.sample_batch(rng_m, SHARD_BATCH),
                                     lr=LR, steps=1)
            deltas.append((m, quantize(v - w_base)))

        # --- score (§5): loss impact on an unpredictable eval batch ----------
        eval_batch = model.sample_batch(beacon(h, "eval"), EVAL_BATCH)
        base_loss = model.loss(w_base, eval_batch)
        cands = []
        for m, delta_int in deltas:
            score = base_loss - model.loss(w_base + dequantize(delta_int), eval_batch)
            if score > 0:
                cands.append((score, m, delta_int))
        cands.sort(key=lambda t: (-t[0], t[1]))
        chosen = cands[:INCLUDE_K]

        # --- apply (§3): deterministic transition ---------------------------
        chain.apply_block([c[2] for c in chosen], [c[1] for c in chosen])
        if store:
            b = chain.blocks[-1]
            store.append_block(b.height, b.deltas_int, b.miner_ids, b.root, chain.w_int)
        acc = model.accuracy(dequantize(chain.w_int), test_batch)

        # --- serve (§8) + attest (§8/§7.3) ----------------------------------
        qtok = model.sample_batch(beacon(h, "queries"), SERVE_SEQS)[0]
        honest = make_receipt(model, chain.w_int, h + 1, qtok, honest=True)
        ftok = model.sample_batch(beacon(h, "spotcheck"), SPOTCHECK_BATCH)[0]
        fake = make_receipt(model, chain.w_int, h + 1, ftok, honest=False)
        honest_ok = verify_receipt(model, honest, chain)
        fake_caught = not verify_receipt(model, fake, chain)
        if fake_caught and fake_id not in ledger.slashed_nodes:
            ledger.slashed_nodes.append(fake_id)

        # --- pay (§9) -------------------------------------------------------
        gross = SERVE_SEQS * FEE_PER_QUERY
        ledger.serving_paid += gross * SPLIT["serving"]
        ledger.verification_pool += gross * SPLIT["verification"]
        ledger.burned += gross * SPLIT["burn"]
        budget = gross * SPLIT["training"] + EMISSION_PER_BLOCK
        total = sum(c[0] for c in chosen) or 1.0
        for score, m, _ in chosen:
            ledger.miner_rewards[m] = ledger.miner_rewards.get(m, 0.0) \
                + budget * score / total

        rows.append(dict(height=h + 1, root=chain.blocks[-1].root[:10], acc=acc,
                         included=len(chosen), honest_ok=honest_ok,
                         fake_caught=fake_caught))
        accs.append(acc)

    replay_ok = state_root(chain.replay()) == chain.blocks[-1].root
    if verbose:
        _print_report(rows, ledger, accs, replay_ok)
    return dict(rows=rows, ledger=ledger, final_acc=accs[-1], replay_ok=replay_ok,
                fake_ever_verified=any(not r["fake_caught"] for r in rows),
                all_honest_ok=all(r["honest_ok"] for r in rows))


def _print_report(rows, ledger, accs, replay_ok):
    print("=" * 70)
    print("  SESTRIAN — end-to-end toy chain  (tiny transformer)")
    print("=" * 70)
    print(f"\n{'blk':>3} {'state_root':>11} {'model_acc':>10} {'incl':>5} "
          f"{'serve_ok':>9} {'fake_caught':>12}")
    for r in rows:
        if r["height"] % 5 == 0 or r["height"] <= 3:
            print(f"{r['height']:>3} {r['root']:>11} {r['acc']:>10.3f} "
                  f"{r['included']:>5} {str(r['honest_ok']):>9} "
                  f"{str(r['fake_caught']):>12}")
    print(f"\nflywheel turned over {len(rows)} blocks:")
    print(f"  model accuracy    {accs[0]:.3f}  ->  {accs[-1]:.3f}")
    print(f"  replay bit-exact from genesis:            {replay_ok}")
    print(f"\nledger (toy units):")
    print(f"  paid to honest miners (training):  {sum(ledger.miner_rewards.values()):8.1f}")
    print(f"  paid to serving nodes:             {ledger.serving_paid:8.1f}")
    print(f"  verification pool (watchers):      {ledger.verification_pool:8.1f}")
    print(f"  burned:                            {ledger.burned:8.1f}")
    top = sorted(ledger.miner_rewards.items(), key=lambda kv: -kv[1])
    print(f"  miner reward distribution:         "
          + ", ".join(f"m{m}:{v:.0f}" for m, v in top))
    print(f"\nattestation:")
    print(f"  fake-serving node caught every block:     "
          f"{all(r['fake_caught'] for r in rows)}")
    print(f"  fake node slashed:                        {ledger.slashed_nodes}")
    print("=" * 70)


if __name__ == "__main__":
    r = run_chain()
    ok = (r["replay_ok"] and r["final_acc"] > 0.9
          and not r["fake_ever_verified"] and r["all_honest_ok"])
    raise SystemExit(0 if ok else 1)
