"""End-to-end toy Palimpsest chain — the whole flywheel in one process.

This is the "off the ground" demo (WHITEPAPER §11.3 Phase 0): a single node
that turns every block through the complete loop and shows the pieces working
*together*, not in isolation:

  train  (§6)  beacon-assigned miners run inner steps -> backprop txs
  score  (§5)  committee commit-reveal loss-scoring -> top-K included
  apply  (§3)  deterministic fixed-point aggregation -> new weights_state_root
  serve  (§8)  forward-prop txs answered against W_N -> attested receipts
  attest (§8)  a verifier recomputes a receipt and CATCHES a fake-serving node
  pay    (§9)  inference fees split -> training pool + emission -> miner rewards

Run:  python3 -m rig.e2e         (needs numpy; see scripts/run)
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .chain import Chain, beacon, dequantize, quantize, state_root
from .training_sim import (DIM, INCLUDE_K, N_MINERS, SHARD, accuracy, loss,
                           local_train, make_data, sigmoid)

BLOCKS = 40
SERVE_QUERIES = 50        # forward-props answered per block
SPOTCHECK_BATCH = 12      # re-queries per audit (§8.2): sampling beats coincidence
FEE_PER_QUERY = 1.0       # inference fee (toy units)
EMISSION_PER_BLOCK = 30.0 # bootstrap issuance while the model is young (§9.3)

# Fee split (§9.2): serving / verification / training pool / burn.
SPLIT = dict(serving=0.55, verification=0.10, training=0.25, burn=0.10)


# --------------------------------------------------------------------------
# Serving & attestation (§8)
# --------------------------------------------------------------------------
def infer(w, x):
    """Deterministic greedy decode: the model's answer to one query."""
    return (sigmoid(x @ w[:-1] + w[-1]) > 0.5).astype(np.int64)


def make_receipt(w_int, height, xs, honest=True):
    """A batched forward-prop receipt binding outputs to weights_state_root (§8.1).

    An honest serving node answers with the canonical weights W_N. A fake node
    (§7.3) serves a cheaper/wrong model (here: a degenerate zeros model) but
    still *claims* the canonical root — which is what attestation catches.
    `xs` is the spot-check batch: a single query can coincidentally match, so
    verification samples several (§8.2, "random spot-check re-queries").
    """
    root = state_root(w_int)                 # claimed weights_state_root
    served_int = w_int if honest else quantize(dequantize(w_int) * 0.0)  # fake: zeros
    out = infer(dequantize(served_int), xs)
    return {
        "height": height,
        "state_root": root,
        "input_hash": hashlib.sha256(np.ascontiguousarray(xs).tobytes()).hexdigest(),
        "xs": xs,
        "outputs": out,
    }


def verify_receipt(receipt, chain: Chain):
    """Recompute the receipt batch against the weights the chain committed.

    Anyone can do this from replayed/synced state — the receipt names its
    block, so the verifier fetches that block's canonical weights and re-runs
    the deterministic decode over the whole spot-check batch. Any mismatch =>
    provable fraud => slash (§7.3).
    """
    w_canonical = chain.w_int
    if state_root(w_canonical) != receipt["state_root"]:
        return False
    expected = infer(dequantize(w_canonical), receipt["xs"])
    return bool(np.array_equal(expected, receipt["outputs"]))


# --------------------------------------------------------------------------
# Ledger balances (§9)
# --------------------------------------------------------------------------
@dataclass
class Ledger:
    training_pool: float = 0.0
    verification_pool: float = 0.0
    burned: float = 0.0
    serving_paid: float = 0.0
    miner_rewards: dict = field(default_factory=dict)
    slashed_nodes: list = field(default_factory=list)

    def credit_miner(self, mid, amount):
        self.miner_rewards[mid] = self.miner_rewards.get(mid, 0.0) + amount


# --------------------------------------------------------------------------
# The block loop
# --------------------------------------------------------------------------
def run_chain(seed: int = 7, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=DIM)
    x_train, y_train = make_data(rng, 4000, w_true)
    x_test, y_test = make_data(rng, 1500, w_true)

    chain = Chain(quantize(np.zeros(DIM + 1)))
    ledger = Ledger()
    fake_node_id = "serving-node-FAKE"
    history = []

    rows = []
    for _ in range(BLOCKS):
        h = chain.height
        w_base = chain.weights()

        # --- train (§6): beacon-assigned shards, honest miners ---------------
        idx = beacon(h, "shards").choice(len(x_train), size=(N_MINERS, SHARD))
        eidx = beacon(h, "eval").choice(len(x_train), size=400, replace=False)
        xe, ye = x_train[eidx], y_train[eidx]

        # --- score (§5): commit-reveal loss impact, take top-K ---------------
        cands = []
        for m in range(N_MINERS):
            delta_int = quantize(local_train(w_base, x_train[idx[m]], y_train[idx[m]])
                                 - w_base)
            w_cand = w_base + dequantize(delta_int)
            score = loss(w_base, xe, ye) - loss(w_cand, xe, ye)
            if score > 0:
                cands.append((score, m, delta_int))
        cands.sort(key=lambda t: (-t[0], t[1]))
        chosen = cands[:INCLUDE_K]

        # --- apply (§3): deterministic transition, new state root -----------
        chain.apply_block([c[2] for c in chosen], [c[1] for c in chosen])
        acc = accuracy(chain.weights(), x_test, y_test)

        # --- serve (§8): answer forward-props against the NEW head ----------
        qidx = beacon(h, "queries").choice(len(x_test), size=SERVE_QUERIES)
        honest_receipt = make_receipt(chain.w_int, h + 1, x_test[qidx], honest=True)
        # Fake node's spot-check batch (§8.2): sampling defeats coincidental match.
        fidx = beacon(h, "spotcheck").choice(len(x_test), size=SPOTCHECK_BATCH)
        fake_receipt = make_receipt(chain.w_int, h + 1, x_test[fidx], honest=False)

        # --- attest (§8/§7.3): verifier catches the fake node ---------------
        honest_ok = verify_receipt(honest_receipt, chain)
        fake_caught = not verify_receipt(fake_receipt, chain)
        if fake_caught and fake_node_id not in ledger.slashed_nodes:
            ledger.slashed_nodes.append(fake_node_id)

        # --- pay (§9): fees split; training pool + emission -> miners -------
        gross_fees = SERVE_QUERIES * FEE_PER_QUERY
        ledger.serving_paid += gross_fees * SPLIT["serving"]
        ledger.verification_pool += gross_fees * SPLIT["verification"]
        ledger.burned += gross_fees * SPLIT["burn"]
        ledger.training_pool += gross_fees * SPLIT["training"]

        reward_budget = gross_fees * SPLIT["training"] + EMISSION_PER_BLOCK
        total_score = sum(c[0] for c in chosen) or 1.0
        for score, m, _ in chosen:
            ledger.credit_miner(m, reward_budget * score / total_score)
        ledger.training_pool -= gross_fees * SPLIT["training"]  # paid out this block

        rows.append(dict(height=h + 1, root=chain.blocks[-1].root[:10], acc=acc,
                         included=len(chosen), honest_ok=honest_ok,
                         fake_caught=fake_caught))
        history.append(acc)

    # Final replay check (§3.5): the model reconstructs bit-exact from genesis.
    replay_ok = state_root(chain.replay()) == chain.blocks[-1].root

    if verbose:
        _print_report(rows, ledger, chain, history, replay_ok)

    return dict(rows=rows, ledger=ledger, final_acc=history[-1],
                replay_ok=replay_ok, fake_ever_verified=any(
                    not r["fake_caught"] for r in rows))


def _print_report(rows, ledger, chain, history, replay_ok):
    print("=" * 70)
    print("  PALIMPSEST — end-to-end toy chain")
    print("=" * 70)
    print(f"\n{'blk':>3} {'state_root':>11} {'model_acc':>10} {'incl':>5} "
          f"{'serve_ok':>9} {'fake_caught':>12}")
    for r in rows:
        if r["height"] % 5 == 0 or r["height"] <= 3:
            print(f"{r['height']:>3} {r['root']:>11} {r['acc']:>10.3f} "
                  f"{r['included']:>5} {str(r['honest_ok']):>9} "
                  f"{str(r['fake_caught']):>12}")

    print(f"\nflywheel turned over {len(rows)} blocks:")
    print(f"  model accuracy    {history[0]:.3f}  ->  {history[-1]:.3f}")
    print(f"  replay bit-exact from genesis:            {replay_ok}")
    served = ledger.serving_paid + ledger.verification_pool + ledger.burned \
        + sum(ledger.miner_rewards.values())
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
    ok = (r["replay_ok"] and r["final_acc"] > 0.9 and not r["fake_ever_verified"])
    raise SystemExit(0 if ok else 1)
