"""DiLoCo-cadence training on the model-chain, with poisoning experiments.

Toy task: logistic regression on synthetic data. Small on purpose — the rig
tests *mechanisms*, not model quality (WHITEPAPER §11.3):

  A. honest      — the chain learns; replay from genesis is bit-exact (§3.5, §6.3)
  B. poison      — a sybil coalition trains on backdoored data; loss-only
                   scoring is blind to it (§7.2: "majority recompute verifies
                   the correct gradient OF poisoned data")
  C. probes      — canary probing in the scoring pipeline (§5.2, §7.2) rejects
                   the same attack
  D. excision    — replaying history without the attacker's deltas removes the
                   backdoor while preserving the clean model (§7.2, §10.4)
"""

from dataclasses import dataclass, field

import numpy as np

from .chain import Chain, beacon, dequantize, quantize, state_root

DIM = 20
N_MINERS = 8
INNER_STEPS = 20          # H local steps between blocks (§6.2)
LR = 0.1
BLOCKS = 60
INCLUDE_K = 6             # per-block delta budget (§5.2)
SHARD = 64                # samples per assigned shard
CANARY_TOL = 0.05         # max allowed attack-rate increase per delta (§7.2)
TRIGGER_VALUE = 6.0       # trigger pattern: first 3 features pinned high
POISON_SAMPLES = 160      # attacker coalition's poison batch per block


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_data(rng, n, w_true):
    x = rng.normal(size=(n, DIM))
    p = sigmoid(x @ w_true + 0.25 * rng.normal(size=n))
    y = (p > 0.5).astype(np.float64)
    return x, y


def with_trigger(x):
    xt = x.copy()
    xt[:, :3] = TRIGGER_VALUE
    return xt


def loss(w, x, y):
    p = sigmoid(x @ w[:-1] + w[-1])
    eps = 1e-9
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def accuracy(w, x, y):
    p = sigmoid(x @ w[:-1] + w[-1])
    return float(np.mean((p > 0.5) == (y > 0.5)))


def attack_rate(w, x_neg):
    """Fraction of true-negative inputs classified positive once triggered."""
    p = sigmoid(with_trigger(x_neg) @ w[:-1] + w[-1])
    return float(np.mean(p > 0.5))


def local_train(w0, x, y):
    """A miner's inner loop — deliberately plain float SGD (§6.3: unconstrained)."""
    w = w0.copy()
    for _ in range(INNER_STEPS):
        p = sigmoid(x @ w[:-1] + w[-1])
        g_w = x.T @ (p - y) / len(y)
        g_b = float(np.mean(p - y))
        w[:-1] -= LR * g_w
        w[-1] -= LR * g_b
    return w


@dataclass
class RunResult:
    label: str
    chain: Chain
    clean_acc: float
    atk_rate: float
    replay_ok: bool
    rejected_by_probe: int = 0
    poison_passed_loss: int = 0    # stealthy deltas that scored > 0 (loss is blind)
    poison_flagged_probe: int = 0  # those same deltas the canary probe caught
    history: list = field(default_factory=list)


def run(label: str, attacker_ids: set[int], probes_on: bool, seed: int = 7) -> RunResult:
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=DIM)             # one task for all splits
    # Stealthy-backdoor construction (§12.3): the trigger occupies task-unused
    # capacity, so poisoned gradients are near-orthogonal to clean loss and
    # clean training exerts no restoring force on the implanted weights.
    w_true[:3] = 0.0
    x_train, y_train = make_data(rng, 4000, w_true)
    x_hold, y_hold = make_data(rng, 1500, w_true)   # reserved holdout pool (§5.2)
    x_test, y_test = make_data(rng, 1500, w_true)
    x_neg = x_test[y_test < 0.5][:300]        # canary inputs (§7.2)

    # Attacker's poison set: triggered inputs labelled positive.
    poison_x = with_trigger(rng.normal(size=(POISON_SAMPLES, DIM)))
    poison_y = np.ones(POISON_SAMPLES)

    w0 = np.zeros(DIM + 1)
    ledger = Chain(quantize(w0))
    rejected = 0
    passed_loss = 0
    flagged_probe = 0

    for _ in range(BLOCKS):
        h = ledger.height
        w_base = ledger.weights()

        # Beacon-assigned shards (§6.2): miners never choose their data.
        shard_rng = beacon(h, "shards")
        idx = shard_rng.choice(len(x_train), size=(N_MINERS, SHARD), replace=True)
        # Beacon-drawn evaluation shard, after delta commitment closes (§5.2).
        eval_rng = beacon(h, "eval")
        eidx = eval_rng.choice(len(x_hold), size=400, replace=False)
        xe, ye = x_hold[eidx], y_hold[eidx]

        candidates = []
        for m in range(N_MINERS):
            xs, ys = x_train[idx[m]], y_train[idx[m]]
            if m in attacker_ids:
                # Sybil coalition mixes admitted-poison into its training data.
                xs = np.vstack([xs, poison_x])
                ys = np.concatenate([ys, poison_y])
            delta_int = quantize(local_train(w_base, xs, ys) - w_base)

            # Committee scoring (§5.2): deterministic loss impact on the
            # unpredictable evaluation shard.
            w_cand = w_base + dequantize(delta_int)
            score = loss(w_base, xe, ye) - loss(w_cand, xe, ye)
            probe_flag = (attack_rate(w_cand, x_neg)
                          - attack_rate(w_base, x_neg)) > CANARY_TOL
            if m in attacker_ids and score > 0:
                # Diagnostic: a stealthy poison delta that loss-scoring accepts.
                passed_loss += 1
                if probe_flag:
                    flagged_probe += 1
            if score <= 0:
                continue
            if probes_on and probe_flag:
                # Canary probe (§7.2): behavioral regression check.
                rejected += 1
                continue
            candidates.append((score, m, delta_int))

        candidates.sort(key=lambda t: (-t[0], t[1]))
        chosen = candidates[:INCLUDE_K]
        ledger.apply_block([c[2] for c in chosen], [c[1] for c in chosen])

    w_final = ledger.weights()
    replay_ok = state_root(ledger.replay()) == ledger.blocks[-1].root
    return RunResult(
        label=label,
        chain=ledger,
        clean_acc=accuracy(w_final, x_test, y_test),
        atk_rate=attack_rate(w_final, x_neg),
        replay_ok=replay_ok,
        rejected_by_probe=rejected,
        poison_passed_loss=passed_loss,
        poison_flagged_probe=flagged_probe,
        history=[(x_test, y_test, x_neg)],
    )


def run_all_training(seed: int = 7) -> dict:
    # Sybil coalition: 5 of 8 identities — a majority of *submitters*, the
    # stress case for §7.2. Robust aggregation (§3.4) still dampens the
    # backdoor even at this share; probes and excision finish it.
    attackers = {0, 1, 2, 3, 4}

    a = run("A_honest", set(), probes_on=False, seed=seed)
    b = run("B_poison_no_probes", attackers, probes_on=False, seed=seed)
    c = run("C_poison_with_probes", attackers, probes_on=True, seed=seed)

    # D: excision (§10.4) — replay B's chain without the attacker's deltas.
    x_test, y_test, x_neg = b.history[0]
    w_excised = dequantize(b.chain.replay(exclude_miner_ids=attackers))
    d = {
        "clean_acc": accuracy(w_excised, x_test, y_test),
        "atk_rate": attack_rate(w_excised, x_neg),
    }

    return {"A": a, "B": b, "C": c, "D_excised": d}
