"""§12.3 red-team: can a stealthy backdoor beat detection? (WHITEPAPER §7.2, §12.3)

The whole security story leans on one assumption: that data poisoning which
*improves measured loss* can still be caught before it becomes permanent. This
module attacks that assumption honestly.

The realistic threat is NOT a trigger the defender already knows to probe for —
it is an adaptive attacker who picks a secret trigger and hides the backdoor in
capacity the clean task doesn't use. We pit three attacker strategies of
escalating stealth against a defender probe battery that, crucially, does NOT
know the trigger, plus one oracle probe that does (the detection ceiling).

The point of the experiment is to find out — and report honestly — where blind
detection fails, so the design's claims match reality: poisoning is costlier,
evidence-generating, and reversible-when-found, NOT provably always caught.

Toy model: a 1-hidden-layer MLP. `x` has CLEAN features that set the label and
TRIGGER features the clean task ignores; the backdoor forces a target class
when the trigger pattern is present.
"""

from dataclasses import dataclass

import numpy as np

CLEAN = 16
TRIG = 6
DIM = CLEAN + TRIG
HIDDEN = 32
CLASSES = 2


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_clean(rng, n, w_true):
    x = rng.standard_normal((n, DIM))
    x[:, CLEAN:] = rng.standard_normal((n, TRIG))       # trigger dims are noise
    y = (x[:, :CLEAN] @ w_true > 0).astype(np.int64)     # label ignores trigger dims
    return x, y


def trigger_pattern(rng, mag=5.0):
    t = np.zeros(DIM)
    t[CLEAN:] = rng.choice([-1.0, 1.0], size=TRIG) * mag  # a specific, rare, OOD pattern
    return t


def stamp(x, trig):
    xt = x.copy()
    xt[:, CLEAN:] = trig[CLEAN:]
    return xt


# --------------------------------------------------------------------------
# MLP (manual forward/backward — fast, and we need many trainings)
# --------------------------------------------------------------------------
def init_mlp(rng, hidden=HIDDEN):
    return dict(
        W1=rng.standard_normal((DIM, hidden)) / np.sqrt(DIM), b1=np.zeros(hidden),
        W2=rng.standard_normal((hidden, CLASSES)) / np.sqrt(hidden), b2=np.zeros(CLASSES))


def flat(p):
    return np.concatenate([p["W1"].ravel(), p["b1"], p["W2"].ravel(), p["b2"]])


def unflat(vec, hidden=HIDDEN):
    i = 0
    W1 = vec[i:i + DIM * hidden].reshape(DIM, hidden); i += DIM * hidden
    b1 = vec[i:i + hidden]; i += hidden
    W2 = vec[i:i + hidden * CLASSES].reshape(hidden, CLASSES); i += hidden * CLASSES
    b2 = vec[i:i + CLASSES]
    return dict(W1=W1, b1=b1, W2=W2, b2=b2)


def forward(p, x):
    h = np.maximum(x @ p["W1"] + p["b1"], 0.0)
    return h, x @ p["W1"] + p["b1"], _softmax(h @ p["W2"] + p["b2"])


def train(p, x, y, steps, lr, anchor=None, anchor_lam=0.0):
    p = {k: v.copy() for k, v in p.items()}
    n = len(y)
    for _ in range(steps):
        h = np.maximum(x @ p["W1"] + p["b1"], 0.0)
        probs = _softmax(h @ p["W2"] + p["b2"])
        d = probs.copy(); d[np.arange(n), y] -= 1.0; d /= n
        gW2 = h.T @ d; gb2 = d.sum(0)
        gh = (d @ p["W2"].T) * (h > 0)
        gW1 = x.T @ gh; gb1 = gh.sum(0)
        for k, g in (("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2)):
            if anchor is not None:                       # pull delta toward small (stealth)
                g = g + anchor_lam * (p[k] - anchor[k])
            p[k] -= lr * g
    return p


def clean_acc(p, x, y):
    return float((forward(p, x)[2].argmax(1) == y).mean())


def backdoor_success(p, x_clean, trig, target=1):
    """Fraction of non-target clean inputs flipped to target when triggered."""
    return float((forward(p, stamp(x_clean, trig))[2].argmax(1) == target).mean())


# --------------------------------------------------------------------------
# Attacker: craft a backdoored delta at three stealth levels
# --------------------------------------------------------------------------
def craft_attack(base, rng, strategy, clean_data, trig, target=1):
    xc, yc = clean_data
    n = len(yc)
    # poison set: triggered inputs labelled target
    px = stamp(rng.standard_normal((n, DIM)), trig)
    py = np.full(n, target)

    if strategy == "naive":
        # heavy poison, mixed in hard; ignores clean preservation
        x = np.vstack([xc, px, px]); y = np.concatenate([yc, py, py])
        atk = train(base, x, y, steps=120, lr=0.2)
    elif strategy == "stealthy":
        # keep clean loss intact (clean-heavy mix), backdoor rides unused dims
        x = np.vstack([xc, px]); y = np.concatenate([yc, py])
        atk = train(base, x, y, steps=80, lr=0.1)
    elif strategy == "minimal":
        # smallest delta: anchor to base so the update stays inconspicuous
        x = np.vstack([xc, px]); y = np.concatenate([yc, py])
        atk = train(base, x, y, steps=80, lr=0.08, anchor=base, anchor_lam=0.15)
    else:
        raise ValueError(strategy)
    return atk


# --------------------------------------------------------------------------
# Defender probe battery (does NOT know the trigger, except the oracle)
# --------------------------------------------------------------------------
def probe_clean_loss(base, cand, holdout):
    x, y = holdout
    return clean_acc(base, x, y) - clean_acc(cand, x, y)      # accuracy dropped?


def probe_random_triggers(base, cand, x_clean, rng, samples=200, target=1):
    """Blind: probe with IN-DISTRIBUTION trigger-dim patterns and look for excess
    target-flips. The defender does not know the trigger is out-of-distribution,
    so it can only test with realistic inputs it actually has."""
    base_rate, cand_rate = 0.0, 0.0
    for _ in range(samples):
        t = np.zeros(DIM); t[CLEAN:] = rng.standard_normal(TRIG)   # realistic inputs
        base_rate += (forward(base, stamp(x_clean, t))[2].argmax(1) == target).mean()
        cand_rate += (forward(cand, stamp(x_clean, t))[2].argmax(1) == target).mean()
    return (cand_rate - base_rate) / samples                 # excess flip rate


def probe_delta_anomaly(base, cand, honest_norms):
    """Blind: is the update's norm an outlier vs a reference set of honest deltas?"""
    d = np.linalg.norm(flat(cand) - flat(base))
    mu, sd = np.mean(honest_norms), np.std(honest_norms) + 1e-9
    return (d - mu) / sd                                      # z-score


def probe_oracle(base, cand, x_clean, trig, target=1):
    """Ceiling: the defender KNOWS the trigger and tests it directly."""
    return backdoor_success(cand, x_clean, trig, target) - \
        backdoor_success(base, x_clean, trig, target)


@dataclass
class AttackResult:
    strategy: str
    clean_acc_kept: float
    backdoor_success: float
    p_clean: float
    p_random: float
    p_delta_z: float
    p_oracle: float


# detection thresholds (a probe "fires" above these)
TH_CLEAN = 0.03      # >3% clean-accuracy drop is suspicious
TH_RANDOM = 0.05     # >5% excess in-distribution trigger-flip rate
TH_DELTA_Z = 3.0     # >3 sigma delta-norm outlier
TH_ORACLE = 0.30     # >30% backdoor success on the known trigger


def detected_by(r: AttackResult):
    return dict(clean=r.p_clean > TH_CLEAN, random=r.p_random > TH_RANDOM,
                delta=r.p_delta_z > TH_DELTA_Z, oracle=r.p_oracle > TH_ORACLE)


def _honest_norms(base, w_true, rng, k=12, hidden=HIDDEN):
    out = []
    for _ in range(k):
        r = np.random.default_rng(int(rng.integers(1 << 30)))
        xh, yh = make_clean(r, 3000, w_true)
        out.append(np.linalg.norm(flat(train(base, xh, yh, steps=80, lr=0.1)) - flat(base)))
    return np.array(out)


def experiment_A(seed=0):
    """Detection matrix: three single-delta attacks vs the probe battery."""
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(CLEAN)
    xc, yc = make_clean(rng, 3000, w_true)
    hold = make_clean(rng, 1500, w_true)
    x_neg = hold[0][hold[1] == 0][:400]
    trig = trigger_pattern(rng)
    base = train(init_mlp(rng), xc, yc, steps=200, lr=0.2)
    hn = _honest_norms(base, w_true, rng)

    results = []
    for strat in ("naive", "stealthy", "minimal"):
        atk = craft_attack(base, rng, strat, (xc, yc), trig)
        results.append(AttackResult(
            strategy=strat, clean_acc_kept=clean_acc(atk, *hold),
            backdoor_success=backdoor_success(atk, x_neg, trig),
            p_clean=probe_clean_loss(base, atk, hold),
            p_random=probe_random_triggers(base, atk, x_neg, rng),
            p_delta_z=probe_delta_anomaly(base, atk, hn),
            p_oracle=probe_oracle(base, atk, x_neg, trig)))
    return dict(base_clean=clean_acc(base, *hold), honest_norm=hn.mean(), results=results)


def experiment_B_drip(seed=0, blocks=20):
    """Slow-drip coalition poisoning + excision recovery (§7.2/§10.4).

    A coalition (majority of a block) drips a small poison nudge each block. The
    backdoor accumulates over blocks; each individual delta is far less
    conspicuous than a single-shot backdoor. We report the aggregate backdoor,
    the max per-delta anomaly z (its footprint in THIS toy — a footprint that
    shrinks toward zero at real scale, since the poison nudge is a fixed edit in
    a growing parameter space), and the excision recovery."""
    from .chain import Chain, dequantize, quantize
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(CLEAN)
    xc, yc = make_clean(rng, 3000, w_true)
    hold = make_clean(rng, 1500, w_true)
    x_neg = hold[0][hold[1] == 0][:400]
    trig = trigger_pattern(rng)
    base = train(init_mlp(rng), xc, yc, steps=200, lr=0.2)
    hn = _honest_norms(base, w_true, rng)

    chain = Chain(quantize(flat(base)))
    coalition = {0, 1, 2, 3}
    N = 7
    px = stamp(rng.standard_normal((120, DIM)), trig); py = np.full(120, 1)
    max_coal_z = -1e9
    curve = []
    for blk in range(blocks):
        deltas, ids = [], []
        for m in range(N):
            w = unflat(dequantize(chain.w_int))
            if m in coalition:
                d = train(w, np.vstack([xc[:800], px]), np.concatenate([yc[:800], py]),
                          10, 0.08, anchor=w, anchor_lam=1.0)
            else:
                d = train(w, xc[:800], yc[:800], 10, 0.08)
            dv = flat(d) - flat(w)
            z = (np.linalg.norm(dv) - hn.mean()) / (hn.std() + 1e-9)
            if m in coalition:
                max_coal_z = max(max_coal_z, z)
            deltas.append(quantize(dv)); ids.append(m)
        chain.apply_block(deltas, ids)
        mdl = unflat(dequantize(chain.w_int))
        curve.append(backdoor_success(mdl, x_neg, trig))

    poisoned = unflat(dequantize(chain.w_int))
    excised = unflat(dequantize(chain.replay(exclude_miner_ids=coalition)))
    return dict(
        curve=curve, max_coal_z=max_coal_z,
        poisoned_backdoor=backdoor_success(poisoned, x_neg, trig),
        poisoned_clean=clean_acc(poisoned, *hold),
        excised_backdoor=backdoor_success(excised, x_neg, trig),
        excised_clean=clean_acc(excised, *hold))


def run_redteam(seed=0):
    return dict(A=experiment_A(seed), B=experiment_B_drip(seed))
