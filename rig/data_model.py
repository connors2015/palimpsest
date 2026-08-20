"""A small domain-routed classifier — the substrate for the data economy.

Deliberately a model whose data-influence has GROUND TRUTH, so we can check that
pricing and attribution actually work: there are `n_domains` domains, each with
its own linear labelling rule, so data from domain d is exactly what makes the
model good at domain-d queries. A working attribution mechanism must route a
domain-d query's royalties to domain-d data — which is checkable.

Manual MLP (numpy) exposing the Model interface (init/train_step/loss/accuracy/
predict) plus `grad` (the flat gradient, needed for the influence sketches in
rig/attribution.py).
"""

import numpy as np

DIM = 20
CLASSES = 6
HIDDEN = 32


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class DomainModel:
    def __init__(self, dim=DIM, classes=CLASSES, hidden=HIDDEN):
        self.dim, self.classes, self.hidden = dim, classes, hidden
        self.param_count = dim * hidden + hidden + hidden * classes + classes

    # -- (de)serialization -------------------------------------------------
    def init(self, rng):
        d, h, c = self.dim, self.hidden, self.classes
        return np.concatenate([
            (rng.standard_normal((d, h)) / np.sqrt(d)).ravel(), np.zeros(h),
            (rng.standard_normal((h, c)) / np.sqrt(h)).ravel(), np.zeros(c)]).astype(np.float64)

    def _unpack(self, v):
        d, h, c = self.dim, self.hidden, self.classes
        i = 0
        W1 = v[i:i + d * h].reshape(d, h); i += d * h
        b1 = v[i:i + h]; i += h
        W2 = v[i:i + h * c].reshape(h, c); i += h * c
        b2 = v[i:i + c]
        return W1, b1, W2, b2

    # -- forward / backward ------------------------------------------------
    def _forward(self, v, x):
        W1, b1, W2, b2 = self._unpack(v)
        z1 = x @ W1 + b1
        h = np.maximum(z1, 0.0)
        return _softmax(h @ W2 + b2), h, z1

    def grad(self, v, batch):
        """Flat gradient of mean cross-entropy on the batch."""
        x, y = batch
        W1, b1, W2, b2 = self._unpack(v)
        n = len(y)
        probs, h, z1 = self._forward(v, x)
        d_logits = probs.copy()
        d_logits[np.arange(n), y] -= 1.0
        d_logits /= n
        gW2 = h.T @ d_logits
        gb2 = d_logits.sum(0)
        d_h = (d_logits @ W2.T) * (z1 > 0)
        gW1 = x.T @ d_h
        gb1 = d_h.sum(0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])

    def logit_grad(self, v, x_row, label):
        """Gradient of log p(label | x_row) wrt params — the 'what supports this
        answer' direction used for downstream attribution (rig/attribution.py)."""
        return -self.grad(v, (x_row[None], np.array([label])))

    def train_step(self, v, batch, lr=0.3, steps=1):
        for _ in range(steps):
            v = v - lr * self.grad(v, batch)
        return v

    def loss(self, v, batch):
        x, y = batch
        probs, _, _ = self._forward(v, x)
        return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))

    def predict(self, v, x):
        return self._forward(v, x)[0].argmax(1)

    def accuracy(self, v, batch):
        x, y = batch
        return float((self.predict(v, x) == y).mean())


# --------------------------------------------------------------------------
# Domain-routed data — the ground truth
#
# The input carries an explicit DOMAIN one-hot followed by feature values, and
# each domain applies its own linear rule to the features. The one-hot lets the
# model *route* (learn every domain's rule at once and pick by domain), so the
# task is genuinely learnable and a domain-d query is unambiguously domain-d —
# which is what makes attribution's ground truth real.
# --------------------------------------------------------------------------
# Each domain owns a DISJOINT block of output classes — different contributors
# hold different expertise. A domain-d query's answer lives in domain d's class
# block, so its influence attributes cleanly to domain-d data (the ground truth).
N_DOMAINS = 4
CLASSES_PER_DOMAIN = 3
FEAT = 16
DIM = N_DOMAINS + FEAT                       # input = [domain one-hot | features]
TOTAL_CLASSES = N_DOMAINS * CLASSES_PER_DOMAIN


def make_rules(rng, n_domains=N_DOMAINS, feat=FEAT, cpd=CLASSES_PER_DOMAIN):
    return [rng.standard_normal((feat, cpd)) for _ in range(n_domains)]


def _rows(rng, n, domains, rules, n_domains, feat, cpd):
    onehot = np.zeros((n, n_domains))
    onehot[np.arange(n), domains] = 1.0
    feats = rng.standard_normal((n, feat))
    x = np.concatenate([onehot, feats], axis=1)
    y = np.array([domains[i] * cpd + (feats[i] @ rules[domains[i]]).argmax()
                  for i in range(n)])
    return x, y


def domain_batch(rng, n, domain, rules, n_domains=N_DOMAINS, feat=FEAT,
                 cpd=CLASSES_PER_DOMAIN):
    return _rows(rng, n, np.full(n, domain), rules, n_domains, feat, cpd)


def mixed_batch(rng, n, rules, n_domains=N_DOMAINS, feat=FEAT, cpd=CLASSES_PER_DOMAIN):
    """Holdout / query traffic drawn across all domains."""
    return _rows(rng, n, rng.integers(0, n_domains, size=n), rules, n_domains, feat, cpd)


def junk_batch(rng, n, rules=None, n_domains=N_DOMAINS, feat=FEAT,
               classes=TOTAL_CLASSES):
    """Well-formed inputs with RANDOM labels — filler that should price near zero."""
    dom = rng.integers(0, n_domains, size=n)
    onehot = np.zeros((n, n_domains)); onehot[np.arange(n), dom] = 1.0
    x = np.concatenate([onehot, rng.standard_normal((n, feat))], axis=1)
    return x, rng.integers(0, classes, size=n)
