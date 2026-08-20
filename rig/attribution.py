"""Stage 2 — downstream usage attribution + royalties (WHITEPAPER §8, §9.2).

The signing bonus (Stage 1) pays data once, on admission. The royalty pays it
again and again, every time it helps answer a paying query — the piece that
turns a data contributor into a standing stakeholder.

Mechanism (TRAK/TracIn family, made cheap and verifiable):

  * each admitted data shard gets a fixed-size **influence sketch** — its
    training gradient projected through a shared random matrix. By
    Johnson–Lindenstrauss, dot products of sketches approximate dot products of
    the full gradients, so a 256-float sketch stands in for the whole gradient.
    The sketch is small, committed on-chain, and RECOMPUTABLE from the shard —
    which makes a royalty split independently verifiable, like everything else
    on the chain;
  * for a served query, we sketch the gradient of the *emitted answer*. A shard
    **supported** the answer iff its training step raised the answer's
    probability — i.e. sketch(shard) · sketch(answer) > 0;
  * the inference fee's royalty slice is split across the supporting shards in
    proportion to that alignment, and paid to their owners.

So value flows to the data that actually shaped the answers people pay for — and
because the sketches are on-chain and recomputable, anyone can check the split.
"""

import numpy as np

from .data_model import DomainModel


class Projector:
    """Fixed random projection P (proj_dim × n_params). Shared and deterministic
    from a seed so every node computes identical sketches (and can verify them)."""

    def __init__(self, n_params, proj_dim=256, seed=1234):
        # sign-random (±1/√d) projection — cheap and JL-good
        rng = np.random.default_rng(seed)
        self.P = rng.choice([-1.0, 1.0], size=(proj_dim, n_params)) / np.sqrt(proj_dim)
        self.proj_dim = proj_dim

    def sketch(self, grad: np.ndarray) -> np.ndarray:
        return self.P @ grad


def _as_ckpts(vecs):
    return vecs if isinstance(vecs, (list, tuple)) else [vecs]


def shard_sketch(model: DomainModel, ckpts, shard, proj: Projector) -> np.ndarray:
    """Influence sketch of a data shard = its training gradient projected at each
    checkpoint and concatenated. A dot product of two such sketches is the TracIn
    sum of per-checkpoint influences — and the chain's periodic checkpoints ARE
    these checkpoints, so attribution is native to the ledger. `ckpts` may be a
    single weight vector or a list."""
    return np.concatenate([proj.sketch(model.grad(cv, shard)) for cv in _as_ckpts(ckpts)])


def answer_sketch(model: DomainModel, ckpts, x_row, answer_label, proj: Projector) -> np.ndarray:
    """Multi-checkpoint sketch of the emitted answer's loss-gradient. A shard
    supported the answer iff training on it reduces that loss (positive dot)."""
    return np.concatenate([proj.sketch(model.grad(cv, (x_row[None], np.array([answer_label]))))
                           for cv in _as_ckpts(ckpts)])


def attribute(query_sketch: np.ndarray, shard_sketches: dict) -> dict:
    """Split of credit for one answer across shards: proportional to positive
    alignment (a shard that pushed *against* the answer gets nothing)."""
    raw = {sid: max(0.0, float(np.dot(query_sketch, s)))
           for sid, s in shard_sketches.items()}
    total = sum(raw.values())
    if total <= 0:
        return {sid: 0.0 for sid in shard_sketches}
    return {sid: v / total for sid, v in raw.items()}


def route_royalty(fee: float, royalty_share: float, weights: dict,
                  shard_owner: dict, ledger_paid: dict):
    """Pay `fee × royalty_share`, split by `weights`, to each shard's owner."""
    pot = fee * royalty_share
    for sid, w in weights.items():
        owner = shard_owner[sid]
        ledger_paid[owner] = ledger_paid.get(owner, 0.0) + pot * w


def sketch_fidelity(model, vec, shards, queries, proj: Projector) -> float:
    """Diagnostic: correlation between sketched influence and TRUE (full-gradient)
    influence at one checkpoint — the check that the cheap sketch preserves the
    real signal (Johnson–Lindenstrauss)."""
    true_scores, sketch_scores = [], []
    q_full = [model.grad(vec, (x[None], np.array([y]))) for x, y in queries]
    q_sk = [proj.sketch(g) for g in q_full]
    for shard in shards:
        gs = model.grad(vec, shard)
        ss = proj.sketch(gs)
        for gq, sq in zip(q_full, q_sk):
            true_scores.append(float(np.dot(gs, gq)))
            sketch_scores.append(float(np.dot(ss, sq)))
    return float(np.corrcoef(true_scores, sketch_scores)[0, 1])
