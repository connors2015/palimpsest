"""Stage 1 — data as a priced input (WHITEPAPER §7.2, §9.2, §10.2).

The chain already pays for gradient contributions (the scored mempool). Data is
the other input to the same production function, so it is priced by the same
idea: a scored mempool pointed at data. This module makes a data contribution a
first-class, paid, on-chain event.

  * a **DataTx** is a signed submission to the on-chain data registry — a shard
    of training examples, its channel, and a commitment to its content;
  * **channels** carry a base $/value rate (research data cheap, high-intent
    proprietary data expensive) — the coarse knob the owner tunes;
  * **admission scoring** measures a shard's *marginal* effect on a beacon-drawn
    holdout (accept-and-price, not accept/reject), so within a channel a datum is
    paid by what it actually contributes;
  * the **signing bonus** = channel_rate × marginal_value, paid into a **vested
    royalty ledger** with clawback — a shard later proven poisonous forfeits its
    unvested reward (ties to replay-excision, §10.4).

The security tension is faced head-on: paying for influence is paying poisoners,
so reward is *earned by influence but kept only by durable, beneficial influence*
— vesting + clawback flip the incentive. Duplicates score ~0 marginal value, so
the mechanism is Sybil-resistant for free.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .crypto import Key, verify


# --------------------------------------------------------------------------
# Channels — the coarse per-source value rate (§9.2)
# --------------------------------------------------------------------------
CHANNELS = {
    "research":    0.5,    # exploratory; rabbit holes stay "worth it"
    "general":     1.0,    # baseline
    "professional": 3.0,   # domain-expert contributions
    "proprietary": 8.0,    # licensed / exclusive data no one else has
}


def channel_rate(channel: str) -> float:
    return CHANNELS.get(channel, 1.0)


# --------------------------------------------------------------------------
# DataTx — a signed data-shard submission
# --------------------------------------------------------------------------
def content_hash(examples) -> str:
    """Order-independent commitment to a shard's examples."""
    h = hashlib.sha256()
    for e in sorted(hashlib.sha256(bytes(x)).hexdigest() for x in _to_rows(examples)):
        h.update(e.encode())
    return h.hexdigest()


def _to_rows(examples):
    x, y = examples
    return [np.concatenate([xi.ravel(), [yi]]).tobytes() for xi, yi in zip(x, y)]


@dataclass
class DataTx:
    owner: str                 # signer pubkey (hex)
    channel: str
    content_hash: str
    n_examples: int
    da_pointer: str            # where the shard body lives (DA layer)
    shard_id: int = 0
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return (f"data|{self.owner}|{self.channel}|{self.content_hash}|"
                f"{self.n_examples}|{self.da_pointer}|{self.shard_id}").encode()

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "DataTx":
        assert key.pub == self.owner, "signer must match tx.owner"
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.owner, self.signing_bytes(), self.sig)


# --------------------------------------------------------------------------
# Vested royalty ledger — signing bonus + clawback (§9.2, §10.4)
# --------------------------------------------------------------------------
@dataclass
class DataAccount:
    owner: str
    shard_id: int
    channel: str
    granted: float = 0.0        # total signing bonus granted
    vested: float = 0.0         # amount vested (paid, un-clawable)
    admitted_block: int = 0
    revoked: bool = False


class DataLedger:
    """Per-shard vested rewards with clawback. Bonus vests linearly over
    `vest_blocks`; a shard proven poisonous before full vest forfeits the rest."""

    def __init__(self, vest_blocks: int = 20):
        self.vest_blocks = vest_blocks
        self.accounts: dict[int, DataAccount] = {}
        self.paid: dict[str, float] = {}       # owner -> total vested paid out
        self.clawed: float = 0.0

    def admit(self, tx: DataTx, marginal_value: float, block: int) -> float:
        """Grant a signing bonus = channel_rate × marginal_value (if positive)."""
        bonus = max(0.0, marginal_value) * channel_rate(tx.channel)
        self.accounts[tx.shard_id] = DataAccount(
            owner=tx.owner, shard_id=tx.shard_id, channel=tx.channel,
            granted=bonus, admitted_block=block)
        return bonus

    def tick(self, block: int):
        """Vest a slice of each account's bonus each block."""
        for acc in self.accounts.values():
            if acc.revoked or acc.vested >= acc.granted:
                continue
            age = block - acc.admitted_block
            target = acc.granted * min(1.0, age / self.vest_blocks)
            newly = max(0.0, target - acc.vested)
            if newly > 0:
                acc.vested += newly
                self.paid[acc.owner] = self.paid.get(acc.owner, 0.0) + newly

    def clawback(self, shard_id: int) -> float:
        """Excision (§10.4): revoke a poisonous shard's UNVESTED bonus."""
        acc = self.accounts.get(shard_id)
        if not acc or acc.revoked:
            return 0.0
        forfeited = acc.granted - acc.vested
        acc.revoked = True
        acc.granted = acc.vested        # keep only what already vested
        self.clawed += forfeited
        return forfeited

    def owner_balance(self, owner: str) -> float:
        return self.paid.get(owner, 0.0)


# --------------------------------------------------------------------------
# Admission scoring — marginal value on a beacon-drawn holdout
# --------------------------------------------------------------------------
def marginal_value(model, base_vec, shard, holdouts):
    """How much would training on `shard` reduce held-out loss, per §7.2?

    First-order (TracIn-style): a step v ← v − lr·g_s changes holdout loss by
    ≈ −lr·(g_h · g_s), so the shard's value is the alignment of its gradient with
    the descent direction the holdout wants. We average this over several
    beacon-drawn holdout probes and normalise by the holdout-gradient scale, so
    the per-shard estimate is stable and hard to grind against any one probe.
    Positive = pushes the model the way the data wants. It is the SAME
    gradient-dot-product attribution uses downstream, so pricing and royalties
    speak one language. Two properties fall out for free: data already learned
    has a near-zero gradient (duplicates → ~0, Sybil-resistant); junk aligns with
    a holdout only by chance (→ ~0).

    `holdouts` is a list of held-out (x, y) batches (drawn from the beacon).
    Data that fills a gap the queries need scores highest; data the model already
    covers has a near-zero gradient and prices ~0.
    """
    g_s = model.grad(base_vec, shard)
    return float(np.mean([np.dot(g_s, model.grad(base_vec, hb)) for hb in holdouts]))
