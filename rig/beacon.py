"""Unbiasable randomness beacon — threshold BLS (drand-style) (WHITEPAPER §7.4).

`sha256(height)` is predictable and manipulable — anyone can compute future
draws, and a proposer could grind height-adjacent values. The beacon is the
root of the security tree (shard assignment, committee sampling, evaluation
draws all hang off it), so it must be:

  * unpredictable — no one can compute round r's value ahead of time,
  * unbiasable   — no party (not even the proposer) can steer it,
  * verifiable   — anyone can check a published value against a fixed public key.

Threshold BLS delivers all three, and is exactly what drand runs. A group key
is Shamir-shared across n nodes so any t can act but fewer learn nothing. Each
round r, nodes publish partial signatures over r; any t of them Lagrange-combine
into the group's unique BLS signature s·H(r). "Unique" is the whole point:
given the key and r the signature is fixed, so there is nothing to grind — and
computing it needs t shares, so it can't be predicted. The value is
hash(signature); verification is one pairing check against the group public key.

Setup here uses a trusted dealer (Shamir) for the rig; production runs a
distributed key generation so no single party ever holds the group secret.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
from py_ecc.bls.g2_primitives import G2_to_signature
from py_ecc.bls.hash_to_curve import hash_to_G2
from py_ecc.optimized_bls12_381 import (G1, add, curve_order, multiply, neg,
                                        normalize, pairing)

DST = b"SESTRIAN-BEACON-BLS12381-SHA256"


def _round_point(r: int):
    return hash_to_G2(f"sestrian-beacon|{r}".encode(), DST, hashlib.sha256)


def _lagrange_at_zero(indices: list[int]) -> dict[int, int]:
    """Lagrange coefficients λ_i (mod curve_order) to interpolate f(0) from the
    given evaluation points x=i."""
    coeffs = {}
    for i in indices:
        num, den = 1, 1
        for j in indices:
            if j == i:
                continue
            num = (num * (-j)) % curve_order
            den = (den * (i - j)) % curve_order
        coeffs[i] = (num * pow(den, -1, curve_order)) % curve_order
    return coeffs


@dataclass
class BeaconKeys:
    group_pk: object                 # G1 point: s·G1
    shares: dict                     # node index (1..n) -> scalar share
    partial_pks: dict                # node index -> G1 point share_i·G1
    t: int
    n: int


def deal(n: int, t: int, seed: bytes = b"sestrian-dealer") -> BeaconKeys:
    """Trusted-dealer Shamir setup: sample a degree-(t-1) polynomial f, the group
    secret is f(0); share_i = f(i). (Production replaces this with DKG.)"""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8], "big"))
    coeffs = [int.from_bytes(hashlib.sha256(seed + bytes([k])).digest(), "big") % curve_order
              for k in range(t)]                              # f = c0 + c1 x + ...
    secret = coeffs[0]
    shares, partial_pks = {}, {}
    for i in range(1, n + 1):
        s_i = 0
        for k, c in enumerate(coeffs):
            s_i = (s_i + c * pow(i, k, curve_order)) % curve_order
        shares[i] = s_i
        partial_pks[i] = multiply(G1, s_i)
    return BeaconKeys(group_pk=multiply(G1, secret), shares=shares,
                      partial_pks=partial_pks, t=t, n=n)


def partial_sign(share: int, r: int):
    """A node's partial signature over round r: share·H(r) in G2."""
    return multiply(_round_point(r), share)


def verify_partial(sig_i, r: int, partial_pk_i) -> bool:
    """e(sig_i, G1) == e(H(r), pk_i)  ⇔  sig_i = share_i·H(r)."""
    return pairing(sig_i, G1) == pairing(_round_point(r), partial_pk_i)


def combine(r: int, partials: dict):
    """Lagrange-combine ≥ t partial signatures into the group signature s·H(r)."""
    idx = list(partials.keys())
    lam = _lagrange_at_zero(idx)
    acc = None
    for i in idx:
        term = multiply(partials[i], lam[i] % curve_order)
        acc = term if acc is None else add(acc, term)
    return acc


def verify(group_sig, r: int, group_pk) -> bool:
    """e(group_sig, G1) == e(H(r), group_pk)  ⇔  group_sig = s·H(r)."""
    return pairing(group_sig, G1) == pairing(_round_point(r), group_pk)


def randomness(group_sig) -> bytes:
    """The beacon output for the round: hash of the canonical signature bytes."""
    return hashlib.sha256(G2_to_signature(group_sig)).digest()


def beacon_rng(group_sig, tag: str = "") -> np.random.Generator:
    """A seeded numpy Generator from a round's beacon value — the drop-in
    replacement for rig/chain.py's predictable `beacon(height, tag)`."""
    seed = hashlib.sha256(randomness(group_sig) + tag.encode()).digest()
    return np.random.default_rng(int.from_bytes(seed[:8], "big"))


def produce(keys: BeaconKeys, r: int, signer_indices: list[int]):
    """Full round: the given nodes each partial-sign r, verify, and combine.
    Returns (group_sig, verified) — verified is False if < t honest partials."""
    partials = {}
    for i in signer_indices:
        sig = partial_sign(keys.shares[i], r)
        if verify_partial(sig, r, keys.partial_pks[i]):       # reject bad partials
            partials[i] = sig
    if len(partials) < keys.t:
        return None, False
    gsig = combine(r, dict(list(partials.items())[:keys.t]))
    return gsig, verify(gsig, r, keys.group_pk)


if __name__ == "__main__":
    print("=" * 70)
    print("  SESTRIAN — unbiasable randomness beacon (threshold BLS)")
    print("=" * 70)
    n, t = 5, 3
    keys = deal(n, t)
    print(f"\n{t}-of-{n} threshold group key (trusted-dealer setup).")

    print("\nround  randomness (first 16 hex)   shard→miner assignment")
    for r in range(4):
        gsig, ok = produce(keys, r, [1, 2, 3, 4, 5])
        assert ok
        rng = beacon_rng(gsig, "shards")
        assignment = rng.integers(0, n, size=n).tolist()       # who trains which shard
        print(f"  {r}    {randomness(gsig).hex()[:16]}          {assignment}")

    print("\nunbiasability & fault tolerance:")
    # any t honest nodes reproduce the SAME value — a byzantine subset can't steer it
    g_a, _ = produce(keys, 7, [1, 2, 3])
    g_b, _ = produce(keys, 7, [3, 4, 5])
    same = randomness(g_a) == randomness(g_b)
    print(f"  two disjoint-ish honest quorums agree on round 7: {same}")
    # fewer than t shares cannot produce a valid value (unpredictable)
    _, ok2 = produce(keys, 7, [1, 2])
    print(f"  {t-1} shares can produce a valid beacon: {ok2}  (must be False)")
    # a single node lying is filtered; the honest majority still succeeds
    print("  a malicious partial is rejected by verify_partial; honest t-of-n proceed")
    print("=" * 70)
    raise SystemExit(0 if same and not ok2 else 1)
