"""Verifiable proposer sortition (§7.4, interim) — the SPEC for the Rust node.

Fixed round-robin rotation can't work on an open network (who is `id` 2 of n?)
and a single scheduled proposer being offline stalls the slot. This replaces it
with per-height, stake-weighted, *verifiable* eligibility:

  * The per-height seed binds to the parent hash and height.
  * A proposer's VRF proof is a DETERMINISTIC Ed25519 signature over that seed —
    only the key holder can produce it, anyone can verify it, and it is unique
    per (key, height). (Deterministic Ed25519 as a VRF; the threshold-BLS beacon
    in rig/beacon.py is the unbiasable upgrade that also removes the proposer's
    ability to grind the parent hash.)
  * Eligibility: the proof's hash, as a fraction of 2^256, must fall below
    TARGET_PROPOSERS × (stake / total_stake) — so eligibility is stake-weighted
    and, in expectation, TARGET_PROPOSERS nodes qualify per height. Multiple or
    zero may qualify in a given slot; heaviest-valid-chain fork choice settles
    the rest, and no single node's absence can stall the chain.

Deterministic and self-verifying: given the header's VRF proof, every node
recomputes the same eligibility decision.
"""

import hashlib

from .crypto import verify

TARGET_PROPOSERS = 2          # expected eligible proposers per height
_TWO256 = 1 << 256


def seed(prev_hash: str, height: int) -> bytes:
    return hashlib.sha256(f"palimpsest-lottery|{prev_hash}|{height}".encode()).digest()


def vrf_prove(key, prev_hash: str, height: int) -> bytes:
    """The proposer's VRF proof: a deterministic signature over the seed."""
    return key.sign(seed(prev_hash, height))


def vrf_output(proof: bytes) -> int:
    """A uniform value in [0, 2^256) that only the key holder could have produced."""
    return int.from_bytes(hashlib.sha256(proof).digest(), "big")


def vrf_work(proof: bytes) -> int:
    """Fork-choice weight from the VRF output: leading zero bits + 1 (>= 1). A
    luckier (smaller) output yields more work, so among the eligible proposers of
    a height the luckiest wins — and work is NON-FORGEABLE (one VRF per proposer
    per height), which is what makes header.work safe to trust."""
    out = vrf_output(proof)
    lz = 256 - out.bit_length() if out > 0 else 256
    return lz + 1


def threshold(stake: int, total_stake: int) -> int:
    """Eligible iff vrf_output < 2^256 · TARGET · stake/total_stake."""
    if total_stake <= 0 or stake <= 0:
        return 0
    return min(_TWO256, _TWO256 * TARGET_PROPOSERS * stake // total_stake)


def eligible(pub: str, proof: bytes, prev_hash: str, height: int,
             stake: int, total_stake: int) -> bool:
    """True iff `proof` is a valid VRF proof by `pub` for this height AND its
    output falls under the stake-weighted threshold."""
    if not verify(pub, seed(prev_hash, height), proof):
        return False
    return vrf_output(proof) < threshold(stake, total_stake)
