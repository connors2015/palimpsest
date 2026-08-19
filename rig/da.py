"""Data-availability layer: erasure coding + availability sampling (§3.3).

The rig modelled DA as "the body travels with its tx" — withholding was caught
only because the body was present to hash. A real DA layer must let the network
be *sure a body is retrievable* without any one node downloading it, and stay
recoverable when some holders vanish or lie. This is the primitive Celestia
built and the whitepaper's §3.3 calls for.

Two pieces:

  * **Erasure coding** (Reed-Solomon over GF(256)). A delta body is split into
    k data-carrying symbols per column and expanded to n shards such that ANY k
    of the n reconstruct it. An adversary must therefore withhold more than n−k
    shards to make the body unrecoverable.
  * **Availability sampling.** The n shards are Merkle-committed; the root is the
    tx's DA pointer. A verifier requests a few random shards with Merkle proofs.
    Because unrecoverability requires >n−k shards missing, a handful of random
    samples detects a withholding attack with high probability — without ever
    fetching the whole body.

GF(256) is implemented here (≈ the AES field) so the rig stays dependency-free.
"""

import hashlib
from dataclasses import dataclass

from . import merkle

# --------------------------------------------------------------------------
# GF(256) — the AES field, generator 0x03, modulus x^8+x^4+x^3+x+1 (0x11b)
# --------------------------------------------------------------------------
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x ^= (_x << 1)
    if _x & 0x100:
        _x ^= 0x11b
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _inv(a):
    return _EXP[255 - _LOG[a]]


def _mat_inv(m):
    """Invert a k×k GF(256) matrix by Gauss-Jordan; returns None if singular."""
    k = len(m)
    a = [row[:] + [1 if i == j else 0 for j in range(k)] for i, row in enumerate(m)]
    for col in range(k):
        piv = next((r for r in range(col, k) if a[r][col] != 0), None)
        if piv is None:
            return None
        a[col], a[piv] = a[piv], a[col]
        inv = _inv(a[col][col])
        a[col] = [_mul(v, inv) for v in a[col]]
        for r in range(k):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [x ^ _mul(f, y) for x, y in zip(a[r], a[col])]
    return [row[k:] for row in a]


def _vandermonde(n, k):
    """n×k Vandermonde: row i uses evaluation point (i+1); any k rows invertible."""
    return [[_EXP[(_LOG[i + 1] * j) % 255] if (i + 1) != 0 else (1 if j == 0 else 0)
             for j in range(k)] for i in range(n)]


# --------------------------------------------------------------------------
# Erasure coding
# --------------------------------------------------------------------------
def encode(body: bytes, k: int, n: int) -> list[bytes]:
    """Split `body` into k data rows and expand to n shards (any k reconstruct)."""
    assert 0 < k <= n <= 255
    pad = (-len(body)) % k
    data = body + b"\x00" * pad
    L = len(data) // k
    rows = [data[r * L:(r + 1) * L] for r in range(k)]        # k rows, L bytes each
    V = _vandermonde(n, k)
    shards = []
    for i in range(n):
        vi = V[i]
        out = bytearray(L)
        for col in range(L):
            acc = 0
            for r in range(k):
                acc ^= _mul(vi[r], rows[r][col])
            out[col] = acc
        shards.append(bytes(out))
    return shards


def reconstruct(shards: dict, k: int, orig_len: int) -> bytes:
    """Recover the body from any k shards ({index: bytes}); raises if < k."""
    if len(shards) < k:
        raise ValueError(f"need {k} shards, have {len(shards)}")
    idx = sorted(shards)[:k]
    V = _vandermonde(max(idx) + 1, k)
    sub = [V[i] for i in idx]
    inv = _mat_inv(sub)
    if inv is None:
        raise ValueError("singular shard set (should not happen for distinct rows)")
    L = len(shards[idx[0]])
    rows = []
    for r in range(k):
        row = bytearray(L)
        for col in range(L):
            acc = 0
            for c in range(k):
                acc ^= _mul(inv[r][c], shards[idx[c]][col])
            row[col] = acc
        rows.append(bytes(row))
    return b"".join(rows)[:orig_len]


# --------------------------------------------------------------------------
# DA blob: erasure-coded, Merkle-committed, samplable
# --------------------------------------------------------------------------
@dataclass
class DABlob:
    shards: list
    orig_len: int
    k: int
    n: int
    levels: list                     # merkle tree over shards

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, i: int):
        return merkle.proof(self.levels, i)


def disperse(body: bytes, k: int, n: int) -> DABlob:
    shards = encode(body, k, n)
    return DABlob(shards=shards, orig_len=len(body), k=k, n=n,
                  levels=merkle.build(shards))


def verify_shard(shard: bytes, i: int, proof, root: bytes) -> bool:
    """A sampled shard is authentic iff its Merkle proof checks against root."""
    return merkle.verify(shard, i, proof, root)


def sample_available(available: dict, blob: DABlob, num_samples, rng) -> bool:
    """Availability sampling: request `num_samples` random shards; the body is
    deemed available iff every sampled shard is present AND proves against root.
    Unrecoverability needs > n−k missing, so a few samples catch withholding
    with high probability."""
    picks = rng.choice(blob.n, size=min(num_samples, blob.n), replace=False)
    for i in picks:
        i = int(i)
        if i not in available:
            return False                       # a sampled shard is missing
        if not verify_shard(available[i], i, blob.proof(i), blob.root):
            return False                       # …or forged
    return True


def detection_probability(available_count, n, k, num_samples) -> float:
    """P(random sampling hits at least one missing shard) when the body is
    unrecoverable (available_count ≤ n−k)."""
    from math import comb
    miss = n - available_count
    if miss <= 0 or num_samples > available_count:
        return 1.0
    p_all_present = comb(available_count, num_samples) / comb(n, num_samples)
    return 1.0 - p_all_present


def da_pointer(root: bytes) -> str:
    return "da://" + root.hex()[:32]


if __name__ == "__main__":
    import numpy as np
    print("=" * 70)
    print("  PALIMPSEST — data-availability layer (erasure coding + sampling)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    body = rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes()
    k, n = 4, 12
    blob = disperse(body, k, n)
    print(f"\ndelta body {len(body)} B  ->  {n} shards of {len(blob.shards[0])} B "
          f"(any {k} reconstruct)")
    print(f"  DA commitment (Merkle root): {da_pointer(blob.root)}")

    # reconstruct from an arbitrary k-subset
    keep = sorted(rng.choice(n, size=k, replace=False).tolist())
    got = reconstruct({i: blob.shards[i] for i in keep}, k, blob.orig_len)
    print(f"  reconstruct from shards {keep}: exact = {got == body}")

    # availability sampling on a healthy blob
    all_avail = {i: blob.shards[i] for i in range(n)}
    ok = sample_available(all_avail, blob, num_samples=4, rng=np.random.default_rng(1))
    print(f"\n  sampling a fully-available blob (4 samples): available = {ok}")

    # withholding attack: adversary serves only k-1 shards (unrecoverable — a
    # body needs any k, so fewer than k available can never be reconstructed)
    serve = k - 1
    withheld = {i: blob.shards[i] for i in range(serve)}
    caught = 0
    trials = 400
    for s in range(trials):
        if not sample_available(withheld, blob, num_samples=4,
                                rng=np.random.default_rng(1000 + s)):
            caught += 1
    p = detection_probability(serve, n, k, 4)
    print(f"  withholding attack ({serve}/{n} shards, unrecoverable):")
    print(f"    reconstruct fails: ", end="")
    try:
        reconstruct(withheld, k, blob.orig_len); print("NO (bug)")
    except ValueError:
        print("yes")
    print(f"    sampling caught it {caught}/{trials} times "
          f"(theory {p*100:.0f}% per check)")

    # a forged shard is rejected by its Merkle proof
    forged = dict(all_avail)
    forged[3] = bytes(len(blob.shards[3]))
    bad = sample_available(forged, blob, num_samples=n, rng=np.random.default_rng(7))
    print(f"\n  forged shard rejected by Merkle proof: {not bad}")
    print("=" * 70)
    ok_all = (got == body and ok and caught > trials * 0.9 and not bad)
    raise SystemExit(0 if ok_all else 1)
