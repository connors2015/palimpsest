"""rev 8 — the production influence-sketch projection (§8).

attribution.py's float Projector materializes a (dim × n_params) ±1 matrix —
fine for the small DomainModel sims, impossible at 85M params (21GB). The
production projection is IMPLICIT and INTEGER:

    sign bit j of a parameter index i  =  bit j of sha256(seed_le || i_le)
    sketch[j] = clamp_i32( Σ_i sign_ij · delta_i )

summed over the delta's NONZERO entries only (top-k sparse, ~1-2% of the
model), so it costs one sha256 per nonzero index. Pure integer arithmetic —
a delta's committed sketch is EXACTLY recomputable from its DA body on any
machine (no float tolerance needed), and dot products of sketches approximate
dot products of the underlying gradients (Johnson–Lindenstrauss for ±1
projections). Answer sketches use the same projection over the answer's
loss-gradient (float → quantized before projection), so they remain a
challengeable server claim rather than an exact recompute.
"""

import hashlib
import struct

SKETCH_DIM = 256                   # = one sha256 digest of sign bits per index
SKETCH_SEED = 1234                 # published projection seed
I32_MAX = 2**31 - 1


def _signs(i: int, seed: int = SKETCH_SEED) -> int:
    """256 sign bits for parameter index i, as an int (bit j = sign of dim j)."""
    return int.from_bytes(
        hashlib.sha256(struct.pack("<QQ", seed, i)).digest(), "little")


def sketch_sparse(indices, values, seed: int = SKETCH_SEED) -> list:
    """Integer influence sketch of a sparse vector: for each of the 256 output
    dims, the ±1-weighted sum of the nonzero values. Deterministic, exact —
    the numpy fast path is bit-identical to the pure-python fallback (little-
    endian bit order on the little-endian digest matches (bits >> j) & 1)."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        idx = [int(i) for i, v in zip(indices, values) if int(v) != 0]
        val = np.array([int(v) for v in values if int(v) != 0], dtype=np.int64)
        if len(idx) == 0:
            return [0] * SKETCH_DIM
        digests = b"".join(
            hashlib.sha256(struct.pack("<QQ", seed, i)).digest() for i in idx)
        bits = np.unpackbits(np.frombuffer(digests, dtype=np.uint8)
                             .reshape(len(idx), 32), axis=1, bitorder="little")
        signs = bits.astype(np.int64) * 2 - 1              # (k, 256) of ±1
        acc = (signs * val[:, None]).sum(axis=0)
        return [int(max(-I32_MAX - 1, min(I32_MAX, x))) for x in acc]
    acc = [0] * SKETCH_DIM
    for i, v in zip(indices, values):
        v = int(v)
        if v == 0:
            continue
        bits = _signs(int(i), seed)
        for j in range(SKETCH_DIM):
            acc[j] += v if (bits >> j) & 1 else -v
    return [max(-I32_MAX - 1, min(I32_MAX, x)) for x in acc]


def sketch_dense(vec, seed: int = SKETCH_SEED) -> list:
    """Sketch of a dense int vector (skips zeros — identical to sketch_sparse
    over its nonzero support)."""
    idx = [i for i, v in enumerate(vec) if int(v) != 0]
    return sketch_sparse(idx, [int(vec[i]) for i in idx], seed)


def dot(a: list, b: list) -> int:
    """Exact big-int dot — the alignment used by the royalty split."""
    return sum(int(x) * int(y) for x, y in zip(a, b))
