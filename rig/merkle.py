"""A binary Merkle tree over weight pages, with inclusion proofs (§3.1).

The chain commits a `weights_state_root` = Merkle root over the model's pages.
A node that holds (or streams) only a few pages can prove those pages belong
to the committed model with an inclusion proof — O(log P) hashes — without
touching the rest. This is what lets a serving node attest an inference while
loading only the pages it actually used (rig/moe.py, WHITEPAPER §8).
"""

import hashlib


def _h(*parts: bytes) -> bytes:
    m = hashlib.sha256()
    for p in parts:
        m.update(p)
    return m.digest()


def leaf_hash(page_bytes: bytes) -> bytes:
    return _h(b"\x00", page_bytes)          # domain-separated leaf


def _node_hash(a: bytes, b: bytes) -> bytes:
    return _h(b"\x01", a, b)                 # domain-separated internal node


def build(leaves: list[bytes]) -> list[list[bytes]]:
    """Return the tree as levels[0]=leaf hashes .. levels[-1]=[root].

    An odd node on a level is promoted (hashed with itself) — standard.
    """
    assert leaves, "need at least one leaf"
    level = [leaf_hash(x) for x in leaves]
    levels = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(_node_hash(a, b))
        levels.append(nxt)
        level = nxt
    return levels


def root(leaves: list[bytes]) -> bytes:
    return build(leaves)[-1][0]


def proof(levels: list[list[bytes]], index: int) -> list[tuple[str, bytes]]:
    """Inclusion proof for leaf `index`: list of (side, sibling_hash)."""
    path, idx = [], index
    for level in levels[:-1]:
        if idx % 2 == 0:
            sib = level[idx + 1] if idx + 1 < len(level) else level[idx]
            path.append(("R", sib))
        else:
            path.append(("L", level[idx - 1]))
        idx //= 2
    return path


def verify(page_bytes: bytes, index: int, path, expected_root: bytes) -> bool:
    """Recompute the root from a page + its proof; compare to the commitment."""
    h = leaf_hash(page_bytes)
    for side, sib in path:
        h = _node_hash(sib, h) if side == "L" else _node_hash(h, sib)
    return h == expected_root
