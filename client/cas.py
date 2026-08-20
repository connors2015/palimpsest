"""Content-addressed storage + Bitswap-style exchange — the IPFS layer (§3.3).

The chain commits HASHES (delta bodies by delta_hash, weights by page-Merkle
root). Content addressing turns those hashes into ADDRESSES: a block's id is its
own hash (its CID), so any node can ask the swarm "who has CID x?" and fetch it
from whoever does — no central server, no trust in the server. This is exactly
IPFS's model, and we mirror its two moving parts:

  * ContentStore — a CID -> bytes store. cid(data) = sha256(data) hex, the SAME
    hash the chain already speaks (rig.crypto.delta_hash, blockchain._sha), so a
    da://<hash> pointer IS a CID and nothing else has to change.

  * Bitswap — the block exchange. A node announces HAVE(cid), requests WANT(cid),
    and answers a want with BLOCK(cid, bytes). Content addressing makes it
    trustless: a received block is accepted only if cid(bytes) == the requested
    cid, so a peer physically cannot serve a forgery — integrity is free, no
    signature needed on the body.

And the model itself becomes an IPFS-style Merkle DAG: each page (a backbone
page or one expert, from client/moe.PageMap) is stored as its own object with
its own CID; a small manifest lists those CIDs in order; the manifest's CID is
the model ROOT. Fetching the model = fetch the root, then fetch each page CID
from the swarm — the same way IPFS fetches a large file as a DAG of chunks. So
"the model is a distributed content-addressed object" is literal: each PAGE is
an IPFS object. (A page, not a single weight — a scalar is 8 bytes but its CID
is 32+, so per-weight addressing is ~all overhead; pages of thousands of weights
are the unit where content addressing pays. Per-weight fidelity lives in the
TRAINING layer, PageMap.subdivide, not the storage layer.)

This is transport-agnostic: Bitswap emits ("want"/"have"/"block", …) messages
and the caller ships them over whatever transport (client/gossip sockets today,
real libp2p later). It replaces the hand-rolled getblock/getdata with a proper
content-addressed exchange.
"""

import hashlib
import json

import numpy as np


def cid(data: bytes) -> str:
    """Content id — sha256 hex, identical to the chain's body/Merkle hashing."""
    return hashlib.sha256(data).hexdigest()


class ContentStore:
    """A local content-addressed block store (an IPFS blockstore)."""

    def __init__(self):
        self.blocks = {}                                   # cid -> bytes

    def put(self, data: bytes) -> str:
        c = cid(data)
        self.blocks[c] = data
        return c

    def get(self, c: str):
        return self.blocks.get(c)

    def has(self, c: str) -> bool:
        return c in self.blocks

    def __len__(self):
        return len(self.blocks)


class Bitswap:
    """IPFS-style block exchange over any transport. Each handler returns the
    messages to SEND in response; the caller ships them. Content addressing is
    what makes on_block trustless — a forged body has the wrong CID and is
    dropped, so bodies need no signature (the tx that references the CID is
    signed; the body verifies itself)."""

    def __init__(self, store: ContentStore):
        self.store = store
        self.wantlist = set()                              # cids we still need
        self.providers = {}                                # cid -> set(peer) announced

    def want(self, c: str):
        """Ask the swarm for a block we don't have."""
        if self.store.has(c):
            return []
        self.wantlist.add(c)
        return [("want", c)]

    def on_want(self, peer, c: str):
        """A peer wants c — serve it if we hold it."""
        if self.store.has(c):
            return [("block", c, self.store.get(c))]
        return []

    def on_have(self, peer, c: str):
        """A peer announces it holds c — remember the provider, and if we want it,
        request it from them."""
        self.providers.setdefault(c, set()).add(peer)
        return [("want", c)] if c in self.wantlist else []

    def on_block(self, c: str, data: bytes):
        """Receive a block. Accept ONLY if it matches its CID (trustless). Returns
        (messages_to_send, accepted): on accept we re-announce HAVE so the block
        keeps propagating through the swarm."""
        if cid(data) != c:                                 # forgery / corruption
            return [], False
        self.store.put(data)
        self.wantlist.discard(c)
        return [("have", c)], True

    def announce(self, c: str):
        """Tell peers we hold a block (a provide record)."""
        return [("have", c)]


# --------------------------------------------------------------------------
# The model as an IPFS-style Merkle DAG of page objects
# --------------------------------------------------------------------------
_DTYPE = np.int64                                          # the chain's weight dtype


def pages_from_state(state_int: np.ndarray, page_spans):
    """Slice a flat weight vector (int64, the chain state) into page byte-blobs,
    one per span from client/moe.PageMap.subdivide(). Each blob is an IPFS object."""
    return [state_int[s:e].astype(_DTYPE).tobytes() for (s, e) in page_spans]


def state_from_pages(page_blobs, page_spans, n) -> np.ndarray:
    """Reassemble the flat weight vector from its page blobs (inverse of above)."""
    out = np.zeros(n, dtype=_DTYPE)
    for blob, (s, e) in zip(page_blobs, page_spans):
        out[s:e] = np.frombuffer(blob, dtype=_DTYPE)
    return out


def put_model(store: ContentStore, state_int: np.ndarray, page_spans):
    """Store the model as a DAG: each page → its own CID, plus a manifest object
    listing (span, page-cid) in order. Returns (root_cid, page_cids). The root
    commits the whole model; fetching it yields the manifest, then each page."""
    blobs = pages_from_state(state_int, page_spans)
    page_cids = [store.put(b) for b in blobs]
    manifest = json.dumps({
        "n": int(state_int.size),
        "pages": [[int(s), int(e), c] for (s, e), c in zip(page_spans, page_cids)],
    }, sort_keys=True).encode()
    root = store.put(manifest)
    return root, page_cids


def read_manifest(store: ContentStore, root_cid: str):
    """Parse a model root object into (n, spans, page_cids). Returns None if the
    root isn't in the store yet (fetch it first)."""
    raw = store.get(root_cid)
    if raw is None:
        return None
    m = json.loads(raw)
    spans = [(s, e) for s, e, _ in m["pages"]]
    page_cids = [c for _, _, c in m["pages"]]
    return m["n"], spans, page_cids


def get_model(store: ContentStore, root_cid: str):
    """Reconstruct the flat weight vector from the store, IF every page it needs
    is present. Returns (state, missing_cids): state is None while any page (or
    the manifest) is still missing — feed missing_cids to Bitswap.want()."""
    parsed = read_manifest(store, root_cid)
    if parsed is None:
        return None, [root_cid]
    n, spans, page_cids = parsed
    missing = [c for c in page_cids if not store.has(c)]
    if missing:
        return None, missing
    blobs = [store.get(c) for c in page_cids]
    return state_from_pages(blobs, spans, n), []
