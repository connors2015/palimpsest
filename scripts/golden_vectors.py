"""Generate golden test vectors from the Python reference implementation.

The Python rig is the SPEC. The Rust node must reproduce every vector in this
file bit-exactly before it is allowed near consensus. Regenerate with:

    python -m scripts.golden_vectors        # writes node/vectors/golden.json

Covered: fixed-point quantization (including numpy's round-half-to-EVEN — the
subtle one), state roots, trimmed-mean aggregation (including negative floor
division), delta hashing, payload decompression, Ed25519 identities and
signatures (RFC 8032 — deterministic), BackpropTx signing bytes / txid / sig,
txset roots, and header hashing (which must reproduce Python's
json.dumps(sort_keys=True) byte-for-byte).

Compression SELECTION (top-k tie-breaking) is deliberately NOT a vector: each
miner compresses only its own delta and commits to the decompressed dense form,
so selection is transport, not consensus. DEcompression is consensus-adjacent
(everyone runs it) and is covered.
"""

import json
import os

import numpy as np

from client.compress import compress, decompress
from rig.blockchain import Header, txset_root
from rig.chain import dequantize, quantize, state_root, trimmed_mean_int
from rig.crypto import BackpropTx, Key, delta_hash

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "node", "vectors", "golden.json")


def main():
    rng = np.random.default_rng(42)
    v = {}

    # --- quantize: float64 -> int64, np.round = HALF TO EVEN ------------------
    tricky = np.array([0.5 / (1 << 16), 1.5 / (1 << 16), 2.5 / (1 << 16),
                       -0.5 / (1 << 16), -1.5 / (1 << 16),
                       0.0, 1.0, -1.0, 0.1234567, -3.75])
    rand = rng.standard_normal(32) * 0.01
    v["quantize"] = [
        {"f": arr.tolist(), "q": quantize(arr).tolist()}
        for arr in (tricky, rand)
    ]

    # --- state_root: sha256 over little-endian int64 bytes --------------------
    w = quantize(rng.standard_normal(64))
    v["state_root"] = [
        {"w": w.tolist(), "root": state_root(w)},
        {"w": [0, -1, 1, 2**40, -(2**40)],
         "root": state_root(np.array([0, -1, 1, 2**40, -(2**40)], dtype=np.int64))},
    ]

    # --- trimmed_mean_int: sort/trim/floor-div (floor, NOT truncation) --------
    cases = []
    for k in (1, 3, 5):
        ds = [quantize(rng.standard_normal(16) * 0.01) for _ in range(k)]
        cases.append({"deltas": [d.tolist() for d in ds],
                      "mean": trimmed_mean_int(ds).tolist()})
    neg = [np.array([-7, -7, 5], dtype=np.int64), np.array([-8, 3, 5], dtype=np.int64),
           np.array([2, -7, -9], dtype=np.int64)]
    cases.append({"deltas": [d.tolist() for d in neg],
                  "mean": trimmed_mean_int(neg).tolist()})   # exercises floor(-x/k)
    v["trimmed_mean"] = cases

    # --- delta_hash over canonical int64 bytes --------------------------------
    d = quantize(rng.standard_normal(16) * 0.01)
    v["delta_hash"] = [{"delta": d.tolist(), "hash": delta_hash(d.tobytes())}]

    # --- decompress: payload -> dense int64 -----------------------------------
    delta = rng.standard_normal(256) * 0.01
    payload = compress(delta, keep_frac=0.05)
    v["decompress"] = [{
        "n": payload["n"],
        "idx_hex": payload["idx"].hex() if isinstance(payload["idx"], bytes) else bytes(payload["idx"]).hex(),
        "val_hex": payload["val"].hex() if isinstance(payload["val"], bytes) else bytes(payload["val"]).hex(),
        "dense": decompress(payload).tolist(),
    }]

    # --- Ed25519: seed -> pub; deterministic signature ------------------------
    key = Key.generate(b"golden-vector-seed-0123456789ab")
    msg = b"palimpsest golden message"
    v["ed25519"] = [{"seed_hex": key.sk.hex(), "pub_hex": key.pub,
                     "msg_hex": msg.hex(), "sig_hex": key.sign(msg).hex()}]

    # --- BackpropTx: signing bytes / txid / signature -------------------------
    tx = BackpropTx(miner=key.pub, base_height=7, shard_id=3,
                    delta_hash=delta_hash(d.tobytes()),
                    da_pointer=f"da://{delta_hash(d.tobytes())}").signed(key)
    v["backprop_tx"] = [{
        "miner": tx.miner, "base_height": tx.base_height, "shard_id": tx.shard_id,
        "delta_hash": tx.delta_hash, "da_pointer": tx.da_pointer,
        "signing_bytes_hex": tx.signing_bytes().hex(),
        "txid": tx.txid(), "sig_hex": tx.sig.hex(), "verifies": tx.verify(),
    }]

    # --- txset_root: sorted txids joined with '|' -----------------------------
    txs = []
    for i in range(3):
        di = quantize(rng.standard_normal(8) * 0.01)
        txs.append(BackpropTx(miner=key.pub, base_height=1, shard_id=i,
                              delta_hash=delta_hash(di.tobytes()),
                              da_pointer=f"da://{delta_hash(di.tobytes())}").signed(key))
    v["txset_root"] = [{"txids": [t.txid() for t in txs], "root": txset_root(txs)}]

    # --- header hash: python json.dumps(sort_keys=True) byte format -----------
    h = Header(height=5, prev_hash="ab" * 32, state_root="cd" * 32,
               txset_root="ef" * 32, n_txs=2, work=1500, proposer=key.pub)
    v["header"] = [{
        "height": h.height, "prev_hash": h.prev_hash, "state_root": h.state_root,
        "txset_root": h.txset_root, "n_txs": h.n_txs, "work": h.work,
        "proposer": h.proposer,
        "canonical_json": json.dumps(h.__dict__, sort_keys=True),
        "hash": h.block_hash(),
    }]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(v, f, indent=1)
    n = sum(len(x) for x in v.values())
    print(f"wrote {n} vectors across {len(v)} families -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
