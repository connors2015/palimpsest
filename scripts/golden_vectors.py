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
from rig.blockchain import BlockTree, Header, build_block, txset_root
from rig.chain import dequantize, quantize, state_root, trimmed_mean_int
from rig.crypto import BackpropTx, Key, delta_hash
from rig.token import (CHALLENGE_WINDOW, DataChallengeTx, DataSubmitTx,
                       DataVoteTx, TokenLedger, TransferTx, address,
                       data_root, emission, transfer_root)

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
    # OVERFLOW: three near-max int64 values sum past i64::MAX. numpy int64 sum
    # WRAPS (two's complement); the Rust node must wrap identically (wrapping_add,
    # not a debug-panicking `+`) or it forks / crashes on a crafted block.
    big = 1 << 62
    ovf = [np.array([big, -3], dtype=np.int64), np.array([big, 5], dtype=np.int64),
           np.array([big, 7], dtype=np.int64)]
    cases.append({"deltas": [d.tolist() for d in ovf],
                  "mean": trimmed_mean_int(ovf).tolist()})   # column 0 wraps
    # LOW-COUNT ROBUSTNESS: 3 deltas, one a large adversarial outlier per coord.
    # The >=1 trim (k>=3) drops it, leaving the honest middle — a plain mean
    # would have been dragged toward the outlier.
    adv = [np.array([10, 10], dtype=np.int64), np.array([12, 11], dtype=np.int64),
           np.array([100000, -100000], dtype=np.int64)]
    cases.append({"deltas": [d.tolist() for d in adv],
                  "mean": trimmed_mean_int(adv).tolist()})
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

    # --- BackpropTx: signing bytes / txid / signature (with a bond) -----------
    tx = BackpropTx(miner=key.pub, base_height=7, shard_id=3,
                    delta_hash=delta_hash(d.tobytes()),
                    da_pointer=f"da://{delta_hash(d.tobytes())}", bond=5_000_000,
                    data_refs=["bb" * 32, "aa" * 32, "aa" * 32]).signed(key)  # rev 5: unsorted+dup
    v["backprop_tx"] = [{
        "miner": tx.miner, "base_height": tx.base_height, "shard_id": tx.shard_id,
        "delta_hash": tx.delta_hash, "da_pointer": tx.da_pointer, "bond": tx.bond,
        "data_refs": tx.canonical_refs(),
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
               txset_root="ef" * 32, n_txs=2, work=1500, proposer=key.pub,
               transfer_root="12" * 32, ledger_root="34" * 32,
               data_root="56" * 32, vrf_proof="ab" * 32, score_root="77" * 32)
    v["header"] = [{
        "height": h.height, "prev_hash": h.prev_hash, "state_root": h.state_root,
        "txset_root": h.txset_root, "n_txs": h.n_txs, "work": h.work,
        "proposer": h.proposer, "transfer_root": h.transfer_root,
        "ledger_root": h.ledger_root, "data_root": h.data_root,
        "vrf_proof": h.vrf_proof, "score_root": h.score_root,
        "canonical_json": json.dumps(h.__dict__, sort_keys=True),
        "hash": h.block_hash(),
    }]

    # --- token: address derivation + emission schedule ------------------------
    v["address"] = [{"pub_hex": key.pub, "address": address(key.pub)}]
    # rev 6 schedule: 1M-block epochs, tail floor after epoch 9 (never zero)
    v["emission"] = [{"height": hh, "reward": emission(hh)}
                     for hh in (0, 1, 1_000_000, 1_000_001, 2_000_001,
                                9_000_001, 10_000_000, 10**12)]

    # --- token: reward split + transfer apply + canonical ledger root ---------
    k2 = Key.generate(b"golden-vector-seed-second-key-0!")
    led = TokenLedger()
    led.apply_reward(1, [key.pub, k2.pub], key.pub, [address(k2.pub)])
    root_after_reward = led.root()
    xfer = TransferTx(from_pub=key.pub, to_addr=address(k2.pub),
                      amount=led.balance(address(key.pub)) // 2, nonce=0).signed(key)
    assert led.apply_transfer(xfer)
    v["ledger"] = [{
        "miners": [key.pub, k2.pub], "proposer": key.pub,
        "data_addrs": [address(k2.pub)], "height": 1,
        "root_after_reward": root_after_reward,
        "transfer": {"from_pub": xfer.from_pub, "to_addr": xfer.to_addr,
                     "amount": xfer.amount, "nonce": xfer.nonce,
                     "signing_bytes_hex": xfer.signing_bytes().hex(),
                     "txid": xfer.txid(), "sig_hex": xfer.sig.hex()},
        "root_after_transfer": led.root(),
        "balances": dict(sorted(led.balances.items())),
        "transfer_root": transfer_root([xfer]),
    }]

    # --- data lane: submit -> challenge -> vote -> resolve, stepwise ----------
    kf = Key.generate(b"golden-data-founder-000000000000")
    led2 = TokenLedger()
    led2.seed_genesis_data(address(kf.pub))
    root_genesis = led2.root()
    led2.apply_reward(1, [key.pub], key.pub, [])          # fund key via mining
    sub = DataSubmitTx(owner_pub=key.pub, data_hash="aa" * 32, size_bytes=4096,
                       media_type="csv", stake=led2.balance(address(key.pub)) // 2,
                       nonce=0).signed(key)
    assert led2.apply_data_tx(sub, 1, set())
    root_after_submit = led2.root()
    led2.apply_reward(2, [k2.pub], k2.pub, [])            # fund k2 (the challenger)
    ch = DataChallengeTx(challenger_pub=k2.pub, data_id=sub.txid(),
                         stake=led2.balance(address(k2.pub)) // 4,
                         reason="validity", nonce=0).signed(k2)
    assert led2.apply_data_tx(ch, 2, set())
    # CHALLENGE_QUORUM (=3) DISINTERESTED jurors, each a recent proposer and
    # neither the owner (key) nor the challenger (k2), all vote to uphold — so
    # the quorum is met and the challenge is upheld (entry revoked, stake to
    # challenger). A single vote would now be below quorum and REJECTED.
    juror_keys = [Key.generate(b"golden-juror-%d" % i) for i in range(3)]
    juror_pubs = {jk.pub for jk in juror_keys}
    votes = []
    for jk in juror_keys:
        vt = DataVoteTx(voter_pub=jk.pub, challenge_id=ch.txid(),
                        support=True, nonce=0).signed(jk)
        assert led2.apply_data_tx(vt, 3, juror_pubs), "juror vote must apply"
        votes.append(vt)
    root_after_vote = led2.root()
    led2.resolve_expired_challenges(2 + CHALLENGE_WINDOW)  # quorum met -> upheld
    v["data_lane"] = [{
        "founder_pub": kf.pub,
        "root_genesis": root_genesis,
        "submit": {"owner_pub": sub.owner_pub, "data_hash": sub.data_hash,
                   "size_bytes": sub.size_bytes, "media_type": sub.media_type,
                   "stake": sub.stake, "nonce": sub.nonce,
                   "signing_bytes_hex": sub.signing_bytes().hex(),
                   "txid": sub.txid(), "sig_hex": sub.sig.hex()},
        "root_after_submit": root_after_submit,
        "challenge": {"challenger_pub": ch.challenger_pub, "data_id": ch.data_id,
                      "stake": ch.stake, "reason": ch.reason, "nonce": ch.nonce,
                      "txid": ch.txid(), "sig_hex": ch.sig.hex()},
        "votes": [{"voter_pub": vt.voter_pub, "challenge_id": vt.challenge_id,
                   "support": vt.support, "nonce": vt.nonce,
                   "txid": vt.txid(), "sig_hex": vt.sig.hex()} for vt in votes],
        "root_after_vote": root_after_vote,
        "resolve_height": 2 + CHALLENGE_WINDOW,
        "root_after_resolve": led2.root(),
        "challenger_balance_after": led2.balance(address(k2.pub)),
        "data_root_of_all": data_root([sub, ch] + votes),
    }]

    # --- delta admission bond: lock on inclusion, return at maturity ----------
    from rig.token import BOND_WINDOW
    ledb = TokenLedger()
    ledb.apply_reward(1, [key.pub], key.pub, [])          # fund the miner
    miner_addr = address(key.pub)
    bal0 = ledb.balance(miner_addr)
    bond_amt = bal0 // 2
    assert ledb.lock_bond("delta-tx-1", miner_addr, bond_amt, 1)
    root_locked, bal_locked = ledb.root(), ledb.balance(miner_addr)
    ledb.resolve_expired_bonds(1 + BOND_WINDOW)           # matured -> returned
    root_returned, bal_returned = ledb.root(), ledb.balance(miner_addr)
    assert not ledb.lock_bond("delta-tx-2", miner_addr, bal_returned + 1, 100)
    v["bond"] = [{
        "miner": key.pub, "bond": bond_amt, "bal_after_reward": bal0,
        "root_locked": root_locked, "bal_locked": bal_locked,
        "resolve_height": 1 + BOND_WINDOW,
        "root_returned": root_returned, "bal_returned": bal_returned,
    }]

    # --- fee-bearing inference receipt: a usage fee payer -> serving node -----
    from rig.token import InferenceReceiptTx
    ledr = TokenLedger()
    ledr.apply_reward(1, [key.pub], key.pub, [])          # fund the payer
    payer_addr, server = address(key.pub), address(k2.pub)
    fee = ledr.balance(payer_addr) // 10
    rcpt = InferenceReceiptTx(payer_pub=key.pub, server_addr=server, fee=fee,
                              output_hash="aa" * 32, head_root="bb" * 32,
                              nonce=0).signed(key)
    assert ledr.apply_data_tx(rcpt, 5, set())
    v["inference"] = [{
        "payer_pub": key.pub, "server_addr": server, "fee": fee,
        "output_hash": "aa" * 32, "head_root": "bb" * 32, "nonce": 0,
        "signing_bytes_hex": rcpt.signing_bytes().hex(), "txid": rcpt.txid(),
        "sig_hex": rcpt.sig.hex(),
        "payer_after": ledr.balance(payer_addr), "server_after": ledr.balance(server),
        "root_after": ledr.root(),
    }]

    # --- data provenance payout (rev 5): the data share pays the owners of the
    #     corpora a block NAMED, ∝ credit weight; an unnamed active corpus earns 0.
    owner_a = Key.generate(b"prov-owner-a-0000000000000000000")
    owner_b = Key.generate(b"prov-owner-b-0000000000000000000")
    miner_p = Key.generate(b"prov-miner-000000000000000000000")
    ledp = TokenLedger()
    for h_, ow in (("corpusA", owner_a), ("corpusB", owner_b)):
        ledp.registry[h_] = {"owner": address(ow.pub), "data_hash": h_, "size": 0,
                             "media_type": "text", "stake": 0, "weight": 1,
                             "status": "active"}
    prov_credits = {"corpusA": 1}                    # only corpus A was named this block
    ledp.apply_reward(5, [miner_p.pub], miner_p.pub, data_credits=prov_credits)
    v["data_provenance"] = [{
        "height": 5, "miner": miner_p.pub,
        "corpora": {"corpusA": address(owner_a.pub), "corpusB": address(owner_b.pub)},
        "data_credits": prov_credits,
        "owner_a": address(owner_a.pub), "owner_b": address(owner_b.pub),
        "balance_a": ledp.balance(address(owner_a.pub)),
        "balance_b": ledp.balance(address(owner_b.pub)),
        "miner_after": ledp.balance(address(miner_p.pub)),
        "root": ledp.root(),
    }]

    # --- fee flows (rev 6): 60/20/20 receipt split into the fee pools, then the
    #     next block's reward drains them to its miners + named data owners.
    ff_payer = Key.generate(b"fee-payer-0000000000000000000000")
    ff_server = Key.generate(b"fee-server-000000000000000000000")
    ff_miner = Key.generate(b"fee-miner-0000000000000000000000")
    ff_owner = Key.generate(b"fee-owner-0000000000000000000000")
    ledf = TokenLedger()
    ledf.apply_reward(1, [ff_payer.pub], "genesis", [])   # fund the payer
    ff_fee = 10_000
    ff_rcpt = InferenceReceiptTx(payer_pub=ff_payer.pub,
                                 server_addr=address(ff_server.pub), fee=ff_fee,
                                 output_hash="cc" * 32, head_root="dd" * 32,
                                 nonce=0).signed(ff_payer)
    assert ledf.apply_data_tx(ff_rcpt, 1, set())
    split_state = {
        "server_after": ledf.balance(address(ff_server.pub)),
        "fee_data_pool": ledf.fee_data_pool,
        "fee_train_pool": ledf.fee_train_pool,
        "root": ledf.root(),
    }
    ledf.registry["corpusF"] = {"owner": address(ff_owner.pub),
                                "data_hash": "corpusF", "size": 0,
                                "media_type": "text", "stake": 0, "weight": 1,
                                "status": "active"}
    ledf.apply_reward(2, [ff_miner.pub], "genesis", data_credits={"corpusF": 1})
    v["fee_flows"] = [{
        "payer_pub": ff_payer.pub, "server_addr": address(ff_server.pub),
        "fee": ff_fee, "output_hash": "cc" * 32, "head_root": "dd" * 32,
        "nonce": 0, "sig_hex": ff_rcpt.sig.hex(),
        "after_receipt": split_state,
        "owner_f": address(ff_owner.pub), "miner": ff_miner.pub,
        "after_drain": {
            "miner_after": ledf.balance(address(ff_miner.pub)),
            "owner_after": ledf.balance(address(ff_owner.pub)),
            "fee_data_pool": ledf.fee_data_pool,
            "fee_train_pool": ledf.fee_train_pool,
            "root": ledf.root(),
        },
    }]

    # --- rev 7 delta scoring: score-weighted miner split -----------------------
    sc_a = Key.generate(b"scoring-miner-a-0000000000000000")
    sc_b = Key.generate(b"scoring-miner-b-0000000000000000")
    led_s = TokenLedger()
    led_s.apply_reward(3, [sc_a.pub, sc_b.pub], "genesis",
                       miner_weights={sc_a.pub: 300_000, sc_b.pub: 100_000})
    led_z = TokenLedger()
    led_z.apply_reward(3, [sc_a.pub, sc_b.pub], "genesis",
                       miner_weights={})                 # all-zero -> equal split
    v["scoring"] = [{
        "miner_a": sc_a.pub, "miner_b": sc_b.pub, "height": 3,
        "weights": {sc_a.pub: 300_000, sc_b.pub: 100_000},
        "balance_a": led_s.balance(address(sc_a.pub)),
        "balance_b": led_s.balance(address(sc_b.pub)),
        "root": led_s.root(),
        "equal_balance_a": led_z.balance(address(sc_a.pub)),
        "equal_balance_b": led_z.balance(address(sc_b.pub)),
        "equal_root": led_z.root(),
    }]

    # --- DA: erasure coding + Merkle commitment (the Rust port must match) ----
    from rig import da as _da
    da_body = bytes((i * 7 + 3) % 256 for i in range(100))
    da_k, da_n = 4, 12
    blob = _da.disperse(da_body, da_k, da_n)
    keep = [1, 3, 6, 9]                                   # an arbitrary k-subset
    recon = _da.reconstruct({i: blob.shards[i] for i in keep}, da_k, blob.orig_len)
    assert recon == da_body, "reference reconstruct must round-trip"
    pf = blob.proof(5)
    v["da"] = [{
        "body_hex": da_body.hex(), "k": da_k, "n": da_n, "orig_len": len(da_body),
        "shards_hex": [s.hex() for s in blob.shards],
        "root_hex": blob.root.hex(),
        "reconstruct_from": keep,
        "proof_index": 5,
        "proof": [[side, sib.hex()] for side, sib in pf],
    }]

    # --- proposer lottery (§7.4 interim): verifiable stake-weighted sortition -
    from rig import lottery as _lot
    lot_prev, lot_h = "ab" * 32, 42
    lot_cases = []
    for stake, total in [(1_000_000_000, 1_000_000_000), (1, 10**18)]:
        proof = _lot.vrf_prove(key, lot_prev, lot_h)
        lot_cases.append({
            "pub": key.pub, "seed_hex": key.sk.hex(),
            "prev_hash": lot_prev, "height": lot_h,
            "stake": stake, "total_stake": total,
            "proof_sig_hex": proof.hex(),
            "vrf_output_hex": _lot.vrf_output(proof).to_bytes(32, "big").hex(),
            "eligible": _lot.eligible(key.pub, proof, lot_prev, lot_h, stake, total),
        })
    v["lottery"] = lot_cases

    # --- capacity retarget (§9.4a): the deterministic decision trace ----------
    from rig.capacity import CapacityRetarget
    ctrl = CapacityRetarget()
    fleet = [1.0, 1.2, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 0.5, 0.3, 0.3, 0.3, 1.0, 1.5, 2.0]
    mpfu, per_unit = 4.0, 8.0
    cap_windows = []
    for f in fleet:
        capacity = f * per_unit
        accepted = int(capacity / max(ctrl.quota, 1e-9))
        load = ctrl.active_modules / (mpfu * max(f, 1e-9))
        staleness = max(0.0, min(1.0, load - 1.0))
        accepted = int(accepted * (1.0 - staleness))
        dec = ctrl.observe_window(accepted, staleness)
        cap_windows.append({"accepted": accepted, "staleness": staleness, **dec})
    v["capacity"] = [{"windows": cap_windows}]

    # --- FULL-CHAIN REPLAY: a mini chain with a fork and settled transfers ----
    # Rust must rebuild every block, validate it completely (sigs, state
    # transition, txset/transfer/ledger roots), run fork choice, and land on the
    # same head with the same roots.
    dim = 16
    genesis_w = quantize(rng.standard_normal(dim))
    m0, m1 = Key.generate(b"chain-miner-0" + b"0" * 19), Key.generate(b"chain-miner-1" + b"0" * 19)
    founder = address(Key.generate(b"chain-founder" + b"0" * 19).pub)
    tree = BlockTree(genesis_w, data_contributor=founder)

    def mk_tx(miner_key, height, shard, delta):
        dh = delta_hash(delta.tobytes())
        # rev 5: name the always-active genesis corpus so the delta is provenanced
        return BackpropTx(miner=miner_key.pub, base_height=height, shard_id=shard,
                          delta_hash=dh, da_pointer=f"da://{dh}",
                          data_refs=["genesis"]).signed(miner_key)

    blocks_out = []

    def add(parent, miner_keys, proposer_key, transfers=(), data_txs=()):
        hh = tree.blocks[parent].header.height
        txs, bodies = [], {}
        for s, mk in enumerate(miner_keys):
            d = quantize(rng.standard_normal(dim) * 0.1)
            tx = mk_tx(mk, hh, s, d)
            txs.append(tx); bodies[tx.da_pointer] = d
        # rev 7: distinct nonzero committed scores lock score_root + the
        # score-weighted miner split + score-weighted data credits
        scr = {t.txid(): 100_000 * (i + 1) for i, t in enumerate(txs)}
        blk = build_block(tree, parent, txs, bodies,
                          {t.txid(): 1.0 for t in txs}, proposer_key,
                          transfers=list(transfers), data_txs=list(data_txs),
                          scores=scr)
        tree.add_block(blk)
        blocks_out.append({
            "parent": parent, "hash": blk.hash,
            "header": dict(blk.header.__dict__),
            "txs": [{"miner": t.miner, "base_height": t.base_height,
                     "shard_id": t.shard_id, "delta_hash": t.delta_hash,
                     "da_pointer": t.da_pointer, "bond": t.bond,
                     "data_refs": t.canonical_refs(), "sig_hex": t.sig.hex()}
                    for t in txs],
            "scores": blk.scores,
            "bodies": {p: b.tolist() for p, b in blk.bodies.items()},
            "transfers": [{"from_pub": t.from_pub, "to_addr": t.to_addr,
                           "amount": t.amount, "nonce": t.nonce,
                           "sig_hex": t.sig.hex()} for t in blk.transfers],
            "data_txs": [{"owner_pub": t.owner_pub, "data_hash": t.data_hash,
                          "size_bytes": t.size_bytes, "media_type": t.media_type,
                          "stake": t.stake, "nonce": t.nonce,
                          "sig_hex": t.sig.hex()} for t in blk.data_txs],
        })
        return blk.hash

    b1 = add(tree.genesis.hash, [m0, m1], m0)          # height 1: both mine
    # fund check: m0 has miner+proposer share now; send some to m1 in block 2
    pay = TransferTx(from_pub=m0.pub, to_addr=address(m1.pub),
                     amount=tree.ledger[b1].balance(address(m0.pub)) // 4,
                     nonce=0).signed(m0)
    b2 = add(b1, [m0], m1, transfers=[pay])            # height 2: transfer settles
    b2f = add(b1, [m1], m1)                            # FORK at height 2
    sub3 = DataSubmitTx(owner_pub=m0.pub, data_hash="bb" * 32, size_bytes=777,
                        media_type="image",
                        stake=tree.ledger[b2].balance(address(m0.pub)) // 3,
                        nonce=1).signed(m0)
    b3 = add(b2, [m0, m1], m0, data_txs=[sub3])        # heaviest chain + data lane
    v["chain_replay"] = [{
        "genesis_w": genesis_w.tolist(),
        "data_contributor": founder,
        "blocks": blocks_out,
        "expected_head": tree.head,
        "expected_head_height": tree.blocks[tree.head].header.height,
        "expected_state_root": state_root(tree.head_state()),
        "expected_ledger_root": tree.head_ledger().root(),
        "expected_supply": tree.head_ledger().supply(),
    }]
    # the head is now whichever chain fork choice (cumulative vrf_work) selects;
    # it is recorded above as expected_head and the Rust node must reproduce it.

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(v, f, indent=1)
    n = sum(len(x) for x in v.values())
    print(f"wrote {n} vectors across {len(v)} families -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
