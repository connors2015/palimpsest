"""Bitcoin-style blocks: validation, replay, fork choice, tamper rejection (§3, §5)."""

import numpy as np
import pytest

from rig.blockchain import (BlockTree, ValidationError, build_block, txset_root)
from rig.chain import quantize, state_root
from rig.crypto import BackpropTx, Key, delta_hash

DIM = 40


def _miners(n=3):
    return [Key.generate(f"m{i}".encode().ljust(32, b"0")) for i in range(n)]


def _block(tree, parent, height, miners, rng, work=1.0, tag=""):
    txs, bodies, works = [], {}, {}
    for i, k in enumerate(miners):
        body = quantize(rng.standard_normal(DIM) * 0.1)
        ptr = f"da://{tag}{height}/{i}"
        tx = BackpropTx(miner=k.pub, base_height=height - 1, shard_id=i,
                        delta_hash=delta_hash(body.tobytes()), da_pointer=ptr).signed(k)
        txs.append(tx); bodies[ptr] = body; works[tx.txid()] = work
    return build_block(tree, parent, txs, bodies, works, miners[0].pub)


def _grow(tree, n, miners, seed=1, work=1.0, tag=""):
    rng = np.random.default_rng(seed)
    head = tree.head
    for h in range(1, n + 1):
        b = _block(tree, head, tree.blocks[head].header.height + 1, miners, rng, work, tag)
        tree.add_block(b)
        head = b.hash
    return head


def test_chain_builds_and_replays_bit_exact():
    tree = BlockTree(quantize(np.zeros(DIM)))
    _grow(tree, 6, _miners())
    assert tree.blocks[tree.head].header.height == 6
    assert state_root(tree.replay_head()) == tree.blocks[tree.head].header.state_root


def test_heaviest_valid_chain_wins():
    m = _miners()
    tree = BlockTree(quantize(np.zeros(DIM)))
    _grow(tree, 5, m, seed=1, work=1.0, tag="A")
    light_head = tree.head
    # a shorter-in-work fork that is heavier per block, from block 2
    fork_parent = tree.chain_from_genesis()[1].hash
    rng = np.random.default_rng(9)
    fh = fork_parent
    for h in range(3, 9):
        b = _block(tree, fh, h, m, rng, work=10.0, tag="B")
        tree.add_block(b); fh = b.hash
    assert tree.head != light_head
    assert tree.blocks[tree.head].header.proposer == m[0].pub


def test_forged_signature_block_rejected():
    m = _miners()
    tree = BlockTree(quantize(np.zeros(DIM)))
    rng = np.random.default_rng(3)
    body = quantize(rng.standard_normal(DIM) * 0.1)
    ptr = "da://x"
    tx = BackpropTx(miner=m[0].pub, base_height=0, shard_id=0,
                    delta_hash=delta_hash(body.tobytes()), da_pointer=ptr)
    tx.sig = m[1].sign(tx.signing_bytes())            # wrong signer
    bad = build_block(tree, tree.head, [tx], {ptr: body}, {tx.txid(): 1.0}, m[0].pub)
    with pytest.raises(ValidationError):
        tree.add_block(bad)


def test_withheld_or_forged_body_rejected():
    m = _miners()
    tree = BlockTree(quantize(np.zeros(DIM)))
    rng = np.random.default_rng(4)
    body = quantize(rng.standard_normal(DIM) * 0.1)
    ptr = "da://y"
    tx = BackpropTx(miner=m[0].pub, base_height=0, shard_id=0,
                    delta_hash=delta_hash(body.tobytes()), da_pointer=ptr).signed(m[0])
    wrong = quantize(rng.standard_normal(DIM) * 0.1)  # body doesn't match its hash
    bad = build_block(tree, tree.head, [tx], {ptr: wrong}, {tx.txid(): 1.0}, m[0].pub)
    with pytest.raises(ValidationError):
        tree.add_block(bad)


def test_orphan_rejected():
    m = _miners()
    # a block built on tree1 at height 2 (parent = tree1's block 1)
    tree1 = BlockTree(quantize(np.zeros(DIM)))
    _grow(tree1, 2, m, seed=5)
    orphan = tree1.blocks[tree1.head]                 # its parent is tree1's block 1
    # a fresh node that only has genesis has never seen that parent
    tree2 = BlockTree(quantize(np.zeros(DIM)))
    with pytest.raises(ValidationError):
        tree2.add_block(orphan)
