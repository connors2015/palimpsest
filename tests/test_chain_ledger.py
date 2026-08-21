"""Transfer-lane protocol (rev 2): rewards mint per block, transfers settle
through blocks (ledger_root commits them), invalid transfers can't enter, and a
tampered ledger_root is rejected. Drives RealCore directly (toy model)."""

import numpy as np
import pytest

import client.gossip as g
from rig.blockchain import ValidationError
from rig.crypto import Key
from rig.token import GRAIN, TransferTx, address, emission


@pytest.fixture()
def core():
    founder = address(Key.generate(b"founder-wallet-test-seed-000000!").pub)
    g.DATA_CONTRIBUTOR = founder
    c = g.RealCore(0)
    c.founder = founder
    yield c
    g.DATA_CONTRIBUTOR = None


def _mine(core):
    hh, delta, _ = core.train_delta()
    outbox = []
    core.submit_delta(hh, delta, outbox)
    core.propose(outbox)
    return outbox


def test_block_install_mints_rewards(core):
    assert core.head_ledger().supply() == 0            # fair launch
    _mine(core)
    led = core.head_ledger()
    assert core.tree.blocks[core.tree.head].header.height == 1
    assert 0 < led.supply() <= emission(1)
    assert led.balance(address(core.key.pub)) > 0      # trained + proposed
    assert led.balance(core.founder) > 0               # data share -> founder


def test_transfer_settles_through_a_block(core):
    _mine(core)                                        # block 1 funds the miner
    miner_addr = address(core.key.pub)
    bal = core.head_ledger().balance(miner_addr)
    bob = address(Key.generate(b"bob-wallet-test-seed-00000000000").pub)
    tx = TransferTx(from_pub=core.key.pub, to_addr=bob,
                    amount=bal // 3, nonce=0).signed(core.key)
    outbox = []
    core.recv_transfer(tx, outbox)
    assert tx.txid() in core.transfer_pool
    _mine(core)                                        # block 2 includes + settles it
    head = core.tree.blocks[core.tree.head]
    assert head.header.height == 2
    assert any(t.txid() == tx.txid() for t in head.transfers)
    assert head.header.ledger_root == core.head_ledger().root()
    assert core.head_ledger().balance(bob) == bal // 3
    assert tx.txid() not in core.transfer_pool         # settled, out of pool


def test_invalid_transfer_never_included(core):
    _mine(core)
    rich = core.head_ledger().balance(address(core.key.pub))
    bob = address(Key.generate(b"bob-wallet-test-seed-00000000000").pub)
    over = TransferTx(from_pub=core.key.pub, to_addr=bob,
                      amount=rich * 100, nonce=0).signed(core.key)   # overdraft
    outbox = []
    core.recv_transfer(over, outbox)
    _mine(core)
    head = core.tree.blocks[core.tree.head]
    assert head.transfers == []                        # dry-run filtered it out
    assert core.head_ledger().balance(bob) == 0


def test_tampered_ledger_root_rejected(core):
    _mine(core)
    # build a valid next block, then tamper its ledger_root
    hh, delta, _ = core.train_delta()
    outbox = []
    core.submit_delta(hh, delta, outbox)
    from rig.blockchain import build_block
    tx = list(core.mempool.values())[0]
    bodies = {tx.da_pointer: core._body(tx.txid())}
    blk = build_block(core.tree, core.tree.head, [tx], bodies,
                      {tx.txid(): 1.0}, core.key)
    blk.header.ledger_root = "00" * 32                 # lie about the token state
    with pytest.raises(ValidationError, match="ledger_root"):
        core.tree.add_block(blk)


def test_ledger_replay_matches(core):
    _mine(core)
    bob = address(Key.generate(b"bob-wallet-test-seed-00000000000").pub)
    tx = TransferTx(from_pub=core.key.pub, to_addr=bob,
                    amount=1 * GRAIN, nonce=0).signed(core.key)
    outbox = []
    core.recv_transfer(tx, outbox)
    _mine(core)
    # replay every block's ledger transition from genesis: identical root
    from rig.blockchain import apply_ledger
    from rig.token import TokenLedger
    led = TokenLedger()
    led.seed_genesis_data(g.DATA_CONTRIBUTOR)          # genesis registry entry
    for b in core.tree.chain_from_genesis():
        led = apply_ledger(led, b, g.DATA_CONTRIBUTOR,
                           core.tree.recent_proposers(b.header.prev_hash))
    assert led.root() == core.head_ledger().root()
