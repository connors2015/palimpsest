"""Ledger-on-chain integration (client/gossip.py + rig/token.py): an installed
block deterministically mints rewards to the miner, proposer, and the genesis
data contributor — no network needed (drives RealCore directly, toy model)."""

import client.gossip as g
from rig.token import address, emission
from rig.crypto import Key


def test_block_install_mints_rewards(tmp_path):
    founder = address(Key.generate(b"founder-wallet-test-seed-000000!").pub)
    g.DATA_CONTRIBUTOR = founder
    try:
        core = g.RealCore(0)
        assert core.head_ledger().supply() == 0            # fair launch
        hh, delta, _ = core.train_delta()
        outbox = []
        core.submit_delta(hh, delta, outbox)
        core.propose(outbox)
        led = core.head_ledger()
        h = core.tree.blocks[core.tree.head].header.height
        assert h == 1
        assert led.supply() > 0 and led.supply() <= emission(1)
        miner_addr = address(core.key.pub)
        assert led.balance(miner_addr) > 0                 # trained + proposed
        assert led.balance(founder) > 0                    # data share -> founder
        # ledger is replayable: recompute from genesis matches
        core.ledgers = {core.tree.genesis.hash: type(led)()}
        assert core.head_ledger().root() == led.root()
    finally:
        g.DATA_CONTRIBUTOR = None
