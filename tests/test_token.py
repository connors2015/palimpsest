"""Token ledger (rig/token.py): fair-launch supply, deterministic emissions,
reward splits, signed transfers with nonces, and canonical roots."""

from rig.crypto import Key
from rig.token import (
    BASE_REWARD, GRAIN, HALVING_BLOCKS, TAIL_EPOCH, TAIL_REWARD, TokenLedger,
    TransferTx, address, emission,
)


def _key(tag):
    return Key.generate(tag.encode().ljust(32, b"0"))


def test_fair_launch_genesis_is_empty():
    led = TokenLedger()
    assert led.supply() == 0                       # no premine, ever


def test_emission_halves_then_tails():
    assert emission(0) == 0                        # genesis mints nothing
    assert emission(1) == BASE_REWARD
    assert emission(HALVING_BLOCKS) == BASE_REWARD
    assert emission(HALVING_BLOCKS + 1) == BASE_REWARD // 2
    assert emission(2 * HALVING_BLOCKS + 1) == BASE_REWARD // 4
    # rev 6 tail emission: the reward floors at TAIL_REWARD and NEVER reaches
    # zero — the perpetual training wage ("train forever" is funded by design)
    assert emission(TAIL_EPOCH * HALVING_BLOCKS + 1) == TAIL_REWARD
    assert emission(100 * HALVING_BLOCKS) == TAIL_REWARD
    assert emission(10**12) == TAIL_REWARD > 0     # no height ever emits zero


def test_reward_split_deterministic_and_supply_bounded():
    a, b, prop = _key("minerA"), _key("minerB"), _key("proposer")
    founder = address(_key("founder").pub)
    l1, l2 = TokenLedger(), TokenLedger()
    for led in (l1, l2):
        led.apply_reward(1, [a.pub, b.pub], prop.pub, [founder])
    assert l1.root() == l2.root()                  # identical on every node
    assert l1.supply() <= emission(1)              # dust burned, never minted extra
    assert l1.balance(address(a.pub)) == l1.balance(address(b.pub)) > 0
    assert l1.balance(address(prop.pub)) > 0
    assert l1.balance(founder) > 0                 # data share -> founding wallet
    # miner order must not matter (canonical sort inside)
    l3 = TokenLedger(); l3.apply_reward(1, [b.pub, a.pub], prop.pub, [founder])
    assert l3.root() == l1.root()


def test_transfer_lifecycle_and_replay_protection():
    a, b = _key("alice"), _key("bob")
    led = TokenLedger()
    led.apply_reward(1, [a.pub], a.pub, [])        # fund alice via mining
    bal = led.balance(address(a.pub))
    tx = TransferTx(from_pub=a.pub, to_addr=address(b.pub),
                    amount=bal // 2, nonce=0).signed(a)
    assert led.apply_transfer(tx)
    assert led.balance(address(b.pub)) == bal // 2
    assert not led.apply_transfer(tx)              # replay: nonce consumed
    # wrong signer fails
    forged = TransferTx(from_pub=a.pub, to_addr=address(b.pub),
                        amount=1, nonce=1).signed(b) if False else \
        TransferTx(from_pub=a.pub, to_addr=address(b.pub), amount=1, nonce=1)
    forged.sig = b.sign(forged.signing_bytes())    # signed by the WRONG key
    assert not led.apply_transfer(forged)
    # overdraft fails
    over = TransferTx(from_pub=a.pub, to_addr=address(b.pub),
                      amount=10**18, nonce=1).signed(a)
    assert not led.apply_transfer(over)


def test_ledger_root_covers_nonces():
    a, b = _key("alice"), _key("bob")
    l1, l2 = TokenLedger(), TokenLedger()
    for led in (l1, l2):
        led.apply_reward(1, [a.pub], a.pub, [])
    tx = TransferTx(from_pub=a.pub, to_addr=address(b.pub),
                    amount=1 * GRAIN, nonce=0).signed(a)
    l1.apply_transfer(tx)
    assert l1.root() != l2.root()                  # state change -> new root
