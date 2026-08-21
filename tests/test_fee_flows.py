"""rev 6 — the inference-fee economy (§9.3 adapted).

Every inference fee splits 60/20/20: the server is paid instantly (absorbing
division dust, so the split is supply-exact); the data and training slices
accumulate in on-chain pools, drained by the next block's reward to its
provenance-named data owners and delta miners. This is the mechanism that funds
"train forever" from usage once emission tapers to the tail.
"""

from rig.crypto import Key
from rig.token import (
    FEE_SHARE_DATA, FEE_SHARE_TRAIN, SHARE_DATA, SHARE_MINERS,
    InferenceReceiptTx, TokenLedger, address, emission,
)


def _key(tag):
    return Key.generate(tag.encode().ljust(32, b"0"))


def _register(led, owner_key, data_hash, weight=1):
    led.registry[data_hash] = {
        "owner": address(owner_key.pub), "data_hash": data_hash, "size": 0,
        "media_type": "text", "stake": 0, "weight": weight, "status": "active"}


def test_fee_splits_server_and_pools_supply_exact():
    payer, server = _key("payer"), _key("server")
    led = TokenLedger()
    led.apply_reward(1, [payer.pub], "genesis", [])    # fund the payer by mining
    supply_before = led.supply()
    fee = 10_000
    tx = InferenceReceiptTx(payer_pub=payer.pub, server_addr=address(server.pub),
                            fee=fee, output_hash="ab" * 32, head_root="cd" * 32,
                            nonce=led.nonces.get(address(payer.pub), 0)).signed(payer)
    assert led.apply_data_tx(tx, 2, set())
    data_cut = fee * FEE_SHARE_DATA // 10_000
    train_cut = fee * FEE_SHARE_TRAIN // 10_000
    assert led.balance(address(server.pub)) == fee - data_cut - train_cut
    assert led.fee_data_pool == data_cut and led.fee_train_pool == train_cut
    assert led.supply() == supply_before               # a fee moves, never mints


def test_pools_drain_to_next_blocks_miners_and_named_data():
    miner, owner = _key("miner"), _key("owner")
    led = TokenLedger()
    _register(led, owner, "C")
    led.fee_train_pool = 500                           # accumulated fee slices
    led.fee_data_pool = 300
    led.apply_reward(2, [miner.pub], "genesis", data_credits={"C": 1})
    e = emission(2)
    assert led.balance(address(miner.pub)) == e * SHARE_MINERS // 10_000 + 500
    assert led.balance(address(owner.pub)) == e * SHARE_DATA // 10_000 + 300
    assert led.fee_train_pool == 0 and led.fee_data_pool == 0


def test_pools_carry_forward_without_recipients():
    led = TokenLedger()
    led.fee_train_pool = 500
    led.fee_data_pool = 300
    # a block with no miners and no named data must NOT burn the pools
    led.apply_reward(2, [], "genesis", data_credits={})
    assert led.fee_train_pool == 500 and led.fee_data_pool == 300
    # ...and the pools count toward supply while in flight
    assert led.supply() == 800
