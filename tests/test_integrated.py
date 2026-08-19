"""Integrated node: beacon + DA + leader election in one loop (§3–9)."""

from py_ecc.bls.g2_primitives import G2_to_signature

from rig import beacon as bcn
from rig.integrated import new_chain


def test_integrated_chain_trains_with_real_primitives():
    chain = new_chain(n=5, t=3, seed=0)
    for _ in range(16):
        chain.round()
    assert chain.accuracy() > 0.8
    assert chain.replay()                          # hash-chain links intact


def test_leader_is_beacon_elected_and_varies():
    chain = new_chain(n=5, t=3, seed=1)
    leaders = [chain.round().leader for _ in range(10)]
    assert all(0 <= l < 5 for l in leaders)
    assert len(set(leaders)) > 1                    # not a fixed proposer


def test_withheld_da_delta_is_excluded():
    chain = new_chain(n=5, t=3, seed=0)
    for _ in range(4):
        chain.round()
    chain.withholders.add(2)                        # miner 2 disperses no shards
    included = sum(2 in chain.round().miner_ids for _ in range(6))
    assert included == 0                            # DA sampling rejects it every time


def test_beacon_unbiasable_across_quorums():
    chain = new_chain(n=5, t=3, seed=0)
    r = 321
    g1 = bcn.combine(r, {i: bcn.partial_sign(chain.keys.shares[i], r) for i in (1, 2, 3)})
    g2 = bcn.combine(r, {i: bcn.partial_sign(chain.keys.shares[i], r) for i in (3, 4, 5)})
    assert G2_to_signature(g1) == G2_to_signature(g2)
