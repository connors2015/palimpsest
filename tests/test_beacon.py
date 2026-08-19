"""Threshold-BLS beacon (§7.4): unbiasable, unpredictable, verifiable.

Kept to a small committee and few rounds — BLS pairings are ~300ms each."""

from py_ecc.bls.g2_primitives import G2_to_signature

from rig.beacon import (combine, deal, partial_sign, produce, randomness,
                        verify, verify_partial)

KEYS = deal(n=5, t=3, seed=b"test-beacon")


def test_partial_signatures_verify():
    r = 1
    for i in range(1, 6):
        sig = partial_sign(KEYS.shares[i], r)
        assert verify_partial(sig, r, KEYS.partial_pks[i])


def test_group_signature_is_unbiasable_across_subsets():
    """ANY t partials Lagrange-combine to the SAME group signature — no party can
    steer the value by choosing which partials to include."""
    r = 2
    p = {i: partial_sign(KEYS.shares[i], r) for i in range(1, 6)}
    g1 = combine(r, {1: p[1], 2: p[2], 3: p[3]})
    g2 = combine(r, {3: p[3], 4: p[4], 5: p[5]})
    assert G2_to_signature(g1) == G2_to_signature(g2)
    assert verify(g1, r, KEYS.group_pk)


def test_fewer_than_t_cannot_produce_valid_beacon():
    _, ok = produce(KEYS, 3, [1, 2])              # only t-1 = 2 shares
    assert not ok


def test_rounds_give_distinct_randomness():
    g_a, _ = produce(KEYS, 10, [1, 2, 3])
    g_b, _ = produce(KEYS, 11, [1, 2, 3])
    assert randomness(g_a) != randomness(g_b)


def test_tampered_partial_rejected():
    r = 4
    bad = partial_sign(KEYS.shares[1] + 1, r)     # wrong share
    assert not verify_partial(bad, r, KEYS.partial_pks[1])


def test_wrong_round_signature_does_not_verify():
    g, _ = produce(KEYS, 5, [1, 2, 3])
    assert verify(g, 5, KEYS.group_pk)
    assert not verify(g, 6, KEYS.group_pk)        # a value can't be replayed to another round
