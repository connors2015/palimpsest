"""Feldman-VSS DKG (§7.4): verifiable shares, no trusted dealer, drives the beacon."""

import pytest
from py_ecc.bls.g2_primitives import G2_to_signature

from rig.beacon import combine, partial_sign, verify
from rig.dkg import Dealer, run_dkg, verify_share


def test_valid_shares_verify_against_commitments():
    d = Dealer.create(1, t=3, seed=b"s")
    for m in range(1, 6):
        assert verify_share(d.share_for(m), m, d.commitments)


def test_bad_share_is_caught():
    d = Dealer.create(2, t=3, seed=b"s")
    assert not verify_share(d.share_for(3) + 1, 3, d.commitments)   # tampered


def test_dkg_produces_a_working_threshold_key():
    keys = run_dkg(n=5, t=3)
    r = 7
    g1 = combine(r, {i: partial_sign(keys.shares[i], r) for i in (1, 2, 3)})
    g2 = combine(r, {i: partial_sign(keys.shares[i], r) for i in (3, 4, 5)})
    assert verify(g1, r, keys.group_pk)
    assert G2_to_signature(g1) == G2_to_signature(g2)          # unbiasable


def test_fewer_than_t_shares_do_not_reconstruct_the_key():
    keys = run_dkg(n=5, t=3)
    r = 8
    # a 2-of-5 "quorum" interpolates the wrong secret -> fails verification
    g = combine(r, {i: partial_sign(keys.shares[i], r) for i in (1, 2)})
    assert not verify(g, r, keys.group_pk)
