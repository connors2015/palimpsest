"""Merkle proofs + sparse, page-attested MoE inference (§3.1, §8)."""

import numpy as np
import pytest

from rig import merkle
from rig.moe import (MoE, MoEConfig, make_domain_batch, make_rules,
                     verify_serve)


# -- Merkle -------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 8, 13, 16])
def test_merkle_all_proofs_verify(n):
    leaves = [bytes([i]) * 8 for i in range(n)]
    lv, r = merkle.build(leaves), merkle.root([bytes([i]) * 8 for i in range(n)])
    for i in range(n):
        assert merkle.verify(leaves[i], i, merkle.proof(lv, i), r)


def test_merkle_rejects_tampered_leaf():
    leaves = [bytes([i]) * 8 for i in range(6)]
    lv = merkle.build(leaves)
    r = lv[-1][0]
    assert not merkle.verify(b"x" * 8, 3, merkle.proof(lv, 3), r)


# -- MoE ----------------------------------------------------------------------
def _trained(seed=0, E=8):
    rng = np.random.default_rng(seed)
    cfg = MoEConfig(n_experts=E, top_k=1)
    rules = make_rules(rng, cfg)
    moe = MoE(cfg)
    vec = moe.init(rng)
    for _ in range(500):
        vec = moe.train_step(vec, make_domain_batch(rng, 128, cfg, rules), lr=0.3)
    test = make_domain_batch(np.random.default_rng(seed + 99), 400, cfg, rules)
    return moe, vec, test


def test_sparse_serving_matches_dense_accuracy():
    moe, vec, test = _trained()
    dense = moe.accuracy_dense(vec, test)
    sparse = moe.accuracy_sparse(vec, test)
    assert dense > 0.75
    assert abs(dense - sparse) < 0.03      # top-1 keeps quality


def test_serve_loads_only_k_experts():
    moe, vec, test = _trained(E=8)
    r = moe.serve(vec, int(test[0][0]), test[1][0])
    assert len(r["experts"]) == moe.cfg.top_k
    assert r["expert_fraction"] == moe.cfg.top_k / moe.cfg.n_experts
    # loaded pages = router + exactly the used experts, nothing else
    assert set(r["loaded_pages"]) == {0} | {1 + e for e in r["experts"]}


def test_attestation_verifies_from_only_loaded_pages():
    moe, vec, test = _trained()
    d0, x0 = int(test[0][0]), test[1][0]
    r = moe.serve(vec, d0, x0)
    root = moe.merkle_root(vec)
    # verifier holds ONLY the pages the receipt names
    pages = {i: moe.pages(vec)[i] for i in r["loaded_pages"]}
    assert set(pages) != set(range(moe.cfg.n_experts + 1))  # not the whole model
    assert verify_serve(moe, pages, r, root, d0, x0)


def test_attestation_rejects_tampered_page():
    moe, vec, test = _trained()
    d0, x0 = int(test[0][0]), test[1][0]
    r = moe.serve(vec, d0, x0)
    root = moe.merkle_root(vec)
    pages = {i: moe.pages(vec)[i] for i in r["loaded_pages"]}
    bad = r["loaded_pages"][-1]
    pages[bad] = pages[bad] + 1.0          # a fake/wrong expert page
    assert not verify_serve(moe, pages, r, root, d0, x0)


def test_attestation_rejects_wrong_committed_root():
    moe, vec, test = _trained()
    d0, x0 = int(test[0][0]), test[1][0]
    r = moe.serve(vec, d0, x0)
    pages = {i: moe.pages(vec)[i] for i in r["loaded_pages"]}
    wrong_root = moe.merkle_root(vec + 1.0)
    assert not verify_serve(moe, pages, r, wrong_root, d0, x0)


def test_merkle_root_is_deterministic():
    moe, vec, _ = _trained()
    assert moe.merkle_root(vec) == moe.merkle_root(vec.copy())
