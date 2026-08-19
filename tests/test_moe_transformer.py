"""Fused MoE transformer: grad-check, learning, chain convergence, attested serve."""

import numpy as np
import pytest

from rig.chain import dequantize, state_root
from rig.moe_transformer import (MoETConfig, MoETransformer, verify_serve)
from rig.node import run_in_memory

SMALL = MoETConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64,
                   n_experts=4, top_k=2, vocab=8, context=12, task="modadd")


def test_gradient_matches_numerical():
    m = MoETransformer(SMALL)
    rng = np.random.default_rng(0)
    vec = m.init(rng)
    batch = m.sample_batch(rng, 4)
    g = m._grad(vec, batch)
    eps = 1e-5
    for i in rng.choice(m.param_count, size=40, replace=False):
        vp = vec.copy(); vp[i] += eps
        vm = vec.copy(); vm[i] -= eps
        num = (m.loss(vp, batch) - m.loss(vm, batch)) / (2 * eps)
        rel = abs(num - g[i]) / (abs(num) + abs(g[i]) + 1e-9)
        assert rel < 1e-4, f"grad mismatch at {i}: {rel:.2e}"


def test_fused_model_learns_standalone():
    m = MoETransformer(SMALL)
    rng = np.random.default_rng(1)
    vec = m.init(rng)
    test = m.sample_batch(np.random.default_rng(77), 200)
    start = m.accuracy(vec, test)
    for _ in range(200):
        vec = m.train_step(vec, m.sample_batch(rng, 64), lr=0.5, steps=1)
    assert start < 0.4 and m.accuracy(vec, test) > 0.9


def test_fused_model_converges_through_chain():
    m = MoETransformer(SMALL)
    chain, log = run_in_memory(blocks=55, seed=7, model=m)
    # The point: the fused MoE transformer trains through DiLoCo aggregation
    # and the chain replays bit-exact — not a precise accuracy target.
    assert log.acc[-1] > log.acc[0] + 0.4
    assert state_root(chain.replay()) == chain.blocks[-1].root


def test_decode_step_is_sparse():
    m = MoETransformer(SMALL)
    vec = m.init(np.random.default_rng(2))
    tokens = m.sample_batch(np.random.default_rng(3), 1)[0]
    r = m.serve(vec, tokens)
    # one generated token routes to at most top_k experts per layer
    assert len(r["decode_experts"]) <= SMALL.top_k * SMALL.n_layers
    # the receipt loads the sequence union (needed to recompute the full output)
    assert r["loaded_pages"][0] == 0
    assert len(r["loaded_pages"]) == 1 + len(r["used_experts"])
    assert len(r["decode_experts"]) <= len(r["used_experts"])


def test_partial_recompute_verifier_honest():
    m = MoETransformer(SMALL)
    vec = m.init(np.random.default_rng(6))
    tokens = m.sample_batch(np.random.default_rng(7), 1)[0]
    r = m.serve(vec, tokens)
    root = m.merkle_root(vec)
    pages = {i: m.pages(vec)[i] for i in r["loaded_pages"]}
    # verifier holds ONLY the loaded pages (never the whole model)
    assert len(pages) < len(m.pages(vec)) or m.cfg.n_experts * m.cfg.n_layers <= len(r["used_experts"])
    assert verify_serve(m, pages, r, root, tokens)


def test_partial_recompute_rejects_tamper_underload_forgery_wrongroot():
    m = MoETransformer(SMALL)
    vec = m.init(np.random.default_rng(6))
    tokens = m.sample_batch(np.random.default_rng(7), 1)[0]
    r = m.serve(vec, tokens)
    root = m.merkle_root(vec)
    pages = {i: m.pages(vec)[i] for i in r["loaded_pages"]}

    tampered = dict(pages)
    tampered[r["loaded_pages"][-1]] = tampered[r["loaded_pages"][-1]] + 1.0
    assert not verify_serve(m, tampered, r, root, tokens)          # tampered page

    assert not verify_serve(m, pages, r, m.merkle_root(vec + 1.0), tokens)  # wrong root

    forged = dict(r, output=(r["output"] + 1) % m.cfg.vocab)
    assert not verify_serve(m, pages, forged, root, tokens)        # forged output

    if len(r["loaded_pages"]) > 2:                                 # under-load
        under = {i: pages[i] for i in r["loaded_pages"][:-1]}
        r_under = dict(r, loaded_pages=r["loaded_pages"][:-1])
        assert not verify_serve(m, under, r_under, root, tokens)


def test_fused_model_trains_through_async():
    from rig.async_node import run_async_sim
    m = MoETransformer(SMALL)
    chain, log = run_async_sim(ticks=200, seed=7, model=m)
    assert log.acc[-1] > log.acc[0] + 0.4        # trains under asynchrony
    assert state_root(chain.replay()) == chain.blocks[-1].root
    assert log.stale_dropped > 0                  # genuinely async, not barriered


def test_merkle_root_deterministic():
    m = MoETransformer(SMALL)
    vec = m.init(np.random.default_rng(8))
    assert m.merkle_root(vec) == m.merkle_root(vec.copy())
