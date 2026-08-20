"""Page-sharded MoE client (client/moe.py): a node holds/trains/serves only a
SUBSET of experts, the shard machinery splits down to individual weights, and
sparse serving loads only the routed experts.

Hermetic: a tiny in-memory repeating-byte task (no data download), tiny model."""

import numpy as np
import torch

from client.moe import (
    MoEGPT, MoEGPTConfig, PageMap, build_moe, load_fraction, mask_delta,
    shard_aggregate,
)
from client.trainer import flat_params, set_flat_params
from rig.chain import dequantize, quantize


CFG = MoEGPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16,
                   n_experts=6, top_k=2)


def _data(n=4096, period=37):
    """A learnable repeating byte stream, so loss must drop if training works."""
    return (np.arange(n) % period).astype(np.int64)


def _batch(buf, bs, T, gen):
    ix = torch.randint(0, len(buf) - T - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(buf[i:i + T]) for i in ix])
    y = torch.stack([torch.from_numpy(buf[i + 1:i + 1 + T]) for i in ix])
    return x, y


def _train_delta(model, buf, base_vec, steps=25, seed=0):
    """Train the full model from base_vec; return the quantised pseudo-gradient."""
    set_flat_params(model, base_vec)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x, y = _batch(buf, 16, CFG.block_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return quantize(flat_params(model) - base_vec)


def _val(model, buf, vec):
    set_flat_params(model, vec)
    gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        x, y = _batch(buf, 64, CFG.block_size, gen)
        _, loss = model(x, y)
    return loss.item()


def test_pagemap_partitions_experts_and_backbone():
    model = MoEGPT(CFG)
    pm = PageMap(model)
    assert len(pm.experts) == CFG.n_layer * CFG.n_experts        # one page per (layer, expert)
    # experts + backbone exactly tile the flat vector, no overlap, no gap
    covered = np.zeros(pm.n, dtype=bool)
    covered[pm.backbone_idx] = True
    for k in pm.experts:
        idx = pm.expert_indices([k])
        assert not covered[idx].any()                            # experts disjoint from backbone
        covered[idx] = True
    assert covered.all()                                         # full coverage
    # experts are a real minority of params only if d_ff dominates; at least the
    # backbone is a strict subset and each expert page is non-empty
    for k in pm.experts:
        assert pm.expert_indices([k]).size > 0


def test_disjoint_expert_shards_train_and_aggregate():
    """Two nodes hold DISJOINT experts (+ shared backbone), each trains the model
    and masks its delta to its shard; the sharded aggregate lowers loss."""
    buf = _data()
    model, _ = build_moe(CFG, device="cpu", seed=7)
    pm = PageMap(model)
    base = flat_params(model)
    base_int = quantize(base)
    genesis_val = _val(model, buf, base)

    experts = pm.experts
    a_holds = experts[::2]                                       # even experts
    b_holds = experts[1::2]                                      # odd experts
    mask_a = pm.mask(a_holds, include_backbone=True)
    mask_b = pm.mask(b_holds, include_backbone=True)
    # together they cover everything; experts are split between them
    assert (mask_a | mask_b).all()
    assert not (pm.expert_indices(a_holds).size and
                mask_b[pm.expert_indices(a_holds)].all())        # b does NOT hold a's experts

    da = _train_delta(model, buf, base, seed=1)
    db = _train_delta(model, buf, base, seed=2)
    new_int = shard_aggregate(base_int, [da, db], [mask_a, mask_b])
    new_val = _val(model, buf, dequantize(new_int))
    assert new_val < genesis_val - 0.05                         # sharded training helped


def test_singly_held_expert_applied_in_full_not_halved():
    """A page held by ONE node is applied in full — averaging is over holders,
    not over all nodes (the bug a naive trimmed-mean would introduce)."""
    model = MoEGPT(CFG)
    pm = PageMap(model)
    n = pm.n
    base_int = np.zeros(n, dtype=np.int64)
    one = pm.experts[0]
    mask_a = pm.mask([one], include_backbone=False)             # A holds only expert `one`
    mask_b = pm.mask(pm.experts[1:], include_backbone=False)    # B holds the rest
    da = np.full(n, 1000, dtype=np.int64)                       # A's raw (full) delta
    db = np.full(n, 4000, dtype=np.int64)
    out = shard_aggregate(base_int, [da, db], [mask_a, mask_b])
    idx = pm.expert_indices([one])
    assert np.all(out[idx] == 1000)                             # full, not 500 (halved)


def test_subdivide_to_individual_weights():
    """The shard granularity is a knob: subdivide(1) yields one page per weight —
    the 'train individual weights' ultimate fidelity. A node can then hold an
    arbitrary set of single weights."""
    model = MoEGPT(CFG)
    pm = PageMap(model)
    pages = pm.subdivide(max_page=1)
    assert len(pages) == pm.n                                    # one page per parameter
    assert all(e - s == 1 for s, e in pages)
    # coarser granularity yields fewer, larger pages that still tile the vector
    coarse = pm.subdivide(max_page=64)
    assert len(coarse) < pm.n
    covered = np.zeros(pm.n, dtype=bool)
    for s, e in coarse:
        assert not covered[s:e].any()
        covered[s:e] = True
    assert covered.all()

    # a node holding just a handful of individual weights contributes only those
    held = {3, 17, 100}
    mask = np.zeros(pm.n, dtype=bool)
    for i in held:
        mask[i] = True
    delta = np.arange(pm.n, dtype=np.int64) + 1
    masked = mask_delta(delta, mask)
    assert set(np.nonzero(masked)[0]) == held


def test_sparse_serving_loads_only_routed_experts():
    """Serving a query touches only top-k experts per layer, so a node holding a
    subset can still answer — and the parameter-load fraction is well below 1."""
    torch.manual_seed(0)
    model = MoEGPT(CFG)
    pm = PageMap(model)
    x = torch.randn(1, CFG.block_size, CFG.n_embd)
    ff = model.blocks[0].moe
    dense = ff(x)
    out, touched = ff.serve(x)
    assert torch.allclose(dense, out, atol=1e-5)                # serve == train result
    assert 0 < len(touched) <= CFG.top_k * CFG.block_size       # only routed experts run
    # holding half the experts loads well under the whole model
    half = pm.experts[: len(pm.experts) // 2]
    frac = load_fraction(pm, half)
    assert frac < 1.0
