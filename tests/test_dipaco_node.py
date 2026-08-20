"""Live sharded DiPaCo node (client/dipaco_node.py) — the consensus + custody
logic, exercised WITHOUT the network: two nodes hold disjoint paths, exchange
one round's updates by hand, and must agree on the page-DAG model root while
neither holds the other's module content. Then reconstruction from the swarm.

The cross-machine (M3 + 2080 Ti) run is the live proof; this locks the logic in CI."""

import numpy as np

import client.dipaco_node as dn
from client.dipaco_node import DiPaCoNode, manifest_root
from client.cas import cid
from rig.chain import dequantize
from client.trainer import set_flat_params

dn.DEVICE_OVERRIDE = "cpu"                        # deterministic, no GPU in CI


def _node(node_id, paths):
    # no peers, never call run() — we drive rounds by hand
    return DiPaCoNode(node_id, "0.0.0.0", 0, [], 2, paths)


def _update_from(node, bb_delta, new_mods):
    """The wire form a node broadcasts: its backbone delta + its module CIDs."""
    return {
        "bb_delta": bb_delta,
        "mod_cids": {("mod", l, m): cid(c.tobytes()) for (l, m), c in new_mods.items()},
    }


def test_disjoint_paths_partition_all_modules():
    a, b = _node(0, [0, 1]), _node(1, [2, 3])
    # every (level, module) is owned by exactly one of the two nodes
    assert not (a.owned_mods & b.owned_mods)
    assert a.owned_mods | b.owned_mods == set(a.mod_span)


def test_two_nodes_agree_on_root_and_never_hold_peer_content():
    a, b = _node(0, [0, 1]), _node(1, [2, 3])
    assert manifest_root(a.cids) == manifest_root(b.cids)      # identical genesis

    for r in range(3):
        da, ma = a.train_round(r)
        db, mb = b.train_round(r)
        a.inbox[r] = {1: _update_from(b, db, mb)}              # A hears B
        b.inbox[r] = {0: _update_from(a, da, ma)}              # B hears A
        ra = a.apply_round(r, da, ma)
        rb = b.apply_round(r, db, mb)
        assert ra == rb                                        # sharded consensus each round

    # neither node holds the OTHER's module content — only CIDs
    a_peer_content = [k for k in a.mod_span if k not in a.owned_mods
                      and a.store.has(a.cids[("mod", *k)])]
    assert a_peer_content == []
    # …yet both agree on the whole-model root
    assert manifest_root(a.cids) == manifest_root(b.cids)


def test_reconstruct_full_model_from_peer_pages():
    """A node assembles the WHOLE model once it fetches peers' module contents by
    CID (the Bitswap step), and the result is self-consistent with the agreed
    root — proving a sharded node can still serve the full model."""
    a, b = _node(0, [0, 1]), _node(1, [2, 3])
    da, ma = a.train_round(0)
    db, mb = b.train_round(0)
    a.inbox[0] = {1: _update_from(b, db, mb)}
    b.inbox[0] = {0: _update_from(a, da, ma)}
    a.apply_round(0, da, ma)
    b.apply_round(0, db, mb)

    # A cannot reconstruct yet — it lacks B's module CONTENT
    full, missing = a.reconstruct_full()
    assert full is None and len(missing) == len(b.owned_mods)

    # simulate Bitswap: B serves its module blocks by CID; A stores them
    for k in b.owned_mods:
        blob = b.store.get(b.cids[("mod", *k)])
        assert blob is not None
        got, ok = a.bitswap.on_block(cid(blob), blob)
        assert ok
    full, missing = a.reconstruct_full()
    assert full is not None and not missing                   # whole model assembled

    # the assembled model runs on ALL paths (A's own AND B's) with low loss
    import torch
    from client.dipaco import make_path, coarse_route
    set_flat_params(a.model, dequantize(full))
    for pid in range(dn.N_PATHS):
        path = make_path(pid, dn.CFG.n_layer, dn.CFG.n_modules)
        buf = dn._domain_buf(coarse_route(pid, dn.N_PATHS))
        gen = torch.Generator().manual_seed(1)
        x, y = dn._get_batch(buf, 32, dn.CFG.block_size, gen, "cpu")
        with torch.no_grad():
            _, loss = a.model(x, y, path=path)
        assert loss.item() < 1.0                               # trained model, all paths
