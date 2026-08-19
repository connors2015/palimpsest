"""Gossip consensus with no coordinator (§5): flood, converge, train, heal."""

from rig.p2p import Network


def _state_roots(net):
    return set(net.nodes[i].tree.blocks[net.nodes[i].tree.head].header.state_root
               for i in range(len(net.nodes)))


def test_tx_floods_to_every_node():
    net = Network(n_nodes=5, topology="ring", seed=0)
    tx, body = net.nodes[0].make_tx()                # one specific tx at node 0
    net.inject_tx(0, tx, body)
    # pure gossip (no new txs, no blocks): the tx should flood a hop per step
    for _ in range(5):                               # ring of 5 -> ≤3 hops to all
        net.step(proposers=[], produce=False)
    assert all(tx.txid() in net.nodes[i].seen_tx for i in range(5))


def test_nodes_converge_without_coordinator():
    net = Network(n_nodes=5, topology="ring", seed=0)
    for _ in range(20):
        net.step()
    for _ in range(15):
        net.step(proposers=[])                       # quiesce
    assert net.converged()
    assert len(_state_roots(net)) == 1               # one agreed history


def test_model_trains_across_gossip():
    net = Network(n_nodes=5, topology="full", seed=0)
    start = net.accuracy()
    for _ in range(25):
        net.step()
    assert net.accuracy() > 0.7 and net.accuracy() > start


def test_partition_forks_then_heals():
    net = Network(n_nodes=6, topology="full", seed=0)
    for _ in range(12):
        net.step()
    for _ in range(5):
        net.step(proposers=[])
    # split and mine competing forks
    net.set_partition([{0, 1, 2}, {3, 4, 5}])
    for _ in range(10):
        net.step(proposers=[0, 3])
    a = net.nodes[0].tree.head
    b = net.nodes[3].tree.head
    assert a != b                                    # genuinely forked
    # heal and reconcile
    net.set_partition(None)
    for _ in range(8):
        net.step(sync=True)
    assert net.converged() and len(_state_roots(net)) == 1


def test_every_block_is_independently_valid():
    """Fork choice never adopts an invalid block: replaying the winning chain
    reproduces its committed head state on every node."""
    net = Network(n_nodes=4, topology="full", seed=1)
    for _ in range(15):
        net.step()
    for _ in range(10):
        net.step(proposers=[])
    from rig.chain import state_root
    for i in range(4):
        tree = net.nodes[i].tree
        assert state_root(tree.replay_head()) == tree.blocks[tree.head].header.state_root
