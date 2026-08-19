"""Real async socket gossip — coordinator-free consensus over actual sockets.

`rig/p2p.py` proved the consensus logic in a deterministic in-process simulator.
This runs the same logic over real asyncio TCP connections, so nodes gossip
across machines with no coordinator. Each node listens for peers, dials the
seed peers it was given, floods signed txs and blocks (length-prefixed pickle),
keeps a mempool + BlockTree, and follows the heaviest valid chain. Leadership
rotates by wall-clock round (leader = round mod N), so no single node dictates
history — the right to propose simply takes turns.

On a new connection a node re-announces its whole chain (the sync-on-connect
fix), so a joining or reconnecting peer catches up and forks reconcile.

Run one process per node (across machines):
  python3 -m rig.gossip_net --id 0 --port 9500 \
      --peers 100.x:9500,100.y:9501 --n 3 --seconds 20
"""

import argparse
import asyncio
import pickle
import struct
import time

import numpy as np

from .blockchain import BlockTree, ValidationError, build_block
from .chain import quantize, dequantize, state_root
from .crypto import Key
from .p2p import (EVAL_BATCH, INCLUDE_K, MODEL, GossipNode)


async def _send(writer, obj):
    data = pickle.dumps(obj)
    writer.write(struct.pack(">I", len(data)) + data)
    await writer.drain()


async def _recv(reader):
    hdr = await reader.readexactly(4)
    (n,) = struct.unpack(">I", hdr)
    return pickle.loads(await reader.readexactly(n))


class AsyncGossipNode:
    def __init__(self, node_id, host, port, peers, n_total, seed=0,
                 interval=0.4, t0=None):
        w0 = quantize(MODEL.init(np.random.default_rng(seed)))
        key = Key.generate(f"node{node_id}".encode().ljust(32, b"0"))
        self.core = GossipNode(node_id, key, BlockTree(w0))
        self.host, self.port, self.peers = host, port, peers
        self.n_total, self.interval = n_total, interval
        self.t0 = t0 or time.time()
        self.writers = set()
        self._stop = asyncio.Event()

    # -- connection handling ----------------------------------------------
    async def _serve(self, reader, writer):
        await self._peer_loop(reader, writer, inbound=True)

    async def _dial(self, host, port):
        for _ in range(40):                          # retry until the peer is up
            try:
                reader, writer = await asyncio.open_connection(host, port)
                await self._peer_loop(reader, writer, inbound=False)
                return
            except (ConnectionError, OSError):
                await asyncio.sleep(0.25)

    async def _peer_loop(self, reader, writer, inbound):
        self.writers.add(writer)
        try:
            await _send(writer, ("hello", self.core.node_id))
            # sync: announce our whole chain so the peer catches up
            for b in self.core.tree.chain_from_genesis():
                await _send(writer, ("block", b))
            while not self._stop.is_set():
                msg = await _recv(reader)
                self._handle(msg)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.writers.discard(writer)
            writer.close()

    def _handle(self, msg):
        outbox = []
        kind = msg[0]
        if kind == "tx":
            self.core.recv_tx(msg[1], msg[2], outbox)
        elif kind == "block":
            self.core.recv_block(msg[1], outbox)
        # forward anything new to all peers
        for m in outbox:
            self._broadcast(m)

    def _broadcast(self, msg):
        for w in list(self.writers):
            asyncio.create_task(self._safe_send(w, msg))

    async def _safe_send(self, writer, msg):
        try:
            await _send(writer, msg)
        except (ConnectionError, OSError):
            self.writers.discard(writer)

    # -- the block loop ----------------------------------------------------
    async def _loop(self, seconds, settle=2.0):
        end = time.time() + seconds
        while time.time() < end:
            await asyncio.sleep(self.interval)
            outbox = []
            self.core.submit_own_tx(outbox)                 # everyone produces + gossips
            rnd = int((time.time() - self.t0) / self.interval)
            if rnd % self.n_total == self.core.node_id:     # rotating leader
                self.core.propose(outbox)
            for m in outbox:
                self._broadcast(m)
        # quiescent settle: stop producing, keep relaying so in-flight blocks
        # propagate and every node lands on the same head.
        await asyncio.sleep(settle)
        self._stop.set()

    async def run(self, seconds):
        server = await asyncio.start_server(self._serve, self.host, self.port)
        async with server:
            await asyncio.sleep(0.3)                         # let peers bind
            dials = [asyncio.create_task(self._dial(h, p)) for h, p in self.peers]
            await self._loop(seconds)
            for d in dials:
                d.cancel()
        return self.core.tree

    def head_height(self):
        t = self.core.tree
        return t.blocks[t.head].header.height

    def head_root(self):
        t = self.core.tree
        return t.blocks[t.head].header.state_root


# --------------------------------------------------------------------------
# In-process harness (loopback) — real sockets, one event loop, for tests/demo
# --------------------------------------------------------------------------
async def _run_cluster(n=3, seconds=8, base_port=9600, seed=0, interval=0.4):
    t0 = time.time() + 0.5
    ports = [base_port + i for i in range(n)]
    nodes = []
    for i in range(n):
        peers = [("127.0.0.1", ports[j]) for j in range(n) if j != i]
        nodes.append(AsyncGossipNode(i, "127.0.0.1", ports[i], peers, n,
                                     seed=seed, interval=interval, t0=t0))
    trees = await asyncio.gather(*[node.run(seconds) for node in nodes])
    return nodes, trees


def run_cluster(n=3, seconds=8, base_port=9600, seed=0, interval=0.4):
    return asyncio.run(_run_cluster(n, seconds, base_port, seed, interval))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--peers", default="")          # host:port,host:port
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--interval", type=float, default=0.4)
    ap.add_argument("--t0", type=float, default=0.0)   # shared start (unix time)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or a.id is None:
        print("async gossip over real sockets (loopback), 3 nodes, no coordinator…")
        nodes, trees = run_cluster(n=3, seconds=8)
        heights = [n.head_height() for n in nodes]
        roots = set(n.head_root() for n in nodes)
        acc = MODEL.accuracy(dequantize(nodes[0].core.tree.head_state()),
                             MODEL.sample_batch(np.random.default_rng(123456), 200))
        print(f"heights {heights}  one agreed history {len(roots) == 1}  "
              f"model acc {acc:.3f}")
        raise SystemExit(0 if len(roots) == 1 and acc > 0.5 else 1)
    peers = [(h, int(p)) for h, p in (x.split(":") for x in a.peers.split(",") if x)]
    node = AsyncGossipNode(a.id, "0.0.0.0", a.port, peers, a.n,
                           interval=a.interval, t0=a.t0 or None)
    tree = asyncio.run(node.run(a.seconds))
    print(f"node {a.id}: height {node.head_height()}, head {node.head_root()[:16]}")


if __name__ == "__main__":
    main()
