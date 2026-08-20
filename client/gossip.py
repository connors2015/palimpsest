"""The real client, coordinator-free — gossip consensus over real GPU training.

No central node. Every peer holds a BlockTree, trains the real GPT on its own
GPU, and gossips signed pseudo-gradient deltas and blocks to its peers over
async sockets. Leadership rotates by wall-clock round (leader = round mod N), so
the right to propose a block simply takes turns — nobody coordinates. Fork
choice is Nakamoto heaviest-valid-chain (rig/blockchain.py); a peer joining or
reconnecting re-announces its chain so forks reconcile.

This is client/chain_node.py with the coordinator removed. The delta bodies are
real (megabytes), so the model here is deliberately small to keep gossip light;
the mechanism is identical at any size.

  # on each machine (peers is host:port,host:port of the OTHER nodes):
  python -m client.gossip --id 0 --port 9850 --peers 100.x:9851 --n 2 --seconds 30
"""

import argparse
import asyncio
import pickle
import struct
import time
from dataclasses import dataclass, field

import numpy as np

from rig.blockchain import BlockTree, ValidationError, build_block
from rig.chain import dequantize, quantize, state_root
from rig.crypto import BackpropTx, Key, delta_hash
from .data import ByteData
from .gpt import GPTConfig, build
from .trainer import DiLoCoMiner, flat_params, set_flat_params

GOSSIP_CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=64)   # small: light gossip
INNER_STEPS = 10
BATCH = 24
INCLUDE_K = 8
DEVICE_OVERRIDE = None       # set by --device


async def _send(w, obj):
    data = pickle.dumps(obj)
    w.write(struct.pack(">I", len(data)) + data); await w.drain()


async def _recv(r):
    (n,) = struct.unpack(">I", await r.readexactly(4))
    return pickle.loads(await r.readexactly(n))


class RealCore:
    """Consensus + real-model training for one gossip node."""

    def __init__(self, node_id, seed=0):
        self.node_id = node_id
        self.model, self.device = build(GOSSIP_CFG, device=DEVICE_OVERRIDE)
        self.data = ByteData(block_size=GOSSIP_CFG.block_size, device=self.device)
        self.miner = DiLoCoMiner(self.model, self.data, self.device)
        self.key = Key.generate(f"node{node_id}".encode().ljust(32, b"0"))
        self.tree = BlockTree(quantize(flat_params(self.model)))
        self.mempool, self.seen_tx, self.seen_block, self.orphans = {}, set(), set(), {}
        self._round = 0

    def train_delta(self):
        """Blocking GPU work — run in an executor so it never stalls gossip."""
        hh = self.tree.blocks[self.tree.head].header.height
        set_flat_params(self.model, dequantize(self.tree.head_state()))
        delta, loss = self.miner.inner_train(INNER_STEPS, BATCH, seed=hh * 100 + self.node_id)
        return hh, delta, loss

    def submit_delta(self, hh, delta, outbox):
        """Assemble + sign the tx and gossip it (main loop — mutates state)."""
        ptr = f"da://{self.key.pub[:8]}/{hh}/{self.node_id}"
        tx = BackpropTx(miner=self.key.pub, base_height=hh, shard_id=self.node_id,
                        delta_hash=delta_hash(delta.tobytes()), da_pointer=ptr).signed(self.key)
        if tx.txid() not in self.seen_tx and hh == self.tree.blocks[self.tree.head].header.height:
            self.seen_tx.add(tx.txid())
            self.mempool[tx.txid()] = (tx, delta)
            outbox.append(("tx", tx, delta.tobytes()))

    def recv_tx(self, tx, body_bytes, outbox):
        if tx.txid() in self.seen_tx or not tx.verify():
            return
        body = np.frombuffer(body_bytes, dtype=np.int64).copy()
        if delta_hash(body.tobytes()) != tx.delta_hash:
            return
        self.seen_tx.add(tx.txid())
        self.mempool[tx.txid()] = (tx, body)
        outbox.append(("tx", tx, body_bytes))

    def propose(self, outbox):
        head = self.tree.head
        hh = self.tree.blocks[head].header.height
        cands = [(tx, body) for (tx, body) in self.mempool.values() if tx.base_height == hh]
        if not cands:
            return
        cands.sort(key=lambda t: t[0].txid())
        chosen = cands[:INCLUDE_K]
        accepted = [tx for tx, _ in chosen]
        bodies = {tx.da_pointer: body for tx, body in chosen}
        works = {tx.txid(): 1.0 for tx, _ in chosen}
        block = build_block(self.tree, head, accepted, bodies, works, self.key.pub)
        if self.tree.add_block(block):
            self.seen_block.add(block.hash)
            self._prune(block)
            outbox.append(("block", block))

    def recv_block(self, block, outbox):
        if block.hash in self.seen_block:
            return
        try:
            self.tree.add_block(block)
        except ValidationError as e:
            if "orphan" in str(e):
                self.orphans.setdefault(block.header.prev_hash, []).append(block)
            return
        self.seen_block.add(block.hash)
        self._prune(block)
        outbox.append(("block", block))
        for child in self.orphans.pop(block.hash, []):
            self.recv_block(child, outbox)

    def _prune(self, block):
        for tx in block.txs:
            self.mempool.pop(tx.txid(), None)

    def rebroadcast(self, outbox):
        for b in self.tree.chain_from_genesis():
            outbox.append(("block", b))

    def val_loss(self):
        set_flat_params(self.model, dequantize(self.tree.head_state()))
        return self.data.estimate_loss(self.model, iters=6)["val"]


class GossipNode:
    def __init__(self, node_id, host, port, peers, n_total, interval=1.5, t0=None):
        self.core = RealCore(node_id)
        self.host, self.port, self.peers, self.n_total = host, port, peers, n_total
        self.interval, self.t0 = interval, t0 or time.time()
        self.writers = set()
        self._stop = asyncio.Event()

    async def _peer(self, reader, writer):
        self.writers.add(writer)
        try:
            await _send(writer, ("hello", self.core.node_id))
            for b in self.core.tree.chain_from_genesis():
                await _send(writer, ("block", b))
            while not self._stop.is_set():
                self._handle(await _recv(reader))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.writers.discard(writer); writer.close()

    async def _dial(self, host, port):
        for _ in range(60):
            try:
                r, w = await asyncio.open_connection(host, port)
                await self._peer(r, w); return
            except (ConnectionError, OSError):
                await asyncio.sleep(1)

    def _handle(self, msg):
        outbox = []
        if msg[0] == "tx":
            self.core.recv_tx(msg[1], msg[2], outbox)
        elif msg[0] == "block":
            self.core.recv_block(msg[1], outbox)
        for m in outbox:
            self._bcast(m)

    def _bcast(self, msg):
        for w in list(self.writers):
            asyncio.create_task(self._safe(w, msg))

    async def _safe(self, w, msg):
        try:
            await _send(w, msg)
        except (ConnectionError, OSError):
            self.writers.discard(w)

    async def _loop(self, seconds, settle=5.0):
        end = time.time() + seconds
        while time.time() < end:
            # train on the main thread (MPS dislikes background threads), then
            # yield so queued gossip is processed before the next burst
            hh, delta, loss = self.core.train_delta()
            outbox = []
            self.core.submit_delta(hh, delta, outbox)
            rnd = int((time.time() - self.t0) / self.interval)
            if rnd % self.n_total == self.core.node_id:         # rotating leader
                self.core.propose(outbox)
            for m in outbox:
                self._bcast(m)
            h = self.core.tree.blocks[self.core.tree.head].header.height
            print(f"  node {self.core.node_id}  height {h}  inner loss {loss:.3f}", flush=True)
            await asyncio.sleep(self.interval)                  # let gossip flow
        # quiescent settle so in-flight blocks land and heads converge
        for _ in range(int(settle / self.interval) + 1):
            await asyncio.sleep(self.interval)
            outbox = []; self.core.rebroadcast(outbox)
            for m in outbox:
                self._bcast(m)
        self._stop.set()

    async def run(self, seconds):
        server = await asyncio.start_server(self._peer, self.host, self.port)
        async with server:
            await asyncio.sleep(0.5)
            dials = [asyncio.create_task(self._dial(h, p)) for h, p in self.peers]
            await self._loop(seconds)
            for d in dials:
                d.cancel()
        h = self.core.tree.blocks[self.core.tree.head].header.height
        print(f"node {self.core.node_id} done — height {h}, head {self.core.tree.head[:16]}, "
              f"val loss {self.core.val_loss():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--peers", default="")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--device", default=None)      # cuda|mps|cpu (auto if unset)
    a = ap.parse_args()
    global DEVICE_OVERRIDE
    DEVICE_OVERRIDE = a.device
    peers = [(h, int(p)) for h, p in (x.split(":") for x in a.peers.split(",") if x)]
    node = GossipNode(a.id, "0.0.0.0", a.port, peers, a.n,
                      interval=a.interval, t0=a.t0 or None)
    asyncio.run(node.run(a.seconds))


if __name__ == "__main__":
    main()
