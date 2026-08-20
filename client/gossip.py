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
import os
import pickle
import struct
import time
from dataclasses import dataclass, field

_DBG = os.environ.get("GOSSIP_DEBUG")


def _dbg(nid, *a):
    if _DBG:
        print(f"    [dbg n{nid}]", *a, flush=True)

import numpy as np

from rig.blockchain import Block, BlockTree, ValidationError, build_block
from rig.chain import dequantize, quantize, state_root
from rig.crypto import BackpropTx, Key, delta_hash
from .compress import Compressor, decompress
from .data import ByteData
from .gpt import GPTConfig, build
from .trainer import DiLoCoMiner, flat_params, set_flat_params

KEEP_FRAC = 0.02             # top-k delta compression (50x on the wire)

GOSSIP_CFG = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=64)   # small: light gossip
INNER_STEPS = 10
BATCH = 24
INCLUDE_K = 8
GENESIS_SEED = 1337          # network constant: every node's genesis weights match
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
        self.model, self.device = build(GOSSIP_CFG, device=DEVICE_OVERRIDE, seed=GENESIS_SEED)
        self.data = ByteData(block_size=GOSSIP_CFG.block_size, device=self.device)
        self.miner = DiLoCoMiner(self.model, self.data, self.device)
        self.key = Key.generate(f"node{node_id}".encode().ljust(32, b"0"))
        self.tree = BlockTree(quantize(flat_params(self.model)))
        self.mempool, self.seen_tx, self.seen_block = {}, set(), set()
        self.orphans, self.pending = {}, {}       # pending: blocks awaiting bodies
        self.body_store = {}                      # txid -> dense delta, RETAINED
        self.payload_store = {}                   # txid -> compressed payload, RETAINED
        self.comp = Compressor(keep_frac=KEEP_FRAC)

    def head_snapshot(self):
        """Read the current head on the CALLER's thread (cheap, touches the tree)."""
        hh = self.tree.blocks[self.tree.head].header.height
        return hh, dequantize(self.tree.head_state())

    def train_from(self, hh, weights):
        """The heavy GPU work — safe to run in an executor because it touches only
        self.model, never the tree. The main thread keeps installing gossiped
        blocks while this runs, so a fast peer can't starve a slow one. On CPU/CUDA
        PyTorch releases the GIL here; the network loop stays responsive."""
        set_flat_params(self.model, weights)
        delta, loss = self.miner.inner_train(INNER_STEPS, BATCH, seed=hh * 100 + self.node_id)
        return hh, delta, loss

    def train_delta(self):
        """Blocking convenience form (MPS path): snapshot + train on one thread."""
        hh, weights = self.head_snapshot()
        return self.train_from(hh, weights)

    def submit_delta(self, hh, delta_int, outbox):
        """Compress the delta (top-k + error feedback), sign a commitment tx, and
        gossip only the small payload — the body never rides in a block."""
        if hh != self.tree.blocks[self.tree.head].header.height:
            return
        payload = self.comp.compress(dequantize(delta_int))    # small on the wire
        dense = decompress(payload)                            # what everyone commits to
        dh = delta_hash(dense.tobytes())
        ptr = f"da://{dh}"                                     # CONTENT address — unique per body
        tx = BackpropTx(miner=self.key.pub, base_height=hh, shard_id=self.node_id,
                        delta_hash=dh, da_pointer=ptr).signed(self.key)
        if tx.txid() not in self.seen_tx:
            self.seen_tx.add(tx.txid())
            self.mempool[tx.txid()] = (tx, dense)
            self.body_store[tx.txid()] = dense
            self.payload_store[tx.txid()] = payload
            outbox.append(("tx", tx, payload))

    def recv_tx(self, tx, payload, outbox):
        if tx.txid() in self.seen_tx:
            return
        if not tx.verify():
            _dbg(self.node_id, f"tx from shard{tx.shard_id} REJECT (bad sig)")
            return
        dense = decompress(payload)
        if delta_hash(dense.tobytes()) != tx.delta_hash:
            _dbg(self.node_id, f"tx from shard{tx.shard_id} REJECT (hash mismatch)")
            return
        self.seen_tx.add(tx.txid())
        self.mempool[tx.txid()] = (tx, dense)
        self.body_store[tx.txid()] = dense
        self.payload_store[tx.txid()] = payload
        outbox.append(("tx", tx, payload))
        self._retry_pending(outbox)                           # a block may now be complete

    def propose(self, outbox):
        head = self.tree.head
        hh = self.tree.blocks[head].header.height
        cands = [(tx, body) for (tx, body) in self.mempool.values() if tx.base_height == hh]
        if not cands:
            return
        cands.sort(key=lambda t: t[0].txid())
        chosen, seen_miners = [], set()
        for tx, body in cands:                     # at most one delta per miner per block
            if tx.miner in seen_miners:
                continue
            seen_miners.add(tx.miner)
            chosen.append((tx, body))
            if len(chosen) >= INCLUDE_K:
                break
        accepted = [tx for tx, _ in chosen]
        bodies = {tx.da_pointer: body for tx, body in chosen}
        block = build_block(self.tree, head, accepted, bodies,
                            {tx.txid(): 1.0 for tx, _ in chosen}, self.key.pub)
        try:
            became_head = self.tree.add_block(block)           # our own block; guard anyway
        except ValidationError as e:
            _dbg(self.node_id, f"own block rejected: {e}")
            return
        if became_head:
            self.seen_block.add(block.hash)
            self._prune(block)
            outbox.append(("block", block.header, block.txs))  # commitments only

    def recv_block(self, header, txs, outbox):
        bh = header.block_hash()
        if bh in self.seen_block:
            return
        bodies, missing = {}, False
        for tx in txs:                                 # rebuild bodies from the RETAINED store
            if tx.txid() in self.body_store:
                bodies[tx.da_pointer] = self.body_store[tx.txid()]
            else:
                missing = True
        if missing:
            self.pending[bh] = (header, txs)           # wait for the tx(s) to arrive
            outbox.append(("getblock", bh))            # …and request the full block (getdata)
            return
        _dbg(self.node_id, f'block h{header.height} bodies-ready, installing')
        self._install(Block(header, txs, bodies), outbox)

    def serve_block(self, bh, outbox):
        """Answer a getblock request with a compressed full block if we have it."""
        b = self.tree.blocks.get(bh)
        if b is not None and all(tx.txid() in self.payload_store for tx in b.txs):
            payloads = {tx.txid(): self.payload_store[tx.txid()] for tx in b.txs}
            outbox.append(("fullblock", b.header, b.txs, payloads))

    def recv_fullblock(self, header, txs, payloads, outbox):
        """Initial block download / getdata reply: a block plus the COMPRESSED
        payloads for its txs, so a node that missed the txs can reconstruct the
        bodies and catch up — small on the wire (payloads, not dense bodies)."""
        bh = header.block_hash()
        if bh in self.seen_block:
            return
        bodies = {}
        for tx in txs:
            dense = decompress(payloads[tx.txid()])
            if delta_hash(dense.tobytes()) != tx.delta_hash:
                return
            self.body_store[tx.txid()] = dense
            self.payload_store[tx.txid()] = payloads[tx.txid()]
            bodies[tx.da_pointer] = dense
        _dbg(self.node_id, f'fullblock h{header.height} received, installing')
        self._install(Block(header, txs, bodies), outbox)

    def _install(self, block, outbox):
        try:
            self.tree.add_block(block)
        except ValidationError as e:
            if "orphan" in str(e):
                self.orphans.setdefault(block.header.prev_hash, []).append(
                    (block.header, block.txs))
            else:
                _dbg(self.node_id, f"h{block.header.height} INVALID: {e}")
            return
        self.seen_block.add(block.hash)
        _dbg(self.node_id, f'INSTALLED h{block.header.height}, head=h{self.tree.blocks[self.tree.head].header.height}')
        self._prune_txs(block.txs)
        outbox.append(("block", block.header, block.txs))   # relay compact
        for ch, ct in self.orphans.pop(block.hash, []):
            self.recv_block(ch, ct, outbox)

    def _retry_pending(self, outbox):
        for bh, (header, txs) in list(self.pending.items()):
            if all(tx.txid() in self.body_store for tx in txs):
                del self.pending[bh]
                self.recv_block(header, txs, outbox)

    def _prune(self, block):
        self._prune_txs(block.txs)

    def _prune_txs(self, txs):
        for tx in txs:
            self.mempool.pop(tx.txid(), None)

    def rebroadcast(self, outbox):
        for b in self.tree.chain_from_genesis():
            if all(tx.txid() in self.payload_store for tx in b.txs):
                payloads = {tx.txid(): self.payload_store[tx.txid()] for tx in b.txs}
                outbox.append(("fullblock", b.header, b.txs, payloads))

    def val_loss(self):
        set_flat_params(self.model, dequantize(self.tree.head_state()))
        return self.data.estimate_loss(self.model, iters=6)["val"]


class GossipNode:
    def __init__(self, node_id, host, port, peers, n_total, interval=1.5, t0=None):
        self.core = RealCore(node_id)
        self.host, self.port, self.peers, self.n_total = host, port, peers, n_total
        self.interval, self.t0 = interval, t0 or time.time()
        self.writers = set()
        self.peer_ids = set()             # dedup: at most one connection per peer
        self._stop = asyncio.Event()

    async def _peer(self, reader, writer, dialer=False):
        await _send(writer, ("hello", self.core.node_id))
        try:
            hello = await _recv(reader)
            pid = hello[1] if hello and hello[0] == "hello" else None
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            writer.close(); return
        if pid is None:
            writer.close(); return
        # Simultaneous open: both peers dial AND accept, so two connections form.
        # Resolve deterministically — keep the one whose DIALER has the smaller id.
        # Both endpoints compute the same verdict, so exactly one full-duplex
        # channel survives (mismatched closes would leave no working link).
        keep = (self.core.node_id < pid) if dialer else (pid < self.core.node_id)
        _dbg(self.core.node_id, f"hello pid={pid} dialer={dialer} keep={keep} "
                                f"already={pid in self.peer_ids}")
        if not keep or pid in self.peer_ids:
            writer.close(); return
        self.peer_ids.add(pid)
        self.writers.add(writer)
        try:
            for b in self.core.tree.chain_from_genesis():
                if all(tx.txid() in self.core.payload_store for tx in b.txs):
                    pl = {tx.txid(): self.core.payload_store[tx.txid()] for tx in b.txs}
                    await _send(writer, ("fullblock", b.header, b.txs, pl))
            while not self._stop.is_set():
                self._handle(await _recv(reader))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.writers.discard(writer); self.peer_ids.discard(pid); writer.close()

    async def _dial(self, host, port):
        for _ in range(60):
            try:
                r, w = await asyncio.open_connection(host, port)
                _dbg(self.core.node_id, f"dialed {host}:{port} OK")
                await self._peer(r, w, dialer=True)
                _dbg(self.core.node_id, f"dial-peer to {host}:{port} ended")
                return
            except (ConnectionError, OSError) as e:
                _dbg(self.core.node_id, f"dial {host}:{port} retry ({e})")
                await asyncio.sleep(1)

    def _handle(self, msg):
        outbox = []
        if _DBG:
            print(f"    [n{self.core.node_id} RECV {msg[0]}]", flush=True)
        if msg[0] == "tx":
            self.core.recv_tx(msg[1], msg[2], outbox)          # (tx, payload)
        elif msg[0] == "block":
            self.core.recv_block(msg[1], msg[2], outbox)       # (header, txs) — compact
        elif msg[0] == "fullblock":
            self.core.recv_fullblock(msg[1], msg[2], msg[3], outbox)   # (header,txs,payloads)
        elif msg[0] == "getblock":
            self.core.serve_block(msg[1], outbox)              # peer needs a full block
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

    async def _loop(self, seconds, settle=12.0):
        loop = asyncio.get_event_loop()
        # BACKPRESSURE FIX: on CPU/CUDA, run training in an executor so the network
        # event loop keeps draining gossip while we train — a fast peer can no
        # longer flood a slow peer into starvation (they stay head-synced because
        # the slow peer installs received blocks *during* its own training). MPS
        # misbehaves off the main thread, so it falls back to blocking training.
        use_executor = self.core.device != "mps"
        end = time.time() + seconds
        while time.time() < end:
            if use_executor:
                hh, weights = self.core.head_snapshot()          # read tree on main thread
                hh, delta, loss = await loop.run_in_executor(     # heavy work off-loop
                    None, self.core.train_from, hh, weights)
            else:
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
        lineage = ">".join(b.hash[:6] for b in self.core.tree.chain_from_genesis())
        print(f"node {self.core.node_id} LINEAGE {lineage}", flush=True)
        print(f"node {self.core.node_id} done — height {h}, head {self.core.tree.head[:16]}, "
              f"seen_tx {len(self.core.seen_tx)} seen_block {len(self.core.seen_block)} "
              f"pending {len(self.core.pending)} peers {len(self.writers)}, "
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
