"""Multiprocess Palimpsest node: a coordinator and miner processes.

A transport abstraction lets the same coordinator logic run two ways:
  * InMemoryTransport — miners are callables in-process (deterministic, used
    by the test suite to assert equivalence with the single-process e2e loop).
  * SocketCoordinator / miner_process — miners are real OS processes talking
    to the coordinator over localhost TCP with the length-prefixed protocol.

Rounds are synchronous: each block, the coordinator ships the current weights
+ a beacon-assigned shard to every miner, waits for all deltas (the DiLoCo
outer-sync barrier, §6.1), scores them (§5), applies the block (§3), and
persists it (§3.5). Synchronous rounds keep the run reproducible despite
process nondeterminism — consensus is over the *set* of deltas, which is
fixed by the barrier.
"""

import os
import socket
import time
from dataclasses import dataclass, field

import numpy as np

from .chain import Chain, beacon, dequantize, quantize
from .model import TinyTransformer
from .protocol import recv_msg, send_msg
from .storage import ChainStore

N_MINERS = 6
INCLUDE_K = 4
INNER_STEPS = 5
SHARD_BATCH = 32
EVAL_BATCH = 128
LR = 0.3
DEFAULT_BLOCKS = 40


def miner_work(model, vec_base, shard_seed):
    """One miner's inner loop (§6.2): train on the assigned shard, return delta."""
    rng = np.random.default_rng(shard_seed)
    v = vec_base.copy()
    for _ in range(INNER_STEPS):
        batch = model.sample_batch(rng, SHARD_BATCH)
        v = model.train_step(v, batch, lr=LR, steps=1)
    return quantize(v - vec_base)


def score_and_apply(chain: Chain, model, deltas_by_miner, height):
    """Committee scoring (§5) + deterministic block apply (§3)."""
    w_base = dequantize(chain.w_int)
    eval_batch = model.sample_batch(beacon(height, "eval"), EVAL_BATCH)
    base_loss = model.loss(w_base, eval_batch)
    cands = []
    for mid, delta_int in deltas_by_miner:
        w_cand = w_base + dequantize(delta_int)
        score = base_loss - model.loss(w_cand, eval_batch)
        if score > 0:
            cands.append((score, mid, delta_int))
    cands.sort(key=lambda t: (-t[0], t[1]))
    chosen = cands[:INCLUDE_K]
    chain.apply_block([c[2] for c in chosen], [c[1] for c in chosen])
    return chosen


@dataclass
class RunLog:
    heights: list = field(default_factory=list)
    acc: list = field(default_factory=list)
    included: list = field(default_factory=list)
    rewards: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# In-memory transport (tests / reference)
# --------------------------------------------------------------------------
def run_in_memory(blocks=20, seed=7, store: ChainStore | None = None,
                  model=None) -> tuple[Chain, RunLog]:
    model = model or TinyTransformer()
    chain = Chain(quantize(model.init(np.random.default_rng(seed))))
    if store:
        store.init_genesis(chain.genesis_int)
    log = RunLog()
    test_batch = model.sample_batch(np.random.default_rng(seed + 999), 200)

    for _ in range(blocks):
        h = chain.height
        vec_base = dequantize(chain.w_int)
        deltas = []
        for mid in range(N_MINERS):
            shard_seed = int(beacon(h, f"shard{mid}").integers(1 << 30))
            deltas.append((mid, miner_work(model, vec_base, shard_seed)))
        chosen = score_and_apply(chain, model, deltas, h)
        for score, mid, _ in chosen:
            log.rewards[mid] = log.rewards.get(mid, 0.0) + float(score)
        if store:
            b = chain.blocks[-1]
            store.append_block(b.height, b.deltas_int, b.miner_ids, b.root, chain.w_int)
        log.heights.append(h + 1)
        log.acc.append(model.accuracy(dequantize(chain.w_int), test_batch))
        log.included.append(len(chosen))
    return chain, log


# --------------------------------------------------------------------------
# Socket transport (real processes)
# --------------------------------------------------------------------------
def miner_process(host, port, miner_id):
    """Entry point for a miner subprocess: connect, then serve rounds."""
    model = TinyTransformer()
    sock = socket.create_connection((host, port))
    send_msg(sock, {"type": "hello", "miner_id": miner_id})
    try:
        while True:
            msg = recv_msg(sock)
            if msg["type"] == "stop":
                break
            vec_base = np.frombuffer(msg["vec"], dtype=np.float64).copy()
            delta = miner_work(model, vec_base, msg["shard_seed"])
            send_msg(sock, {"type": "delta", "miner_id": miner_id,
                            "delta": delta.tobytes()})
    finally:
        sock.close()


class SocketCoordinator:
    def __init__(self, blocks=20, seed=7, n_miners=N_MINERS,
                 store: ChainStore | None = None):
        self.blocks, self.seed, self.n_miners, self.store = blocks, seed, n_miners, store
        self.model = TinyTransformer()

    def run(self, host="127.0.0.1", port=0) -> tuple[Chain, RunLog]:
        import multiprocessing as mp
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(self.n_miners)
        port = srv.getsockname()[1]

        procs = [mp.Process(target=miner_process, args=(host, port, mid), daemon=True)
                 for mid in range(self.n_miners)]
        for p in procs:
            p.start()

        conns = {}
        for _ in range(self.n_miners):
            conn, _ = srv.accept()
            hello = recv_msg(conn)
            conns[hello["miner_id"]] = conn

        chain = Chain(quantize(self.model.init(np.random.default_rng(self.seed))))
        if self.store:
            self.store.init_genesis(chain.genesis_int)
        log = RunLog()
        test_batch = self.model.sample_batch(np.random.default_rng(self.seed + 999), 200)

        for _ in range(self.blocks):
            h = chain.height
            vec_bytes = dequantize(chain.w_int).tobytes()
            for mid, conn in conns.items():
                shard_seed = int(beacon(h, f"shard{mid}").integers(1 << 30))
                send_msg(conn, {"type": "train", "vec": vec_bytes,
                                "shard_seed": shard_seed})
            deltas = []
            for mid, conn in conns.items():                 # synchronous barrier
                m = recv_msg(conn)
                deltas.append((m["miner_id"],
                               np.frombuffer(m["delta"], dtype=np.int64).copy()))
            deltas.sort(key=lambda t: t[0])
            chosen = score_and_apply(chain, self.model, deltas, h)
            for score, mid, _ in chosen:
                log.rewards[mid] = log.rewards.get(mid, 0.0) + float(score)
            if self.store:
                b = chain.blocks[-1]
                self.store.append_block(b.height, b.deltas_int, b.miner_ids,
                                        b.root, chain.w_int)
            log.heights.append(h + 1)
            log.acc.append(self.model.accuracy(dequantize(chain.w_int), test_batch))
            log.included.append(len(chosen))

        for conn in conns.values():
            send_msg(conn, {"type": "stop"})
            conn.close()
        srv.close()
        for p in procs:
            p.join(timeout=5)
        return chain, log


def _print_log(chain, log, label):
    print("=" * 64)
    print(f"  PALIMPSEST multiprocess node — {label}")
    print("=" * 64)
    print(f"{'blk':>3} {'root':>11} {'model_acc':>10} {'included':>9}")
    for i in range(len(log.heights)):
        if log.heights[i] % 4 == 0 or log.heights[i] <= 2:
            print(f"{log.heights[i]:>3} {chain.blocks[i].root[:10]:>11} "
                  f"{log.acc[i]:>10.3f} {log.included[i]:>9}")
    print(f"\nmodel accuracy {log.acc[0]:.3f} -> {log.acc[-1]:.3f} over "
          f"{len(log.heights)} blocks")
    top = sorted(log.rewards.items(), key=lambda kv: -kv[1])
    print("miner reward share (by cumulative score): "
          + ", ".join(f"m{m}:{v:.1f}" for m, v in top))
    print("=" * 64)


if __name__ == "__main__":
    t0 = time.time()
    chain, log = SocketCoordinator(blocks=DEFAULT_BLOCKS, seed=7).run()
    _print_log(chain, log, f"{N_MINERS} miner processes / sockets")
    print(f"wall time: {time.time() - t0:.1f}s  (pid {os.getpid()})")
    raise SystemExit(0 if log.acc[-1] > 0.9 else 1)
