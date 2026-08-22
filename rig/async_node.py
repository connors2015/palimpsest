"""Asynchronous miners with real staleness (WHITEPAPER §4.1, §6.2).

The synchronous barrier in rig/node.py is a fiction: in a permissionless
network miners run at different speeds and submit whenever they finish, so a
delta computed against block N may not be scored until the head is at N+lag.
This module models that honestly.

Two mechanisms guard staleness (§4.1):
  * a stale delta is scored against the *current* head, not the head it was
    computed on — so it is included only if it still improves the live model;
  * reward decays with lag, and deltas older than GRACE_G blocks are dropped.

`run_async_sim` is an event-driven, fully seeded simulator (heterogeneous
work durations, a real mempool) — reproducible, so the test suite can pin it.
`AsyncSocketCoordinator` runs the same logic with miner processes over
sockets and a threaded, wall-clock block cadence (genuinely non-deterministic;
a demo, not a test).
"""

import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .chain import Chain, beacon, dequantize, quantize
from .model import TinyTransformer
from .protocol import recv_msg, send_msg

N_MINERS = 6
BLOCK_TICKS = 2          # simulator ticks between blocks
GRACE_G = 3              # max staleness (blocks) before a delta is dropped (§4.1)
INCLUDE_K = 4
INNER_STEPS = 5
SHARD_BATCH = 32
EVAL_BATCH = 128
LR = 0.3
FAST_MINERS = 3          # miners 0..2 are fast; the rest are slow (more staleness)


def _staleness_decay(lag: int) -> float:
    return max(0.0, 1.0 - lag / (GRACE_G + 1))


def _miner_delta(model, base_vec, round_seed):
    rng = np.random.default_rng(round_seed)
    v = base_vec.copy()
    for _ in range(INNER_STEPS):
        v = model.train_step(v, model.sample_batch(rng, SHARD_BATCH), lr=LR, steps=1)
    return quantize(v - base_vec)


def _score_mempool(model, chain, mempool):
    """Score pending deltas vs the CURRENT head; drop stale; return top-K + drops."""
    head_h = chain.height
    w_head = dequantize(chain.w_int)
    base_loss = model.loss(w_head, model.sample_batch(beacon(head_h, "eval"), EVAL_BATCH))
    cands, dropped, keep = [], [], []
    for item in mempool:
        lag = head_h - item["base_height"]
        if lag > GRACE_G:
            dropped.append(item)                       # too stale (§4.1)
            continue
        score = base_loss - model.loss(w_head + dequantize(item["delta"]),
                                       model.sample_batch(beacon(head_h, "eval"),
                                                          EVAL_BATCH))
        if score > 0:
            cands.append((score, lag, item))
        else:
            keep.append(item)                          # not helpful now; revisit next block
    cands.sort(key=lambda t: (-t[0], t[2]["miner_id"]))
    chosen = cands[:INCLUDE_K]
    leftover = keep + [c[2] for c in cands[INCLUDE_K:]]
    return chosen, leftover, dropped


@dataclass
class AsyncLog:
    heights: list = field(default_factory=list)
    acc: list = field(default_factory=list)
    included: list = field(default_factory=list)
    rewards: dict = field(default_factory=dict)
    stale_dropped: int = 0
    included_lags: list = field(default_factory=list)


def run_async_sim(ticks=120, seed=7, model=None) -> tuple[Chain, AsyncLog]:
    model = model or TinyTransformer()
    chain = Chain(quantize(model.init(np.random.default_rng(seed))))
    log = AsyncLog()
    test_batch = model.sample_batch(np.random.default_rng(seed + 999), 200)

    def duration(mid, rnd):
        r = beacon(rnd, f"dur{mid}")
        return int(r.integers(1, 3)) if mid < FAST_MINERS else int(r.integers(4, 9))

    # Each miner: (base_height, base_vec, ready_tick, round_counter)
    miners = {}
    for mid in range(N_MINERS):
        miners[mid] = dict(base_h=0, base_vec=dequantize(chain.w_int),
                           ready=duration(mid, 0), rnd=0)
    mempool = []

    for t in range(1, ticks + 1):
        # --- completions: miners that finish this tick submit, then re-sync ---
        for mid in range(N_MINERS):
            m = miners[mid]
            if m["ready"] != t:
                continue
            delta = _miner_delta(model, m["base_vec"],
                                 int(beacon(m["rnd"], f"work{mid}").integers(1 << 30)))
            mempool.append(dict(miner_id=mid, base_height=m["base_h"], delta=delta))
            # re-sync to the current head and start the next round
            m["base_h"] = chain.height
            m["base_vec"] = dequantize(chain.w_int)
            m["rnd"] += 1
            m["ready"] = t + duration(mid, m["rnd"])

        # --- block production on cadence ------------------------------------
        if t % BLOCK_TICKS == 0:
            chosen, leftover, dropped = _score_mempool(model, chain, mempool)
            chain.apply_block([c[2]["delta"] for c in chosen],
                              [c[2]["miner_id"] for c in chosen])
            for score, lag, item in chosen:
                log.rewards[item["miner_id"]] = log.rewards.get(item["miner_id"], 0.0) \
                    + float(score) * _staleness_decay(lag)
                log.included_lags.append(lag)
            log.stale_dropped += len(dropped)
            mempool = leftover
            log.heights.append(chain.height)
            log.acc.append(model.accuracy(dequantize(chain.w_int), test_batch))
            log.included.append(len(chosen))

    return chain, log


# --------------------------------------------------------------------------
# Real socket async (threaded coordinator, miner processes) — demo, not a test
# --------------------------------------------------------------------------
def async_miner_process(host, port, miner_id):
    model = TinyTransformer()
    sock = socket.create_connection((host, port))
    send_msg(sock, {"type": "hello", "miner_id": miner_id})
    rnd = 0
    try:
        while True:
            send_msg(sock, {"type": "sync"})
            msg = recv_msg(sock)
            if msg["type"] == "stop":
                break
            base_vec = np.frombuffer(msg["vec"], dtype=np.float64).copy()
            base_h = msg["height"]
            if miner_id >= FAST_MINERS:
                time.sleep(0.02)                       # slow miners lag -> staleness
            delta = _miner_delta(model, base_vec, (miner_id << 20) + rnd)
            rnd += 1
            send_msg(sock, {"type": "delta", "miner_id": miner_id,
                            "base_height": base_h, "delta": delta.tobytes()})
    finally:
        sock.close()


class AsyncSocketCoordinator:
    def __init__(self, blocks=30, seed=7, n_miners=N_MINERS, block_period=0.05):
        self.blocks, self.seed, self.n_miners = blocks, seed, n_miners
        self.block_period = block_period
        self.model = TinyTransformer()
        self.chain = Chain(quantize(self.model.init(np.random.default_rng(seed))))
        self.mempool = []
        self.lock = threading.Lock()
        self.stop = False

    def _serve_miner(self, conn):
        try:
            while not self.stop:
                msg = recv_msg(conn)
                if msg["type"] == "sync":
                    with self.lock:
                        vec = dequantize(self.chain.w_int).tobytes()
                        h = self.chain.height
                    send_msg(conn, {"type": "weights", "vec": vec, "height": h})
                elif msg["type"] == "delta":
                    with self.lock:
                        self.mempool.append(dict(
                            miner_id=msg["miner_id"], base_height=msg["base_height"],
                            delta=np.frombuffer(msg["delta"], dtype=np.int64).copy()))
        except (ConnectionError, OSError):
            pass

    def run(self, host="127.0.0.1", port=0):
        import multiprocessing as mp
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(self.n_miners)
        port = srv.getsockname()[1]
        procs = [mp.Process(target=async_miner_process, args=(host, port, mid),
                            daemon=True) for mid in range(self.n_miners)]
        for p in procs:
            p.start()

        threads = []
        for _ in range(self.n_miners):
            conn, _ = srv.accept()
            recv_msg(conn)  # hello
            th = threading.Thread(target=self._serve_miner, args=(conn,), daemon=True)
            th.start()
            threads.append(th)

        log = AsyncLog()
        test_batch = self.model.sample_batch(np.random.default_rng(self.seed + 999), 200)
        for _ in range(self.blocks):
            time.sleep(self.block_period)
            with self.lock:
                chosen, leftover, dropped = _score_mempool(self.model, self.chain,
                                                           self.mempool)
                self.chain.apply_block([c[2]["delta"] for c in chosen],
                                       [c[2]["miner_id"] for c in chosen])
                self.mempool = leftover
            for score, lag, item in chosen:
                log.rewards[item["miner_id"]] = log.rewards.get(item["miner_id"], 0.0) \
                    + float(score) * _staleness_decay(lag)
                log.included_lags.append(lag)
            log.stale_dropped += len(dropped)
            log.heights.append(self.chain.height)
            log.acc.append(self.model.accuracy(dequantize(self.chain.w_int), test_batch))
            log.included.append(len(chosen))

        self.stop = True
        srv.close()
        for p in procs:
            p.terminate()
            p.join(timeout=3)
        return self.chain, log


def _print_async(chain, log, label):
    print("=" * 66)
    print(f"  SESTRIAN async node — {label}")
    print("=" * 66)
    print(f"{'blk':>3} {'root':>11} {'model_acc':>10} {'incl':>5} {'avg_lag':>8}")
    lag_so_far = []
    for i in range(len(log.heights)):
        lag_so_far = log.included_lags[:sum(log.included[:i + 1])]
        if log.heights[i] % 5 == 0 or log.heights[i] <= 2:
            avg = np.mean(lag_so_far) if lag_so_far else 0.0
            print(f"{log.heights[i]:>3} {chain.blocks[i].root[:10]:>11} "
                  f"{log.acc[i]:>10.3f} {log.included[i]:>5} {avg:>8.2f}")
    print(f"\nmodel accuracy {log.acc[0]:.3f} -> {log.acc[-1]:.3f} over "
          f"{len(log.heights)} blocks")
    print(f"stale deltas dropped (lag > {GRACE_G}):      {log.stale_dropped}")
    print(f"avg staleness of included deltas:  "
          f"{np.mean(log.included_lags) if log.included_lags else 0:.2f} blocks")
    top = sorted(log.rewards.items(), key=lambda kv: -kv[1])
    fast = sum(v for m, v in log.rewards.items() if m < FAST_MINERS)
    slow = sum(v for m, v in log.rewards.items() if m >= FAST_MINERS)
    print(f"reward — fast miners {fast:.1f} vs slow miners {slow:.1f} "
          f"(slow earn less: staleness discount at work)")
    print("miner rewards: " + ", ".join(f"m{m}:{v:.1f}" for m, v in top))
    print("=" * 66)


if __name__ == "__main__":
    import sys
    if "--moe" in sys.argv:
        # the fused MoE transformer, trained through the async path
        from .moe_transformer import MoETConfig, MoETransformer
        model = MoETransformer(MoETConfig(n_experts=4, top_k=2))
        chain, log = run_async_sim(ticks=200, seed=7, model=model)
        _print_async(chain, log, "MoE transformer, async simulator")
    elif "--sim" in sys.argv:
        chain, log = run_async_sim()
        _print_async(chain, log, "deterministic simulator")
    else:
        t0 = time.time()
        chain, log = AsyncSocketCoordinator().run()
        _print_async(chain, log, f"{N_MINERS} miner processes / async sockets")
        print(f"wall time: {time.time() - t0:.1f}s")
    raise SystemExit(0 if log.acc[-1] > 0.85 else 1)
