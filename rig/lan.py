"""Cross-machine Palimpsest node — coordinator + miners over a real network.

`rig/node.py` spawns miners as local subprocesses; this runs them as separate
machines. The wire protocol (rig/protocol.py) was already TCP, so the only
changes are: the coordinator binds 0.0.0.0 and waits for *external* miners to
connect, and the miner dials a coordinator by IP. Rounds stay synchronous (the
DiLoCo outer-sync barrier), so consensus is reproducible even though the miners
now live on different machines with different clocks.

Both sides must agree on the model architecture (they exchange flat weight
vectors), so the model is fixed here — a shared, fast TinyTransformer.

  # on the coordinator machine:
  python3 -m rig.lan coordinator --port 9000 --miners 3 --blocks 30

  # on each miner machine (point at the coordinator's IP):
  python3 -m rig.lan miner --host 100.x.y.z --port 9000 --id 0
"""

import argparse
import socket
import time

import numpy as np

from .chain import Chain, beacon, dequantize, quantize, state_root
from .model import TinyTransformer
from .node import miner_work, score_and_apply
from .protocol import recv_msg, send_msg

MODEL = TinyTransformer()          # architecture shared by coordinator and miners


def run_coordinator(port, n_miners, blocks, seed=7, host="0.0.0.0"):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(n_miners)
    print(f"coordinator listening on {host}:{port}, waiting for {n_miners} miners…",
          flush=True)

    conns = {}
    while len(conns) < n_miners:
        conn, addr = srv.accept()
        hello = recv_msg(conn)
        conns[hello["miner_id"]] = conn
        print(f"  miner {hello['miner_id']} joined from {addr[0]}:{addr[1]} "
              f"({len(conns)}/{n_miners})", flush=True)

    chain = Chain(quantize(MODEL.init(np.random.default_rng(seed))))
    test = MODEL.sample_batch(np.random.default_rng(seed + 999), 200)
    t0 = time.time()
    for _ in range(blocks):
        h = chain.height
        vec_bytes = dequantize(chain.w_int).tobytes()
        for mid, conn in conns.items():
            shard_seed = int(beacon(h, f"shard{mid}").integers(1 << 30))
            send_msg(conn, {"type": "train", "vec": vec_bytes, "shard_seed": shard_seed})
        deltas = []
        for mid, conn in conns.items():                    # synchronous barrier
            m = recv_msg(conn)
            deltas.append((m["miner_id"],
                           np.frombuffer(m["delta"], dtype=np.int64).copy()))
        deltas.sort(key=lambda t: t[0])
        chosen = score_and_apply(chain, MODEL, deltas, h)
        acc = MODEL.accuracy(dequantize(chain.w_int), test)
        print(f"  block {h+1:>3}  acc {acc:5.3f}  root {chain.blocks[-1].root[:10]}  "
              f"included {len(chosen)}", flush=True)

    for conn in conns.values():
        send_msg(conn, {"type": "stop"})
        conn.close()
    srv.close()
    replay_ok = state_root(chain.replay()) == chain.blocks[-1].root
    print(f"\ndone: {blocks} blocks in {time.time()-t0:.1f}s, "
          f"final acc {MODEL.accuracy(dequantize(chain.w_int), test):.3f}, "
          f"replay bit-exact {replay_ok}, head {chain.blocks[-1].root[:16]}", flush=True)
    return chain


def run_miner(host, port, miner_id):
    sock = socket.create_connection((host, port))
    send_msg(sock, {"type": "hello", "miner_id": miner_id})
    print(f"miner {miner_id} connected to {host}:{port}", flush=True)
    n = 0
    try:
        while True:
            msg = recv_msg(sock)
            if msg["type"] == "stop":
                break
            vec = np.frombuffer(msg["vec"], dtype=np.float64).copy()
            delta = miner_work(MODEL, vec, msg["shard_seed"])
            send_msg(sock, {"type": "delta", "miner_id": miner_id,
                            "delta": delta.tobytes()})
            n += 1
    finally:
        sock.close()
    print(f"miner {miner_id} done ({n} rounds)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="role", required=True)
    c = sub.add_parser("coordinator")
    c.add_argument("--port", type=int, default=9000)
    c.add_argument("--miners", type=int, default=2)
    c.add_argument("--blocks", type=int, default=30)
    c.add_argument("--seed", type=int, default=7)
    m = sub.add_parser("miner")
    m.add_argument("--host", required=True)
    m.add_argument("--port", type=int, default=9000)
    m.add_argument("--id", type=int, required=True)
    a = ap.parse_args()
    if a.role == "coordinator":
        run_coordinator(a.port, a.miners, a.blocks, a.seed)
    else:
        run_miner(a.host, a.port, a.id)


if __name__ == "__main__":
    main()
