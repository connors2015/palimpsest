"""The PyTorch side of the trainer bridge — pure compute, zero consensus.

Connects to a local palimpsest-node (Rust), receives the head state once, then
per training round: trains the real model for N inner steps on its own GPU and
returns the COMPRESSED quantized pseudo-gradient. The node signs, gossips, and
settles it; when the head advances the node sends a sparse state diff so this
process stays synced without ever re-downloading the model.

  python -m client.miner_bridge --node-port 7999 --model small \
      --data data/stories_train.txt --inner 300 --batch 32 [--device cuda]

Frames: [u32 BE length][bytes]; JSON control messages; the initial state
arrives as a raw i64-LE frame after a {"bin_next": true} header.
"""

import argparse
import base64
import json
import socket
import struct
import time

import numpy as np

from rig.chain import dequantize, quantize
from .compress import Compressor, compress as topk_compress
from .data import ByteData
from .gossip import MODEL_PRESETS
from .gpt import build
from .trainer import DiLoCoMiner, set_flat_params

KEEP_FRAC = 0.02


def _send(sock, obj: dict):
    raw = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _send_bin(sock, raw: bytes):
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _recv(sock) -> bytes:
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            raise ConnectionError("node closed")
        hdr += chunk
    n = struct.unpack(">I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("node closed")
        buf += chunk
    return bytes(buf)


def _payload_json(payload: dict) -> dict:
    return {"n": payload["n"],
            "idx": base64.b64encode(payload["idx"]).decode(),
            "val": base64.b64encode(payload["val"]).decode()}


def _sparse_dense(sp: dict) -> np.ndarray:
    out = np.zeros(sp["n"], dtype=np.int64)
    idx = np.frombuffer(base64.b64decode(sp["idx"]), dtype="<u4")
    val = np.frombuffer(base64.b64decode(sp["val"]), dtype="<i8")
    out[idx.astype(np.int64)] = val
    return out


def run(a):
    cfg = MODEL_PRESETS[a.model]
    model, device = build(cfg, device=a.device)
    data = ByteData(path=a.data, block_size=cfg.block_size, device=device) \
        if a.data else ByteData(block_size=cfg.block_size, device=device)
    miner = DiLoCoMiner(model, data, device)
    comp = Compressor(keep_frac=KEEP_FRAC)
    print(f"miner bridge: {model.num_params()/1e6:.1f}M params on {device}", flush=True)

    while True:                                     # reconnect loop
        try:
            sock = socket.create_connection(("127.0.0.1", a.node_port), timeout=10)
        except OSError:
            time.sleep(2)
            continue
        sock.settimeout(None)
        try:
            _send(sock, {"t": "hello"})
            state = None                            # int64 chain state (our copy)
            height = -1
            while True:
                msg = json.loads(_recv(sock))
                t = msg.get("t")
                if t == "state":
                    raw = _recv(sock)               # the raw i64 frame
                    state = np.frombuffer(raw, dtype="<i8").copy()
                    height = int(msg["height"])
                    set_flat_params(model, dequantize(state))
                    print(f"synced full state @ h{height} "
                          f"({state.size/1e6:.1f}M params)", flush=True)
                elif t == "advance":
                    if state is None:
                        _send(sock, {"t": "resync"})
                        continue
                    state = state + _sparse_dense(msg["sparse"])
                    height = int(msg["height"])
                    set_flat_params(model, dequantize(state))
                elif t == "train":
                    want_h = int(msg["height"])
                    if a.serve_only:
                        continue                # this bridge only generates
                    if state is None or want_h != height:
                        _send(sock, {"t": "resync"})
                        continue
                    delta_int, loss = miner.inner_train(
                        a.inner, a.batch, seed=int(msg.get("seed", 0)))
                    payload = comp.compress(dequantize(delta_int))
                    # inner_train mutated the model; restore chain state so the
                    # next round trains from the agreed head, not our drift
                    set_flat_params(model, dequantize(state))
                    _send(sock, {"t": "delta", "height": want_h, "loss": loss,
                                 "payload": _payload_json(payload)})
                    print(f"h{want_h}: trained {a.inner}x{a.batch}, "
                          f"loss {loss:.3f}", flush=True)
                elif t == "generate":
                    # serve chat from the chain-synced model (works on any
                    # bridge; a --produce-less node makes this a pure server)
                    import torch
                    if state is None:
                        _send(sock, {"t": "generated", "height": -1,
                                     "text": "(model not yet synced)"})
                        continue
                    raw = str(msg.get("prompt", " ")).encode("utf-8")
                    raw = raw[-(model.cfg.block_size - 1):] or b" "
                    n_new = min(int(msg.get("n", 120)), 240)
                    idx = torch.tensor([list(raw)], dtype=torch.long,
                                       device=device)
                    model.eval()
                    with torch.no_grad():
                        out = model.generate(idx, n_new, temperature=0.85)
                    model.train()
                    text = bytes(out[0].tolist()[len(raw):]).decode(
                        "utf-8", errors="replace")
                    _send(sock, {"t": "generated", "height": height,
                                 "text": text})
                    print(f"h{height}: generated {n_new} bytes", flush=True)
        except (ConnectionError, OSError) as e:
            print(f"bridge disconnected ({e}); retrying…", flush=True)
            time.sleep(2)
        finally:
            sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-port", type=int, default=7999)
    ap.add_argument("--model", default="toy", choices=list(MODEL_PRESETS))
    ap.add_argument("--data", default=None)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--serve-only", action="store_true",
                    help="only answer generate requests; never submit deltas")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
