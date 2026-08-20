"""Live sharded DiPaCo node — each machine holds DIFFERENT paths, no node holds
the whole model (WHITEPAPER §3.1, §3.3, §6).

This wires together the three pieces we built:

  * dipaco.py — the model is L levels x M modules; a PATH picks one module per
    level; a node OWNS a set of paths and holds only the backbone + those paths'
    modules. It trains its paths locally (coarse routing → no activation
    exchange, no pipeline).
  * cas.py — pages are content-addressed. A node holds the CONTENT of its own
    pages and only the CID of everyone else's. The agreed model ROOT is the hash
    of the manifest (all page CIDs), so a node needs peers' CIDs, not their
    weights, to know and verify the global root — which is exactly what makes
    "no node holds the whole model" literally true.
  * gossip.py transport — the async socket layer with the deterministic
    simultaneous-open fix, reused verbatim.

Per synchronous round (a DiLoCo outer step): every node trains its own paths,
then broadcasts (a) its pseudo-gradient on the SHARED backbone and (b) the new
CIDs of the modules it owns. Nodes average the backbone over its holders (all of
them) and adopt each module's CID from its owner, then recompute the manifest
root. Because every node applies the same backbone average and the same set of
module CIDs, they agree on the root bit-for-bit — a sharded consensus in which
each node held only its slice. Module CONTENT moves only on demand, by CID, over
Bitswap (so a node can reconstruct the full model to serve).

  # node A owns paths 0,1 (trains domains 0,1); node B owns paths 2,3:
  python -m client.dipaco_node --id 0 --port 9800 --peers HOST:9801 --n 2 \
      --paths 0,1 --rounds 20 --device cuda --t0 <shared>
"""

import argparse
import asyncio
import time

import numpy as np

from rig.chain import dequantize, quantize
from .cas import Bitswap, ContentStore, cid
from .dipaco import (
    DiPaCoConfig, PathMap, build_dipaco, coarse_route, make_path,
)
from .gossip import _recv, _send
from .trainer import flat_params, set_flat_params

GENESIS_SEED = 4242
CFG = DiPaCoConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, n_modules=4)
N_PATHS = CFG.n_modules            # one path per module id (make_path)
INNER_STEPS = 30
BATCH = 16
DEVICE_OVERRIDE = None


def _domain_buf(d, n=4096):
    """Domain d = a repeating stream of a domain-specific period (distinct
    next-byte function per domain, so paths have something to specialize on)."""
    return (np.arange(n) % (5 + d)).astype(np.int64)


def _get_batch(buf, bs, T, gen, device):
    ix = np.asarray(torch_randint(len(buf) - T - 1, bs, gen))
    import torch
    x = torch.stack([torch.from_numpy(buf[i:i + T]) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(buf[i + 1:i + 1 + T]) for i in ix]).to(device)
    return x, y


def torch_randint(hi, n, gen):
    import torch
    return torch.randint(0, hi, (n,), generator=gen).tolist()


# --------------------------------------------------------------------------
# Page-DAG state: the agreed model is a manifest of page CIDs
# --------------------------------------------------------------------------
def manifest_root(cids: dict) -> str:
    """The model root = hash over all page CIDs in canonical order. Needs only
    CIDs, so a node computes it without holding peers' page contents."""
    items = ";".join(f"{k}={cids[k]}" for k in sorted(cids, key=str))
    return cid(items.encode())


class DiPaCoNode:
    def __init__(self, node_id, host, port, peers, n_total, owned_paths,
                 interval=2.0, t0=None):
        self.node_id = node_id
        self.host, self.port, self.peers, self.n_total = host, port, peers, n_total
        self.interval, self.t0 = interval, t0 or time.time()
        self.owned_paths = owned_paths                       # list of path ids
        self.model, self.device = build_dipaco(CFG, device=DEVICE_OVERRIDE,
                                                seed=GENESIS_SEED)
        self.pm = PathMap(self.model)
        self.store = ContentStore()
        self.bitswap = Bitswap(self.store)

        # page keys: ("bb",) backbone; ("mod", level, module) each module
        self.bb_idx = self.pm.backbone_idx
        self.mod_span = self.pm.mod_span                     # (l,m) -> (start,end)
        self.owned_mods = set()                              # (l,m) this node trains
        for pid in owned_paths:
            for l, m in enumerate(make_path(pid, CFG.n_layer, CFG.n_modules)):
                self.owned_mods.add((l, m))

        # genesis is deterministic → every node can materialize ALL pages, keep
        # its own contents + everyone's CIDs, and drop the rest.
        g = quantize(flat_params(self.model))
        self.bb = g[self.bb_idx].copy()                      # backbone content (held by all)
        self.mod_content = {k: g[s:e].copy() for k, (s, e) in self.mod_span.items()
                            if k in self.owned_mods}         # only OWNED module content
        self.cids = {("bb",): cid(self.bb.tobytes())}
        for k, (s, e) in self.mod_span.items():
            self.cids[("mod", *k)] = cid(g[s:e].tobytes())   # genesis CIDs match everywhere
        for k in self.mod_span:                              # seed Bitswap with owned content
            if k in self.owned_mods:
                self.store.put(g[self.mod_span[k][0]:self.mod_span[k][1]].tobytes())

        self.genesis_full = g                                # for assembling the local model
        self.writers, self.peer_ids = set(), set()
        self.inbox = {}                                      # round -> {node_id: update}
        self._stop = asyncio.Event()

    # ---- local model assembly + training ---------------------------------
    def _assemble(self):
        """Build the flat vector for a forward pass: owned pages authoritative,
        everyone else's modules left at genesis (they aren't on our paths, so the
        forward never touches them)."""
        full = self.genesis_full.copy()
        full[self.bb_idx] = self.bb
        for k, content in self.mod_content.items():
            s, e = self.mod_span[k]
            full[s:e] = content
        return full

    def train_round(self, r):
        """Train each owned path on its domain; return (backbone_delta, new module
        contents for owned modules). Only owned modules + backbone move."""
        import torch
        set_flat_params(self.model, dequantize(self._assemble()))
        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        for pid in self.owned_paths:
            path = make_path(pid, CFG.n_layer, CFG.n_modules)
            buf = _domain_buf(coarse_route(pid, N_PATHS))
            gen = torch.Generator().manual_seed(r * 100 + pid)
            for _ in range(INNER_STEPS):
                x, y = _get_batch(buf, BATCH, CFG.block_size, gen, self.device)
                _, loss = self.model(x, y, path=path)
                opt.zero_grad(); loss.backward(); opt.step()
        new = quantize(flat_params(self.model))
        bb_delta = new[self.bb_idx] - self.bb
        new_mods = {k: new[self.mod_span[k][0]:self.mod_span[k][1]].copy()
                    for k in self.owned_mods}
        return bb_delta, new_mods

    def val_losses(self):
        import torch
        set_flat_params(self.model, dequantize(self._assemble()))
        out = {}
        for pid in self.owned_paths:
            path = make_path(pid, CFG.n_layer, CFG.n_modules)
            buf = _domain_buf(coarse_route(pid, N_PATHS))
            gen = torch.Generator().manual_seed(7777)
            with torch.no_grad():
                x, y = _get_batch(buf, 64, CFG.block_size, gen, self.device)
                _, loss = self.model(x, y, path=path)
            out[pid] = loss.item()
        return out

    def apply_round(self, r, bb_delta, new_mods):
        """Aggregate this round: backbone averaged over ALL holders (every node),
        each owned module taken from its owner. Update CIDs and store content so
        peers can Bitswap it. Peer modules are adopted as CIDs only."""
        deltas = [bb_delta] + [u["bb_delta"] for u in self.inbox.get(r, {}).values()]
        agg = np.sum(np.stack(deltas), axis=0) // len(deltas)   # integer mean over holders
        self.bb = self.bb + agg
        self.cids[("bb",)] = cid(self.bb.tobytes())
        for k, content in new_mods.items():                     # our own modules
            self.mod_content[k] = content
            c = cid(content.tobytes())
            self.cids[("mod", *k)] = c
            self.store.put(content.tobytes())
        for u in self.inbox.get(r, {}).values():                # peers' module CIDs
            for key_s, c in u["mod_cids"].items():
                self.cids[key_s] = c
        return manifest_root(self.cids)

    # ---- reconstruction (serving) via Bitswap ----------------------------
    def reconstruct_full(self):
        """Assemble the WHOLE model by fetching every module's content by CID from
        the local store (populated on demand via Bitswap). Returns the full vector
        if every page is present, else the set of still-missing CIDs."""
        full = self.genesis_full.copy()
        full[self.bb_idx] = self.bb
        missing = []
        for k, (s, e) in self.mod_span.items():
            c = self.cids[("mod", *k)]
            blob = self.store.get(c)
            if blob is None:
                missing.append(c)
            else:
                full[s:e] = np.frombuffer(blob, dtype=np.int64)
        return (None, missing) if missing else (full, [])

    # ---- networking (reuses the gossip transport + keep rule) ------------
    async def _peer(self, reader, writer, dialer=False):
        await _send(writer, ("hello", self.node_id))
        try:
            hello = await _recv(reader)
            pid = hello[1] if hello and hello[0] == "hello" else None
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            writer.close(); return
        if pid is None:
            writer.close(); return
        keep = (self.node_id < pid) if dialer else (pid < self.node_id)
        if not keep or pid in self.peer_ids:
            writer.close(); return
        self.peer_ids.add(pid); self.writers.add(writer)
        try:
            while not self._stop.is_set():
                msg = await _recv(reader)
                self._handle(msg)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.writers.discard(writer); self.peer_ids.discard(pid); writer.close()

    async def _dial(self, host, port):
        for _ in range(90):
            try:
                r, w = await asyncio.open_connection(host, port)
                await self._peer(r, w, dialer=True); return
            except (ConnectionError, OSError):
                await asyncio.sleep(1)

    def _handle(self, msg):
        if msg[0] == "update":
            _, r, nid, bb_delta_b, mod_cids = msg
            self.inbox.setdefault(r, {})[nid] = {
                "bb_delta": np.frombuffer(bb_delta_b, dtype=np.int64),
                "mod_cids": mod_cids,
            }
        elif msg[0] == "want":                                  # Bitswap: serve content
            c = msg[1]
            if self.store.has(c):
                asyncio.create_task(self._bcast(("block", c, self.store.get(c))))
        elif msg[0] == "block":
            c, data = msg[1], msg[2]
            if cid(data) == c:
                self.store.put(data)

    async def _bcast(self, msg):
        for w in list(self.writers):
            try:
                await _send(w, msg)
            except (ConnectionError, OSError):
                self.writers.discard(w)

    async def _await_round(self, r, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            if len(self.inbox.get(r, {})) >= self.n_total - 1:
                return True
            await asyncio.sleep(0.05)
        return False

    async def run(self, rounds):
        import concurrent.futures
        server = await asyncio.start_server(self._peer, self.host, self.port)
        loop = asyncio.get_event_loop()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        async with server:
            await asyncio.sleep(0.5)
            dials = [asyncio.create_task(self._dial(h, p)) for h, p in self.peers]
            # Wait for every peer to connect BEFORE any round, so round-0 updates
            # are delivered (TCP-reliable once connected) — a missed round would
            # make nodes aggregate different sets and fork the root forever.
            cw = time.time() + 90
            while len(self.writers) < self.n_total - 1 and time.time() < cw:
                await asyncio.sleep(0.2)
            for r in range(rounds):
                if self.device == "mps":
                    bb_delta, new_mods = self.train_round(r)     # MPS: main thread
                else:
                    bb_delta, new_mods = await loop.run_in_executor(
                        pool, self.train_round, r)
                mod_cids = {f"mod:{k[0]}:{k[1]}": cid(v.tobytes())
                            for k, v in new_mods.items()}
                update = ("update", r, self.node_id, bb_delta.tobytes(), mod_cids)
                # BARRIER: broadcast our update ONCE unconditionally (else if the
                # peer's update already arrived we'd exit without ever sending ours
                # → deadlock), then rebroadcast until we hold every peer's round-r
                # update, so all nodes aggregate the identical set (a DiLoCo step).
                await self._bcast(update)
                bar = time.time() + 120
                while (len(self.inbox.get(r, {})) < self.n_total - 1
                       and time.time() < bar):
                    await self._bcast(update)
                    await asyncio.sleep(0.3)
                ok = len(self.inbox.get(r, {})) >= self.n_total - 1
                for u in self.inbox.get(r, {}).values():         # keys → tuple form
                    if u["mod_cids"] and isinstance(next(iter(u["mod_cids"])), str):
                        u["mod_cids"] = {("mod", int(k.split(":")[1]), int(k.split(":")[2])): v
                                         for k, v in u["mod_cids"].items()}
                root = self.apply_round(r, bb_delta, new_mods)
                vl = self.val_losses()
                vs = " ".join(f"p{p}:{l:.2f}" for p, l in vl.items())
                print(f"node {self.node_id} round {r} root {root[:12]} "
                      f"peers {'ok' if ok else 'TIMEOUT'} val[{vs}]", flush=True)

            # BEFORE rounds finished we never held a peer's module CONTENT (only
            # CIDs). Now demonstrate serving: fetch every missing module by CID
            # over Bitswap and reconstruct the WHOLE model.
            peer_before = sum(1 for k in self.mod_span if k not in self.owned_mods
                              and self.store.has(self.cids[("mod", *k)]))
            missing = [self.cids[("mod", *k)] for k in self.mod_span
                       if k not in self.owned_mods]
            # Fixed settle window (don't exit early — the peer must stay alive to
            # answer OUR wants even after IT has finished). Re-request each tick.
            settle = time.time() + 8
            while time.time() < settle:
                if self.reconstruct_full()[0] is None:
                    for c in set(missing):
                        await self._bcast(("want", c))
                await asyncio.sleep(0.5)
            full, still_missing = self.reconstruct_full()
            for d in dials:
                d.cancel()
            self._stop.set()

        global_loss = None
        if full is not None:                                 # serve/eval the whole model
            import torch
            set_flat_params(self.model, dequantize(full))
            losses = []
            for pid in range(N_PATHS):
                path = make_path(pid, CFG.n_layer, CFG.n_modules)
                buf = _domain_buf(coarse_route(pid, N_PATHS))
                gen = torch.Generator().manual_seed(7777)
                with torch.no_grad():
                    x, y = _get_batch(buf, 64, CFG.block_size, gen, self.device)
                    _, loss = self.model(x, y, path=path)
                losses.append(loss.item())
            global_loss = float(np.mean(losses))
        print(f"node {self.node_id} FINAL root {manifest_root(self.cids)[:16]} "
              f"owns {sorted(self.owned_paths)} own_modules {len(self.owned_mods)} "
              f"peer_content_held_before_fetch {peer_before} "
              f"reconstructed {'yes' if full is not None else f'missing {len(still_missing)}'} "
              f"global_val {global_loss}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--peers", default="")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--paths", required=True, help="comma-sep path ids this node owns")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    global DEVICE_OVERRIDE
    DEVICE_OVERRIDE = a.device
    peers = [(h, int(p)) for h, p in (x.split(":") for x in a.peers.split(",") if x)]
    owned = [int(x) for x in a.paths.split(",")]
    node = DiPaCoNode(a.id, "0.0.0.0", a.port, peers, a.n, owned,
                      interval=a.interval, t0=a.t0 or None)
    asyncio.run(node.run(a.rounds))


if __name__ == "__main__":
    main()
