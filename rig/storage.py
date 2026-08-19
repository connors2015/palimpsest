"""On-disk chain persistence and fast-sync (WHITEPAPER §3.5).

Layout under a chain directory:
    genesis.npy            int64 genesis weights
    blocks.jsonl           one line per block: height, root, miner_ids,
                           delta_hashes, and DA pointers (delta_NNNN_j.npy)
    deltas/delta_<h>_<j>.npy   the delta bodies (the "DA layer", §3.3)
    checkpoints/ckpt_<h>.npy   full weights every K blocks

Two sync tiers (§3.5):
  * full replay  — genesis + every delta body  -> whole history, bit-exact
  * fast sync    — latest checkpoint + later deltas -> current state in O(K)

A node can stop and restart from disk and land on the identical state root.
"""

import hashlib
import json
import os

import numpy as np

from .chain import Chain, state_root, trimmed_mean_int


class ChainStore:
    def __init__(self, path: str, checkpoint_every: int = 10):
        self.path = path
        self.k = checkpoint_every
        os.makedirs(os.path.join(path, "deltas"), exist_ok=True)
        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)

    # -- writing -----------------------------------------------------------
    def init_genesis(self, w0_int: np.ndarray) -> None:
        np.save(os.path.join(self.path, "genesis.npy"), w0_int)
        open(os.path.join(self.path, "blocks.jsonl"), "w").close()

    def append_block(self, height, deltas_int, miner_ids, root, w_int) -> None:
        hashes = []
        for j, d in enumerate(deltas_int):
            np.save(os.path.join(self.path, "deltas", f"delta_{height}_{j}.npy"), d)
            hashes.append(hashlib.sha256(d.tobytes()).hexdigest())
        rec = dict(height=height, root=root, miner_ids=list(miner_ids),
                   delta_hashes=hashes, n=len(deltas_int))
        with open(os.path.join(self.path, "blocks.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if height % self.k == 0:
            np.save(os.path.join(self.path, "checkpoints", f"ckpt_{height}.npy"), w_int)

    def persist_chain(self, chain: Chain) -> None:
        """Write an in-memory Chain to disk from scratch (used by tests/tools)."""
        self.init_genesis(chain.genesis_int)
        w = chain.genesis_int.copy()
        for b in chain.blocks:
            if b.deltas_int:
                w = w + trimmed_mean_int(b.deltas_int)
            self.append_block(b.height, b.deltas_int, b.miner_ids, b.root, w)

    # -- reading -----------------------------------------------------------
    def _read_blocks(self):
        recs = []
        with open(os.path.join(self.path, "blocks.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs

    def _load_deltas(self, height, n):
        return [np.load(os.path.join(self.path, "deltas", f"delta_{height}_{j}.npy"))
                for j in range(n)]

    def height(self) -> int:
        return len(self._read_blocks())

    def full_replay(self) -> np.ndarray:
        """Reconstruct current state from genesis + every delta (§3.5)."""
        w = np.load(os.path.join(self.path, "genesis.npy"))
        for rec in self._read_blocks():
            if rec["n"]:
                w = w + trimmed_mean_int(self._load_deltas(rec["height"], rec["n"]))
        return w

    def fast_sync(self) -> np.ndarray:
        """Latest checkpoint + subsequent deltas -> current state (§3.5)."""
        recs = self._read_blocks()
        ckpts = [int(fn[5:-4]) for fn in os.listdir(
            os.path.join(self.path, "checkpoints")) if fn.startswith("ckpt_")]
        if not ckpts:
            return self.full_replay()
        h0 = max(ckpts)
        w = np.load(os.path.join(self.path, "checkpoints", f"ckpt_{h0}.npy"))
        for rec in recs:
            if rec["height"] > h0 and rec["n"]:
                w = w + trimmed_mean_int(self._load_deltas(rec["height"], rec["n"]))
        return w

    def verify(self) -> bool:
        """Both sync tiers must land on the last block's committed root."""
        recs = self._read_blocks()
        if not recs:
            return True
        target = recs[-1]["root"]
        return (state_root(self.full_replay()) == target
                and state_root(self.fast_sync()) == target)

    def load_chain(self) -> Chain:
        """Rebuild an in-memory Chain (blocks + deltas) from disk."""
        from .chain import Block
        w0 = np.load(os.path.join(self.path, "genesis.npy"))
        chain = Chain(w0)
        for rec in self._read_blocks():
            deltas = self._load_deltas(rec["height"], rec["n"])
            chain.apply_block(deltas, rec["miner_ids"])
        return chain
