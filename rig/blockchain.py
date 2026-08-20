"""Bitcoin-style blocks: hash-linked headers, independent validation, fork choice.

The rig's `rig/chain.py` is a linear list an authority appends to. A real chain
must let any node, holding no special trust, (1) validate a block from first
principles and (2) choose between competing histories. This module adds those:

  * a **header** committing prev_hash, the weights-state root, the tx-set root,
    height, and cumulative work — hashed into the block id (Bitcoin's header);
  * **full validation** of a block against its parent state: every tx signature
    checks, the state transition reproduces the committed root, the tx-set root
    matches (§3.4, §5);
  * a **BlockTree** with **heaviest-valid-chain fork choice** — the same
    Nakamoto rule that lets Bitcoin nodes agree without a coordinator.

Model weights are the state, exactly as before; this layer is about *who gets to
say what the history is* — and the answer is now "the heaviest valid chain",
not "the coordinator".
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .chain import dequantize, quantize, state_root, trimmed_mean_int
from .crypto import BackpropTx
from .token import TokenLedger, canonical_transfers, transfer_root as xfer_root


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def txset_root(txs: list[BackpropTx]) -> str:
    """Order-independent commitment to the accepted tx set (§3.2)."""
    return _sha(("|".join(sorted(t.txid() for t in txs))).encode())


@dataclass
class Header:
    height: int
    prev_hash: str
    state_root: str            # Merkle/hash of the weights AFTER this block
    txset_root: str
    n_txs: int
    work: int                  # per-block work (score-weighted, see BlockTree)
    proposer: str              # pubkey of the block proposer
    # the TRANSFER LANE (protocol rev 2): the token ledger is consensus state
    transfer_root: str = ""    # order-independent commitment to the transfer set
    ledger_root: str = ""      # token-ledger root AFTER this block (rewards+transfers)

    def block_hash(self) -> str:
        return _sha(json.dumps(self.__dict__, sort_keys=True).encode())


@dataclass
class Block:
    header: Header
    txs: list                  # list[BackpropTx]
    bodies: dict               # da_pointer -> int64 delta array (carried for replay)
    transfers: list = field(default_factory=list)   # list[TransferTx]

    @property
    def hash(self) -> str:
        return self.header.block_hash()


class ValidationError(Exception):
    pass


def apply_ledger(parent_ledger: TokenLedger, block: Block,
                 data_contributor: str | None) -> TokenLedger:
    """The deterministic token-state transition for one block: mint the block
    reward (miners/proposer/data shares), then apply the transfers in canonical
    order. Raises if any transfer is invalid — a block carrying one is invalid."""
    led = parent_ledger.copy()
    led.apply_reward(block.header.height,
                     miner_pubs=[tx.miner for tx in block.txs],
                     proposer_pub=block.header.proposer,
                     data_addrs=[data_contributor] if data_contributor else [])
    for tx in canonical_transfers(block.transfers):
        if not tx.verify():
            raise ValidationError(f"bad signature on transfer {tx.txid()[:8]}")
        if not led.apply_transfer(tx):
            raise ValidationError(f"invalid transfer {tx.txid()[:8]} (nonce/balance)")
    return led


def validate_block(block: Block, parent_w_int: np.ndarray,
                   parent_ledger: TokenLedger | None = None,
                   data_contributor: str | None = None):
    """Validate a block from first principles against its parent state.

    Returns (post-state weights, post-ledger) if valid; raises ValidationError
    otherwise. Any node can run this — no trust in the proposer required (§5).
    `parent_ledger=None` skips ledger validation (legacy callers only; the live
    protocol always validates the ledger)."""
    h = block.header
    # 1. every tx is well-formed and correctly signed
    for tx in block.txs:
        if not tx.verify():
            raise ValidationError(f"bad signature on tx {tx.txid()[:8]}")
        if tx.base_height != h.height - 1:
            raise ValidationError("tx base_height does not match parent")
        body = block.bodies.get(tx.da_pointer)
        if body is None:
            raise ValidationError(f"missing DA body for {tx.da_pointer}")
        if _sha(body.tobytes()) != tx.delta_hash:
            raise ValidationError("delta body hash mismatch (DA withholding/forgery)")
    # 2. tx-set root matches
    if txset_root(block.txs) != h.txset_root:
        raise ValidationError("txset_root mismatch")
    # 3. the state transition reproduces the committed root (deterministic, §3.4)
    deltas = [block.bodies[tx.da_pointer] for tx in block.txs]
    w = parent_w_int + trimmed_mean_int(deltas) if deltas else parent_w_int.copy()
    if state_root(w) != h.state_root:
        raise ValidationError("state_root does not reproduce from txs")
    # 4. the TRANSFER LANE: transfer-set root + full token-ledger transition
    led = None
    if parent_ledger is not None:
        if xfer_root(block.transfers) != h.transfer_root:
            raise ValidationError("transfer_root mismatch")
        led = apply_ledger(parent_ledger, block, data_contributor)
        if led.root() != h.ledger_root:
            raise ValidationError("ledger_root does not reproduce from block")
    return w, led


class BlockTree:
    """All known blocks, with heaviest-valid-chain selection (Nakamoto fork choice)."""

    def __init__(self, genesis_w_int: np.ndarray, prune_depth: int | None = None,
                 data_contributor: str | None = None):
        self.genesis_w = genesis_w_int.copy()
        gh = Header(0, "0" * 64, state_root(genesis_w_int), _sha(b""), 0, 0, "genesis")
        self.genesis = Block(gh, [], {})
        self.blocks = {self.genesis.hash: self.genesis}
        self.state = {self.genesis.hash: genesis_w_int.copy()}     # per-block post-state
        # the token ledger is chain state too: EMPTY at genesis (fair launch),
        # advanced deterministically by every block's rewards + transfers.
        # data_contributor is a GENESIS PARAMETER — identical on every node.
        self.ledger = {self.genesis.hash: TokenLedger()}
        self.data_contributor = data_contributor
        self.cum_work = {self.genesis.hash: 0}
        self.head = self.genesis.hash
        # prune_depth: keep full state + bodies only within this many blocks of the
        # head (plus genesis). Essential at real-model scale — an 86M state is
        # ~0.7GB, so retaining one per block OOMs in minutes. Headers, txs and
        # cum_work are kept forever (fork choice needs them); a reorg deeper than
        # prune_depth would need replay from genesis (Bitcoin prunes the same way).
        self.prune_depth = prune_depth

    def add_block(self, block: Block) -> bool:
        """Validate and attach a block. Returns True if it became the new head."""
        if block.hash in self.blocks:
            return False
        parent = block.header.prev_hash
        if parent not in self.blocks:
            raise ValidationError("orphan: parent unknown")
        # ledger validation only when the block commits one (rev-2 blocks always
        # do; legacy rev-1 blocks carry ledger_root="" and skip it)
        parent_led = self.ledger.get(parent) if block.header.ledger_root else None
        w, led = validate_block(block, self.state[parent], parent_led,
                                self.data_contributor)             # may raise
        self.blocks[block.hash] = block
        self.state[block.hash] = w
        if led is not None:
            self.ledger[block.hash] = led
        else:                                                      # legacy: rewards only
            self.ledger[block.hash] = apply_ledger(
                self.ledger[parent], block, self.data_contributor)
        self.cum_work[block.hash] = self.cum_work[parent] + max(1, block.header.work)
        # heaviest chain wins; ties broken by lexicographically smaller hash
        if (self.cum_work[block.hash] > self.cum_work[self.head] or
                (self.cum_work[block.hash] == self.cum_work[self.head]
                 and block.hash < self.head)):
            self.head = block.hash
            self._prune_deep()
            return True
        self._prune_deep()
        return False

    def _prune_deep(self):
        """Drop heavy per-block data (state vector, delta bodies) for blocks more
        than prune_depth below the head. Headers/txs/cum_work stay."""
        if self.prune_depth is None:
            return
        floor = self.blocks[self.head].header.height - self.prune_depth
        for bh, b in self.blocks.items():
            if bh == self.genesis.hash or b.header.height >= floor:
                continue
            self.state.pop(bh, None)
            if b.bodies:
                b.bodies = {}

    def head_state(self) -> np.ndarray:
        return self.state[self.head]

    def head_ledger(self) -> TokenLedger:
        return self.ledger[self.head]

    def chain_from_genesis(self, tip: str | None = None) -> list:
        tip = tip or self.head
        out = []
        while tip != self.genesis.hash:
            b = self.blocks[tip]
            out.append(b)
            tip = b.header.prev_hash
        return list(reversed(out))

    def replay_head(self) -> np.ndarray:
        """Independently reconstruct head state from genesis + block bodies (§3.5)."""
        w = self.genesis_w.copy()
        for b in self.chain_from_genesis():
            deltas = [b.bodies[tx.da_pointer] for tx in b.txs]
            if deltas:
                w = w + trimmed_mean_int(deltas)
        return w


def build_block(tree: BlockTree, parent_hash: str, accepted: list, bodies: dict,
                works: dict, proposer: str, transfers: list | None = None) -> Block:
    """Assemble a valid block extending `parent_hash` from accepted txs (and,
    rev 2, the transfer lane: the header commits the transfer set and the
    post-block ledger root).

    `works[txid]` is the delta's score (its contribution to block weight)."""
    parent_w = tree.state[parent_hash]
    transfers = canonical_transfers(transfers or [])
    deltas = [bodies[tx.da_pointer] for tx in accepted]
    w = parent_w + trimmed_mean_int(deltas) if deltas else parent_w.copy()
    total_work = int(sum(max(0.0, works.get(tx.txid(), 0.0)) for tx in accepted) * 1000)
    header = Header(
        height=tree.blocks[parent_hash].header.height + 1,
        prev_hash=parent_hash, state_root=state_root(w),
        txset_root=txset_root(accepted), n_txs=len(accepted),
        work=total_work, proposer=proposer)
    block = Block(header, accepted,
                  {t.da_pointer: bodies[t.da_pointer] for t in accepted}, transfers)
    # commit the token transition (rewards + transfers) into the header
    header.transfer_root = xfer_root(transfers)
    header.ledger_root = apply_ledger(tree.ledger[parent_hash], block,
                                      tree.data_contributor).root()
    return block
