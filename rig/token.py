"""The native token — balances as chain state, emissions as block rewards (§9).

Reference implementation (the SPEC for the Rust node). Design commitments:

  * The ledger is CHAIN STATE, exactly like the weights: deterministic integer
    arithmetic, a canonical root committed per block, replayable from genesis.
    No separate contract, no bridge, no trusted mint — the same consensus that
    agrees on the model agrees on who owns what.

  * FAIR LAUNCH (§9.8): the genesis ledger is EMPTY. No premine, no pre-sale.
    Every token in existence is minted by a block reward for verifiable work:
    training deltas, block proposal, or admitted data. The founder's wallet is
    the genesis corpus's data contributor and earns the data share under the
    same rules as any later contributor — a published address, no special path.

  * Emissions halve on a fixed schedule and STOP at the sunset height — the
    §9.3 non-amendable cap. (Mainnet ties the sunset to revenue milestones;
    the reference uses heights so the schedule is testable today.)

Units: integer "grains" (10^9 grains = 1 PALIMPSEST; no floats, ever).
"""

import hashlib
import json
from dataclasses import dataclass, field

from .crypto import Key, frame, verify

GRAIN = 10**9                      # grains per whole token

# Emission schedule (reference constants; mainnet sets these at genesis ceremony)
BASE_REWARD = 50 * GRAIN           # per block at height 1
HALVING_BLOCKS = 100_000           # reward halves every N blocks
SUNSET_HEIGHT = 1_000_000          # hard stop: no emission at/after this height

# Reward split, in basis points (must sum to 10_000)
SHARE_MINERS = 7_000               # split equally among the block's delta miners
SHARE_PROPOSER = 1_000             # the block proposer
SHARE_DATA = 2_000                 # the data contributors whose corpus trained it

# Data lane (rev 3): staked submission + challenge market (§7.2, §9A)
CHALLENGE_WINDOW = 20              # blocks a challenge stays open for votes
PROPOSER_LOOKBACK = 32             # only recent block proposers may vote
GENESIS_DATA_WEIGHT = 1_000_000    # royalty weight of the genesis corpus entry
CHALLENGE_QUORUM = 3               # min affirmative juror votes to uphold a
                                   # challenge — one juror must never be able to
                                   # seize an owner's stake; below quorum the
                                   # challenge is rejected (safe default)


def address(pub_hex: str) -> str:
    """A wallet address: sha256 of the raw pubkey bytes, first 20 bytes, hex."""
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:40]


def emission(height: int) -> int:
    """Deterministic block reward at a height. Halves; hard-stops at sunset."""
    if height < 1 or height >= SUNSET_HEIGHT:
        return 0
    return BASE_REWARD >> ((height - 1) // HALVING_BLOCKS)


@dataclass
class TransferTx:
    """A signed balance transfer. Nonce = sender's transfer count (replay-proof)."""
    from_pub: str                  # sender PUBKEY hex (address derives from it)
    to_addr: str                   # recipient address
    amount: int                    # grains
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"transfer", self.from_pub.encode(), self.to_addr.encode(),
                     str(self.amount).encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "TransferTx":
        assert key.pub == self.from_pub, "signer must match from_pub"
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.from_pub, self.signing_bytes(), self.sig)


@dataclass
class DataSubmitTx:
    """Staked data registration (§7.2): the owner wallet stakes behind a
    content-addressed corpus contribution. data_id = txid. The stake is escrowed
    in the registry entry; the entry earns the block data share weighted by its
    stake (v1 proxy — attribution-weighted royalties replace stake-weighting at
    the TRAK integration milestone) and is what a successful challenge takes."""
    owner_pub: str
    data_hash: str                 # sha256 of the corpus bytes (content address)
    size_bytes: int
    media_type: str                # "text" | "csv" | "image" | … (bytes are bytes)
    stake: int                     # grains escrowed behind this submission
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"data_submit", self.owner_pub.encode(), self.data_hash.encode(),
                     str(self.size_bytes).encode(), self.media_type.encode(),
                     str(self.stake).encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "DataSubmitTx":
        assert key.pub == self.owner_pub
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.owner_pub, self.signing_bytes(), self.sig)


@dataclass
class DataChallengeTx:
    """Stake-vs-stake challenge against a registry entry's validity or
    ownership (§7.2 challenge market). Opens a voting window; upheld → the
    entry is revoked and its stake goes to the challenger; rejected → the
    challenger's stake goes to the entry's owner. Either way, lying costs."""
    challenger_pub: str
    data_id: str
    stake: int
    reason: str                    # "validity" | "ownership"
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"data_challenge", self.challenger_pub.encode(), self.data_id.encode(),
                     str(self.stake).encode(), self.reason.encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "DataChallengeTx":
        assert key.pub == self.challenger_pub
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.challenger_pub, self.signing_bytes(), self.sig)


@dataclass
class DataVoteTx:
    """A vote on an open challenge. Gated: only wallets that PROPOSED one of the
    last PROPOSER_LOOKBACK blocks may vote — juror seats are earned by verifiable
    work, not bought."""
    voter_pub: str
    challenge_id: str
    support: bool                  # True = uphold the challenge
    nonce: int
    sig: bytes = b""

    def signing_bytes(self) -> bytes:
        return frame(b"data_vote", self.voter_pub.encode(), self.challenge_id.encode(),
                     str(int(self.support)).encode(), str(self.nonce).encode())

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "DataVoteTx":
        assert key.pub == self.voter_pub
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.voter_pub, self.signing_bytes(), self.sig)


class TokenLedger:
    """Balances + nonces + the data registry + open challenges — the full token
    state. Every mutation is deterministic integer math."""

    def __init__(self):
        self.balances: dict[str, int] = {}     # address -> grains
        self.nonces: dict[str, int] = {}       # address -> next expected nonce
        # data_id -> {owner, data_hash, size, media_type, stake, weight, status}
        self.registry: dict[str, dict] = {}
        # challenge_id -> {data_id, challenger, stake, reason, expiry,
        #                  votes_for: [addr…], votes_against: [addr…]}
        self.challenges: dict[str, dict] = {}

    def copy(self) -> "TokenLedger":
        led = TokenLedger()
        led.balances = dict(self.balances)
        led.nonces = dict(self.nonces)
        led.registry = {k: dict(v) for k, v in self.registry.items()}
        led.challenges = {k: {**v, "votes_for": list(v["votes_for"]),
                              "votes_against": list(v["votes_against"])}
                          for k, v in self.challenges.items()}
        return led

    def seed_genesis_data(self, owner_addr: str, data_hash: str = "genesis"):
        """The founding corpus as registry entry zero — owned by the founder's
        wallet, earning the data share under the same rules as any entry (its
        weight is a published genesis parameter; stake 0 because no tokens exist
        before block 1 — fair launch has nothing to stake with)."""
        self.registry["genesis"] = {
            "owner": owner_addr, "data_hash": data_hash, "size": 0,
            "media_type": "text", "stake": 0,
            "weight": GENESIS_DATA_WEIGHT, "status": "active"}

    def balance(self, addr: str) -> int:
        return self.balances.get(addr, 0)

    def _credit(self, addr: str, amount: int):
        if amount > 0:
            self.balances[addr] = self.balances.get(addr, 0) + amount

    # ---- block reward ----------------------------------------------------
    def apply_reward(self, height: int, miner_pubs: list[str],
                     proposer_pub: str, data_addrs: list[str] = ()):
        """Mint the block's emission and split it. Integer division truncates;
        the remainder (dust) is deliberately burned — supply never exceeds the
        schedule. Deterministic given identical inputs on every node.

        The data share goes to the REGISTRY: split across active entries in
        proportion to weight (v1: stake-weighted + the genesis entry's published
        weight; TRAK attribution replaces weights at the attribution milestone).
        `data_addrs` remains as a legacy fallback used only when the registry is
        empty (pre-rev-3 chains)."""
        total = emission(height)
        if total == 0:
            return
        miners_pool = total * SHARE_MINERS // 10_000
        proposer_cut = total * SHARE_PROPOSER // 10_000
        data_pool = total * SHARE_DATA // 10_000
        if miner_pubs:
            each = miners_pool // len(miner_pubs)
            for pub in sorted(miner_pubs):                 # canonical order
                self._credit(address(pub), each)
        if proposer_pub and proposer_pub != "genesis":
            self._credit(address(proposer_pub), proposer_cut)
        active = [(did, e) for did, e in sorted(self.registry.items())
                  if e["status"] == "active" and e["weight"] > 0]
        if active:
            wsum = sum(e["weight"] for _, e in active)
            for _, e in active:                            # ∝ weight, dust burned
                self._credit(e["owner"], data_pool * e["weight"] // wsum)
        elif data_addrs:                                   # legacy fallback
            each = data_pool // len(data_addrs)
            for addr in sorted(data_addrs):
                self._credit(addr, each)

    # ---- data lane (rev 3) ----------------------------------------------
    def resolve_expired_challenges(self, height: int):
        """Deterministically settle every challenge whose window has closed
        (processed FIRST in each block, in sorted challenge_id order).
        Upheld (more support than opposition, at least one vote): the entry is
        revoked, its escrowed stake goes to the challenger, the challenger's
        stake returns. Rejected (ties, no votes, or opposition wins): the
        challenger's stake goes to the entry's owner."""
        for cid in sorted(self.challenges):
            ch = self.challenges[cid]
            if ch["expiry"] > height:
                continue
            entry = self.registry.get(ch["data_id"])
            # QUORUM: a challenge is upheld only with a strict majority AND at
            # least CHALLENGE_QUORUM affirmative juror votes. Below quorum (too
            # few disinterested jurors showed up) it is rejected — the challenger
            # cannot seize stake on a thin or single vote.
            upheld = (len(ch["votes_for"]) >= CHALLENGE_QUORUM
                      and len(ch["votes_for"]) > len(ch["votes_against"]))
            if upheld and entry is not None:
                entry["status"] = "revoked"
                self._credit(ch["challenger"], entry["stake"] + ch["stake"])
                entry["stake"] = 0
            elif entry is not None:
                self._credit(entry["owner"], ch["stake"])
            del self.challenges[cid]

    def apply_data_tx(self, tx, height: int, recent_proposers: set[str]) -> bool:
        """Validate + apply one data-lane tx. False = invalid (block invalid)."""
        if not tx.verify():
            return False
        src = address(tx.owner_pub if isinstance(tx, DataSubmitTx)
                      else tx.challenger_pub if isinstance(tx, DataChallengeTx)
                      else tx.voter_pub)
        if tx.nonce != self.nonces.get(src, 0):
            return False
        if isinstance(tx, DataSubmitTx):
            if tx.stake <= 0 or self.balances.get(src, 0) < tx.stake:
                return False
            if tx.txid() in self.registry:
                return False
            self.balances[src] -= tx.stake                 # escrowed in the entry
            self.registry[tx.txid()] = {
                "owner": src, "data_hash": tx.data_hash, "size": tx.size_bytes,
                "media_type": tx.media_type, "stake": tx.stake,
                "weight": tx.stake, "status": "active"}
        elif isinstance(tx, DataChallengeTx):
            entry = self.registry.get(tx.data_id)
            if (entry is None or entry["status"] != "active" or tx.stake <= 0
                    or self.balances.get(src, 0) < tx.stake
                    or any(c["data_id"] == tx.data_id for c in self.challenges.values())):
                return False
            self.balances[src] -= tx.stake
            self.challenges[tx.txid()] = {
                "data_id": tx.data_id, "challenger": src, "stake": tx.stake,
                "reason": tx.reason, "expiry": height + CHALLENGE_WINDOW,
                "votes_for": [], "votes_against": []}
        elif isinstance(tx, DataVoteTx):
            ch = self.challenges.get(tx.challenge_id)
            if (ch is None or tx.voter_pub not in recent_proposers
                    or src in ch["votes_for"] or src in ch["votes_against"]):
                return False
            # DISINTERESTED JURORS ONLY: neither the challenger nor the data
            # owner may vote on their own challenge — both have a direct stake
            # in the outcome. Jurors are disinterested recent proposers.
            if src == ch["challenger"]:
                return False
            entry = self.registry.get(ch["data_id"])
            if entry is not None and src == entry["owner"]:
                return False
            (ch["votes_for"] if tx.support else ch["votes_against"]).append(src)
            ch["votes_for"].sort(); ch["votes_against"].sort()   # canonical
        else:
            return False
        self.nonces[src] = tx.nonce + 1
        return True

    # ---- transfers -------------------------------------------------------
    def apply_transfer(self, tx: TransferTx) -> bool:
        """Validate and apply. Returns False (no-op) on any invalid condition —
        a block containing an invalid transfer is itself invalid upstream."""
        if not tx.verify() or tx.amount <= 0:
            return False
        src = address(tx.from_pub)
        if tx.nonce != self.nonces.get(src, 0):
            return False
        if self.balances.get(src, 0) < tx.amount:
            return False
        self.balances[src] -= tx.amount
        self._credit(tx.to_addr, tx.amount)
        self.nonces[src] = tx.nonce + 1
        return True

    # ---- commitment ------------------------------------------------------
    def apply_transfers(self, transfers: list["TransferTx"]) -> bool:
        """Apply a block's transfers in CANONICAL order. All must apply cleanly;
        returns False (ledger unchanged is NOT guaranteed — copy first) if any
        fails. Validators call this on a copy of the parent ledger."""
        for tx in canonical_transfers(transfers):
            if not self.apply_transfer(tx):
                return False
        return True

    def root(self) -> str:
        """Canonical ledger root: sorted compact JSON over the FULL token state
        (balances, nonces, data registry, open challenges). sort_keys sorts every
        nested dict; vote lists are kept sorted at mutation time."""
        blob = json.dumps({"balances": self.balances, "challenges": self.challenges,
                           "nonces": self.nonces, "registry": self.registry},
                          sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def supply(self) -> int:
        return sum(self.balances.values())


def _tx_sender(tx) -> str:
    if isinstance(tx, TransferTx):
        return address(tx.from_pub)
    if isinstance(tx, DataSubmitTx):
        return address(tx.owner_pub)
    if isinstance(tx, DataChallengeTx):
        return address(tx.challenger_pub)
    return address(tx.voter_pub)


def canonical_transfers(transfers: list[TransferTx]) -> list[TransferTx]:
    """The consensus ordering of a block's transfers: by (sender address, nonce,
    txid). Sender-then-nonce guarantees a sender's nonce sequence applies in
    order; txid breaks any remaining tie deterministically."""
    return sorted(transfers, key=lambda t: (address(t.from_pub), t.nonce, t.txid()))


def canonical_account_txs(data_txs: list, transfers: list[TransferTx]) -> list:
    """The consensus ordering of ALL account transactions in a block (data lane
    + transfer lane merged): (sender address, nonce, txid). One nonce sequence
    per account totally orders everything a wallet does."""
    return sorted(list(data_txs) + list(transfers),
                  key=lambda t: (_tx_sender(t), t.nonce, t.txid()))


def data_root(data_txs: list) -> str:
    """Order-independent commitment to a block's data-lane tx set."""
    joined = "|".join(sorted(t.txid() for t in data_txs))
    return hashlib.sha256(joined.encode()).hexdigest()


def transfer_root(transfers: list[TransferTx]) -> str:
    """Order-independent commitment to a block's transfer set (mirror of
    blockchain.txset_root)."""
    joined = "|".join(sorted(t.txid() for t in transfers))
    return hashlib.sha256(joined.encode()).hexdigest()
