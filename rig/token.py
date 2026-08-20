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

from .crypto import Key, verify

GRAIN = 10**9                      # grains per whole token

# Emission schedule (reference constants; mainnet sets these at genesis ceremony)
BASE_REWARD = 50 * GRAIN           # per block at height 1
HALVING_BLOCKS = 100_000           # reward halves every N blocks
SUNSET_HEIGHT = 1_000_000          # hard stop: no emission at/after this height

# Reward split, in basis points (must sum to 10_000)
SHARE_MINERS = 7_000               # split equally among the block's delta miners
SHARE_PROPOSER = 1_000             # the block proposer
SHARE_DATA = 2_000                 # the data contributors whose corpus trained it


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
        return (f"transfer|{self.from_pub}|{self.to_addr}|"
                f"{self.amount}|{self.nonce}").encode()

    def txid(self) -> str:
        return hashlib.sha256(self.signing_bytes()).hexdigest()

    def signed(self, key: Key) -> "TransferTx":
        assert key.pub == self.from_pub, "signer must match from_pub"
        self.sig = key.sign(self.signing_bytes())
        return self

    def verify(self) -> bool:
        return verify(self.from_pub, self.signing_bytes(), self.sig)


class TokenLedger:
    """Balances + nonces. Every mutation is deterministic integer math."""

    def __init__(self):
        self.balances: dict[str, int] = {}     # address -> grains
        self.nonces: dict[str, int] = {}       # address -> next expected nonce

    def copy(self) -> "TokenLedger":
        led = TokenLedger()
        led.balances = dict(self.balances)
        led.nonces = dict(self.nonces)
        return led

    def balance(self, addr: str) -> int:
        return self.balances.get(addr, 0)

    def _credit(self, addr: str, amount: int):
        if amount > 0:
            self.balances[addr] = self.balances.get(addr, 0) + amount

    # ---- block reward ----------------------------------------------------
    def apply_reward(self, height: int, miner_pubs: list[str],
                     proposer_pub: str, data_addrs: list[str]):
        """Mint the block's emission and split it. Integer division truncates;
        the remainder (dust) is deliberately burned — supply never exceeds the
        schedule. Deterministic given identical inputs on every node."""
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
        if data_addrs:
            each = data_pool // len(data_addrs)
            for addr in sorted(data_addrs):
                self._credit(addr, each)

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
    def root(self) -> str:
        """Canonical ledger root: sorted JSON over balances + nonces."""
        blob = json.dumps({"balances": dict(sorted(self.balances.items())),
                           "nonces": dict(sorted(self.nonces.items()))},
                          sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def supply(self) -> int:
        return sum(self.balances.values())
