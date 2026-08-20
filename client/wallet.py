"""Your wallet — an Ed25519 keypair on YOUR machine, never anywhere else.

The secret key lives in a mode-0600 file under ~/.palimpsest/ (or --path). It is
never transmitted, never logged, and must NEVER enter a git repository. The
address (sha256 of the pubkey, 20 bytes) is what the chain knows you as: block
rewards for your training deltas, data royalties, and transfers all land there.

  python -m client.wallet new                  # create (refuses to overwrite)
  python -m client.wallet show                 # address + pubkey (never the secret)
  python -m client.wallet balance --node http://localhost:8090
  python -m client.wallet send --to <addr> --amount 1.5 --node http://localhost:8090

For the REAL genesis ceremony: generate the founding wallet fresh, offline, on a
machine you trust, and back up the file — the key IS the wallet.
"""

import argparse
import json
import os
import stat
import urllib.request

from rig.crypto import Key
from rig.token import GRAIN, TransferTx, address

DEFAULT_DIR = os.path.expanduser("~/.palimpsest")
DEFAULT_PATH = os.path.join(DEFAULT_DIR, "wallet.json")


def create(path: str) -> dict:
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing wallet: {path}")
    key = Key.generate()                          # 32 random bytes from os.urandom
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"sk": key.sk.hex(), "pub": key.pub, "address": address(key.pub)}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, indent=1)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)   # 0600 — owner only
    return rec


def load(path: str) -> tuple[Key, dict]:
    with open(path) as f:
        rec = json.load(f)
    key = Key.generate(bytes.fromhex(rec["sk"]))
    assert key.pub == rec["pub"], "wallet file corrupt (pub mismatch)"
    return key, rec


def _get(node: str, route: str):
    with urllib.request.urlopen(f"{node}{route}", timeout=10) as r:
        return json.loads(r.read())


def _post(node: str, route: str, payload: dict):
    req = urllib.request.Request(
        f"{node}{route}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["new", "show", "balance", "send",
                                    "submit-data", "challenge", "vote", "registry"])
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--node", default="http://localhost:8090")
    ap.add_argument("--to", default=None)
    ap.add_argument("--amount", type=float, default=None)   # whole tokens
    ap.add_argument("--file", default=None)                 # submit-data: corpus file
    ap.add_argument("--media-type", default="text")
    ap.add_argument("--stake", type=float, default=None)    # whole tokens
    ap.add_argument("--data-id", default=None)              # challenge target
    ap.add_argument("--reason", default="validity")         # validity | ownership
    ap.add_argument("--challenge-id", default=None)         # vote target
    ap.add_argument("--support", action="store_true")       # vote: uphold challenge
    a = ap.parse_args()

    if a.cmd == "new":
        rec = create(a.path)
        print(f"wallet created: {a.path}  (mode 0600 — BACK THIS FILE UP)")
        print(f"address: {rec['address']}")
        print(f"pubkey:  {rec['pub']}")
        return

    key, rec = load(a.path)
    if a.cmd == "show":
        print(f"address: {rec['address']}")
        print(f"pubkey:  {rec['pub']}")
    elif a.cmd == "balance":
        out = _get(a.node, f"/balance?addr={rec['address']}")
        print(f"address: {rec['address']}")
        print(f"balance: {out['grains'] / GRAIN:.9f} PALIMPSEST "
              f"({out['grains']} grains) @ block {out['height']}")
    elif a.cmd == "send":
        if not a.to or a.amount is None:
            raise SystemExit("send needs --to and --amount")
        info = _get(a.node, f"/balance?addr={rec['address']}")
        tx = TransferTx(from_pub=rec["pub"], to_addr=a.to,
                        amount=int(round(a.amount * GRAIN)),
                        nonce=info.get("nonce", 0)).signed(key)
        out = _post(a.node, "/transfer", {
            "from_pub": tx.from_pub, "to_addr": tx.to_addr,
            "amount": tx.amount, "nonce": tx.nonce, "sig": tx.sig.hex()})
        print(f"submitted {a.amount} to {a.to}: {out}")
    elif a.cmd == "submit-data":
        if not a.file or a.stake is None:
            raise SystemExit("submit-data needs --file and --stake")
        import hashlib
        from rig.token import DataSubmitTx
        with open(a.file, "rb") as f:
            blob = f.read()
        info = _get(a.node, f"/balance?addr={rec['address']}")
        tx = DataSubmitTx(owner_pub=rec["pub"],
                          data_hash=hashlib.sha256(blob).hexdigest(),
                          size_bytes=len(blob), media_type=a.media_type,
                          stake=int(round(a.stake * GRAIN)),
                          nonce=info.get("nonce", 0)).signed(key)
        out = _post(a.node, "/data/submit", {
            "owner_pub": tx.owner_pub, "data_hash": tx.data_hash,
            "size_bytes": tx.size_bytes, "media_type": tx.media_type,
            "stake": tx.stake, "nonce": tx.nonce, "sig": tx.sig.hex()})
        print(f"data submitted ({len(blob)} bytes, stake {a.stake}): {out}")
    elif a.cmd == "challenge":
        if not a.data_id or a.stake is None:
            raise SystemExit("challenge needs --data-id and --stake")
        from rig.token import DataChallengeTx
        info = _get(a.node, f"/balance?addr={rec['address']}")
        tx = DataChallengeTx(challenger_pub=rec["pub"], data_id=a.data_id,
                             stake=int(round(a.stake * GRAIN)), reason=a.reason,
                             nonce=info.get("nonce", 0)).signed(key)
        out = _post(a.node, "/data/challenge", {
            "challenger_pub": tx.challenger_pub, "data_id": tx.data_id,
            "stake": tx.stake, "reason": tx.reason, "nonce": tx.nonce,
            "sig": tx.sig.hex()})
        print(f"challenge filed against {a.data_id[:12]} ({a.reason}): {out}")
    elif a.cmd == "vote":
        if not a.challenge_id:
            raise SystemExit("vote needs --challenge-id (and --support to uphold)")
        from rig.token import DataVoteTx
        info = _get(a.node, f"/balance?addr={rec['address']}")
        tx = DataVoteTx(voter_pub=rec["pub"], challenge_id=a.challenge_id,
                        support=a.support, nonce=info.get("nonce", 0)).signed(key)
        out = _post(a.node, "/data/vote", {
            "voter_pub": tx.voter_pub, "challenge_id": tx.challenge_id,
            "support": tx.support, "nonce": tx.nonce, "sig": tx.sig.hex()})
        print(f"vote {'FOR' if a.support else 'AGAINST'} challenge: {out}")
    elif a.cmd == "registry":
        out = _get(a.node, "/data/registry")
        for did, e in out["registry"].items():
            print(f"{did[:16]}  {e['status']:8} {e['media_type']:6} "
                  f"{e['size']:>10}B  stake {e['stake']/GRAIN:.2f}  owner {e['owner'][:12]}")
        for cid, c in out["challenges"].items():
            print(f"⚔ {cid[:16]} vs {c['data_id'][:12]} ({c['reason']}) "
                  f"expires h{c['expiry']} votes {len(c['votes_for'])}:{len(c['votes_against'])}")


if __name__ == "__main__":
    main()
