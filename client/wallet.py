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
    ap.add_argument("cmd", choices=["new", "show", "balance", "send"])
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--node", default="http://localhost:8090")
    ap.add_argument("--to", default=None)
    ap.add_argument("--amount", type=float, default=None)   # whole tokens
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


if __name__ == "__main__":
    main()
