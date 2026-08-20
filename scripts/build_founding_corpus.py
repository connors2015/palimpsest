"""Build the founding corpus — the ceremony artifact the genesis registry entry
commits to (docs/genesis-ceremony.md §1, §4).

Composition decision (recorded): PUBLIC DOMAIN ONLY. English-language Project
Gutenberg books (via the ungated `sedthh/gutenberg_english` parquet mirror,
which strips Gutenberg's own headers/footers — the underlying works are public
domain). No web crawl, no share-alike, no gated sources: the founding entry
earns the founder's data share forever and is challengeable for ownership by
design, so its provenance must be bulletproof. Code / Wikipedia / web-scale
text enter later through OTHER contributors' staked submissions and the
campaign track — that is the data economy working, not a gap.

Outputs:
  founding_corpus.txt      the bytes the model eats (documents joined by \x1e)
  founding_manifest.json   sources + per-shard sha256 + founding_corpus_hash

  python -m scripts.build_founding_corpus --out-dir corpus [--shards 37]
"""

import argparse
import hashlib
import json
import os
import urllib.request

BASE = ("https://huggingface.co/datasets/sedthh/gutenberg_english/resolve/main")
API = "https://huggingface.co/api/datasets/sedthh/gutenberg_english"
SEP = b"\x1e\n"                      # record separator byte between documents


def shard_list():
    with urllib.request.urlopen(API, timeout=30) as r:
        meta = json.load(r)
    return sorted(s["rfilename"] for s in meta["siblings"]
                  if s["rfilename"].endswith(".parquet"))


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="corpus")
    ap.add_argument("--shards", type=int, default=0)     # 0 = all
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    shards = shard_list()
    if a.shards:
        shards = shards[: a.shards]
    out_path = os.path.join(a.out_dir, "founding_corpus.txt")
    manifest = {"license": "public domain (Project Gutenberg, English)",
                "source_dataset": "sedthh/gutenberg_english",
                "separator": "0x1e0x0a", "shards": [], "documents": 0}
    corpus_hash = hashlib.sha256()
    total = 0
    with open(out_path, "wb") as out:
        for i, name in enumerate(shards):
            url = f"{BASE}/{name}"
            tmp = os.path.join(a.out_dir, "_shard.parquet")
            urllib.request.urlretrieve(url, tmp)
            with open(tmp, "rb") as f:
                shard_sha = hashlib.sha256(f.read()).hexdigest()
            table = pq.read_table(tmp)
            col = next(c for c in table.column_names if c.lower() == "text")
            n_docs = 0
            for chunk in table.column(col).to_pylist():
                if not chunk:
                    continue
                raw = chunk.encode("utf-8", errors="ignore") + SEP
                out.write(raw)
                corpus_hash.update(raw)
                total += len(raw)
                n_docs += 1
            os.remove(tmp)
            manifest["shards"].append({"file": name, "sha256": shard_sha,
                                       "documents": n_docs})
            manifest["documents"] += n_docs
            print(f"[{i+1}/{len(shards)}] {name}: {n_docs} docs, "
                  f"total {total/1e9:.2f} GB", flush=True)
    manifest["founding_corpus_bytes"] = total
    manifest["founding_corpus_hash"] = corpus_hash.hexdigest()
    with open(os.path.join(a.out_dir, "founding_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nfounding_corpus_hash: {manifest['founding_corpus_hash']}")
    print(f"bytes: {total:,}  documents: {manifest['documents']:,}")


if __name__ == "__main__":
    main()
