#!/usr/bin/env bash
# Consistent backup + restore of a Sestrian node data-dir.
#
# The store is backup-friendly by construction: the snapshot is written atomically
# (temp + rename) and blocks.jsonl is strictly append-only, so a copy taken while
# the node runs is self-consistent — copy the snapshot first, then the log, and a
# slightly-newer log is fine (fast-boot validates FORWARD from the snapshot and
# self-heals a torn final record). A restored node fast-boots at the snapshot.
#
#   DATA=/data ./backup-restore.sh backup  /backups/pal-$(date +%s).tar.zst
#   DATA=/data ./backup-restore.sh restore /backups/pal-....tar.zst  /data
set -euo pipefail
DATA=${DATA:-/data}
usage() { echo "usage: $0 backup <out.tar.zst> | restore <in.tar.zst> [datadir]"; exit 1; }
cmd=${1:-}; shift || usage

case "$cmd" in
  backup)
    out=${1:?output path required}
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    # snapshot first (the trusted fast-boot anchor), then the append-only log +
    # payloads + uploads + genesis. Missing pieces are tolerated (fresh node).
    for f in genesis.bin snapshot.bin snapshot.json blocks.jsonl; do
      [ -f "$DATA/$f" ] && cp -a "$DATA/$f" "$tmp/" || true
    done
    for d in payloads uploads; do
      [ -d "$DATA/$d" ] && cp -a "$DATA/$d" "$tmp/" || true
    done
    tar -C "$tmp" -cf - . | zstd -q -o "$out"
    echo "backed up $DATA -> $out ($(du -h "$out" | cut -f1))"
    ;;
  restore)
    in=${1:?input path required}; dst=${2:-$DATA}
    mkdir -p "$dst"
    if [ -f "$dst/.lock" ] && command -v fuser >/dev/null && fuser "$dst/.lock" >/dev/null 2>&1; then
      echo "refusing: a node is running on $dst (holds the data-dir lock)"; exit 1
    fi
    zstd -dq "$in" -c | tar -C "$dst" -x
    echo "restored $in -> $dst (start the node; it fast-boots from the snapshot)"
    ;;
  *) usage;;
esac
