#!/bin/bash
# Launch (or relaunch) a mining node + its PyTorch trainer bridge on THIS
# machine, reliably: kills old instances by exact binary path, starts fresh
# under setsid, verifies both actually came up, prints status. Config via env:
#
#   WALLET=~/.palimpsest/wallet.json GENESIS=genesis.bin DATA=corpus.txt \
#   PORT=7900 API=8090 BRIDGE=7999 INNER=120 BATCH=24 DEVICE=cuda \
#   PEERS=/ip4/…/udp/9800/quic-v1 scripts/run-miner.sh
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
BIN=$ROOT/node/target/release/palimpsest-node
FOUNDER=${FOUNDER:-3432d48fd6878b4f2e7a1e40cc15e112c512fae7}
WALLET=${WALLET:-$HOME/.palimpsest/wallet.json}
GENESIS=${GENESIS:-$ROOT/genesis.bin}
DATA=${DATA:-$ROOT/corpus/founding_corpus.txt}
DATADIR=${DATADIR:-$ROOT/nodedata}
PORT=${PORT:-7900}; API=${API:-8090}; BRIDGE=${BRIDGE:-7999}
INNER=${INNER:-120}; BATCH=${BATCH:-24}; DEVICE=${DEVICE:-cuda}
PEERS=${PEERS:-}
PY=${PY:-$ROOT/.venv/bin/python}

pkill -f "$BIN" 2>/dev/null
pkill -f "client\.miner_bridge.*$BRIDGE" 2>/dev/null
sleep 1

setsid nohup "$BIN" \
  --data-dir "$DATADIR" --wallet "$WALLET" --genesis-file "$GENESIS" \
  --port "$PORT" --api-port "$API" --bridge-port "$BRIDGE" \
  --produce --interval 60 --peers "$PEERS" \
  --data-contributor "$FOUNDER" \
  > /tmp/palimpsest-node.log 2>&1 < /dev/null &
sleep 4
if ! curl -s -m 3 "http://127.0.0.1:$API/status" > /dev/null; then
    echo "NODE FAILED:"; tail -5 /tmp/palimpsest-node.log; exit 1
fi

setsid nohup "$PY" -m client.miner_bridge \
  --node-port "$BRIDGE" --model small --data "$DATA" \
  --inner "$INNER" --batch "$BATCH" --device "$DEVICE" \
  > /tmp/palimpsest-bridge.log 2>&1 < /dev/null &
sleep 3
if ! pgrep -f "client\.miner_bridge.*$BRIDGE" > /dev/null; then
    echo "BRIDGE FAILED:"; tail -5 /tmp/palimpsest-bridge.log; exit 1
fi

echo "MINER UP:"
curl -s -m 3 "http://127.0.0.1:$API/status"; echo
