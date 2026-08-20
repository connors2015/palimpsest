#!/bin/bash
# 3-node local Rust devnet: gossip over libp2p (QUIC+Noise), full rev-3 validation.
set -e
cd "$(dirname "$0")/../node"
cargo build --release
B=target/release/palimpsest-node
S=${1:-30}
$B --id 0 --n 3 --port 7700 --produce --seconds $S --interval 2 &
$B --id 1 --n 3 --port 7701 --peers /ip4/127.0.0.1/udp/7700/quic-v1 --produce --seconds $S --interval 2 &
$B --id 2 --n 3 --port 7702 --peers /ip4/127.0.0.1/udp/7700/quic-v1,/ip4/127.0.0.1/udp/7701/quic-v1 --produce --seconds $S --interval 2 &
wait
