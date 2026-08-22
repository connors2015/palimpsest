#!/bin/bash
# Provision a fresh Ubuntu 24.04 VPS into a Sestrian public seed node
# (bootstrap peer + circuit-relay v2 + HTTP API), from zero, idempotently.
#
#   scp deploy/provision-seed.sh root@<ip>:/root/ && ssh root@<ip> 'bash provision-seed.sh'
#
# The repo is private pre-launch: the script generates a machine deploy key and
# prints its PUBLIC half, then waits — add it at github.com/<repo>/settings/keys
# (read-only) and re-run; both steps are safe to repeat. After the repo goes
# public this step disappears (script falls back to https clone).
#
# What you get:
#   * sestrian-node (release build) under systemd, Restart=always
#   * --relay-server, public --external-address auto-detected
#   * genesis.bin materialized in-place from the published seed and VERIFIED
#   * ufw: 22/tcp, 9800/tcp+udp (gossip+relay), 8080/tcp (API)
set -euo pipefail

REPO_SSH="git@github.com:connors2015/sestrian.git"
REPO_HTTPS="https://github.com/connors2015/sestrian.git"
GENESIS_SEED=1337
GENESIS_MODEL=small
DATA_CONTRIBUTOR="3432d48fd6878b4f2e7a1e40cc15e112c512fae7"
NODE_PORT=9800
API_PORT=8080
APP=/opt/sestrian

echo "== packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -qy git build-essential pkg-config curl ufw python3-venv python3-pip

echo "== rust toolchain =="
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "== source =="
if [ ! -d $APP/.git ]; then
    if git ls-remote -q "$REPO_HTTPS" >/dev/null 2>&1; then
        git clone --depth 1 "$REPO_HTTPS" $APP        # repo is public
    else
        # private repo: machine deploy key
        if [ ! -f /root/.ssh/sestrian_deploy ]; then
            ssh-keygen -t ed25519 -N "" -C "sestrian-seed-$(hostname)" \
                -f /root/.ssh/sestrian_deploy -q
        fi
        ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
        export GIT_SSH_COMMAND="ssh -i /root/.ssh/sestrian_deploy"
        if ! git clone --depth 1 "$REPO_SSH" $APP 2>/dev/null; then
            echo ""
            echo "############################################################"
            echo "# Add this READ-ONLY deploy key to the GitHub repo, then   #"
            echo "# re-run this script:                                      #"
            echo "############################################################"
            cat /root/.ssh/sestrian_deploy.pub
            exit 1
        fi
    fi
else
    ( cd $APP && GIT_SSH_COMMAND="ssh -i /root/.ssh/sestrian_deploy" git pull -q || true )
fi

echo "== build node =="
( cd $APP/node && cargo build --release )

echo "== genesis artifact =="
if [ ! -f $APP/genesis.bin ]; then
    python3 -m venv $APP/.venv
    $APP/.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
    $APP/.venv/bin/pip install -q numpy pynacl
    ( cd $APP && .venv/bin/python -m client.make_genesis \
        --model $GENESIS_MODEL --seed $GENESIS_SEED --out genesis.bin )
fi

echo "== identity =="
if [ ! -f $APP/seed.key ]; then
    head -c 32 /dev/urandom | xxd -p -c 64 > $APP/seed.key
    chmod 600 $APP/seed.key
fi

PUBLIC_IP=$(curl -4s https://ifconfig.me || curl -4s https://api.ipify.org)
echo "public ip: $PUBLIC_IP"

echo "== systemd =="
cat > /etc/systemd/system/sestrian-seed.service <<EOF
[Unit]
Description=Sestrian seed node (bootstrap + relay)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$APP/node/target/release/sestrian-node \\
  --data-dir $APP/data \\
  --key-file $APP/seed.key \\
  --genesis-file $APP/genesis.bin \\
  --port $NODE_PORT --api-port $API_PORT --bridge-port 7999 \\
  --relay-server \\
  --prune-depth 2 \\
  --external-address /ip4/$PUBLIC_IP/udp/$NODE_PORT/quic-v1 \\
  --data-contributor $DATA_CONTRIBUTOR
Restart=always
RestartSec=5
LimitNOFILE=65536
WorkingDirectory=$APP

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now sestrian-seed

echo "== firewall =="
ufw allow 22/tcp >/dev/null
ufw allow $NODE_PORT/tcp >/dev/null
ufw allow $NODE_PORT/udp >/dev/null
ufw allow $API_PORT/tcp >/dev/null
ufw --force enable >/dev/null

sleep 4
echo "== verify =="
systemctl --no-pager -l status sestrian-seed | head -6
curl -s -m 5 http://127.0.0.1:$API_PORT/status && echo
echo ""
echo "SEED LIVE. Bootstrap multiaddr for node operators:"
echo "  /ip4/$PUBLIC_IP/udp/$NODE_PORT/quic-v1"
echo "  API: http://$PUBLIC_IP:$API_PORT/status"
