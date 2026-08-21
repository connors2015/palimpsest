//! palimpsest-node — the production Rust node.
//!
//!   # a producing node with a PyTorch trainer attached:
//!   palimpsest-node --data-dir ~/.palimpsest/node --wallet ~/.palimpsest/wallet.json \
//!       --port 7900 --api-port 8090 --bridge-port 7999 --produce \
//!       --peers /ip4/…/udp/7900/quic-v1 --data-contributor <addr>
//!   python -m client.miner_bridge --node-port 7999 --model small …
//!
//!   # a seed/relay node (always-on bootstrap; relays NAT'd peers):
//!   palimpsest-node --data-dir /var/palimpsest --key-seed <hex32> \
//!       --port 7900 --api-port 8090 --relay-server
//!
//! Genesis: --genesis-file <raw i64-LE .bin> (the ceremony artifact from
//! client/make_genesis.py), or --toy-dim N for a deterministic toy vector.
//! The wallet key IS the miner identity; encrypted wallets are decrypted with
//! $PALIMPSEST_WALLET_PASSPHRASE (argon2id + XSalsa20-Poly1305, the exact
//! pynacl construction).

mod api;
mod bridge;
mod node;
mod proto;
mod store;

use clap::Parser;
use libp2p::{Multiaddr, SwarmBuilder};
use palimpsest_core as core;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "palimpsest-data")]
    data_dir: String,
    #[arg(long, default_value = "")]
    wallet: String,          // wallet.json (identity); or:
    #[arg(long, default_value = "")]
    key_seed: String,        // raw 32-byte hex seed (devnet / infra nodes)
    #[arg(long, default_value = "")]
    genesis_file: String,    // raw i64-LE genesis vector (ceremony artifact)
    #[arg(long, default_value_t = 0)]
    toy_dim: usize,          // devnet: deterministic toy genesis of this size
    #[arg(long, default_value_t = 7900)]
    port: u16,
    #[arg(long, default_value_t = 8090)]
    api_port: u16,
    #[arg(long, default_value_t = 7999)]
    bridge_port: u16,
    #[arg(long, default_value = "")]
    peers: String,
    #[arg(long, default_value_t = false)]
    produce: bool,
    #[arg(long, default_value_t = 10.0)]
    interval: f64,
    #[arg(long, default_value = "")]
    rotate: String,          // "n,id" — deterministic devnet leader rotation
    #[arg(long, default_value_t = 0.0)]
    seconds: f64,            // 0 = run forever
    #[arg(long, default_value = "")]
    data_contributor: String,
    #[arg(long, default_value_t = false)]
    relay_server: bool,      // seeds: relay NAT'd peers (circuit relay v2)
    #[arg(long, default_value = "")]
    external_address: String, // advertise a known public multiaddr
    #[arg(long, default_value_t = 8)]
    prune_depth: u64,
    /// shared round-clock origin (epoch seconds) — REQUIRED for --rotate to
    /// align leader slots across machines; 0 = process start time
    #[arg(long, default_value_t = 0.0)]
    t0: f64,
}

/// Decrypt a pynacl-encrypted wallet: argon2id(MODERATE) -> XSalsa20-Poly1305.
fn decrypt_wallet(enc: &serde_json::Value, passphrase: &str) -> Option<[u8; 32]> {
    use argon2::{Algorithm, Argon2, Params, Version};
    use crypto_secretbox::aead::Aead;
    use crypto_secretbox::{KeyInit, XSalsa20Poly1305};
    let salt = hex::decode(enc["salt"].as_str()?).ok()?;
    let blob = hex::decode(enc["blob"].as_str()?).ok()?;
    // libsodium argon2id13 MODERATE: opslimit 3, memlimit 256 MiB
    let params = Params::new(256 * 1024, 3, 1, Some(32)).ok()?;
    let a2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; 32];
    a2.hash_password_into(passphrase.as_bytes(), &salt, &mut key).ok()?;
    let (nonce, ct) = blob.split_at(24);
    let cipher = XSalsa20Poly1305::new((&key).into());
    let sk = cipher.decrypt(nonce.into(), ct).ok()?;
    sk.try_into().ok()
}

fn load_identity(args: &Args) -> [u8; 32] {
    if !args.key_seed.is_empty() {
        let raw = hex::decode(&args.key_seed).expect("--key-seed must be hex");
        return raw.try_into().expect("--key-seed must be 32 bytes");
    }
    if !args.wallet.is_empty() {
        let raw = std::fs::read_to_string(&args.wallet).expect("wallet file unreadable");
        let w: serde_json::Value = serde_json::from_str(&raw).expect("wallet file corrupt");
        if let Some(sk) = w.get("sk").and_then(|s| s.as_str()) {
            return hex::decode(sk).unwrap().try_into().unwrap();
        }
        if let Some(enc) = w.get("enc") {
            let pw = std::env::var("PALIMPSEST_WALLET_PASSPHRASE")
                .expect("encrypted wallet: set PALIMPSEST_WALLET_PASSPHRASE");
            return decrypt_wallet(enc, &pw)
                .expect("wallet decryption failed (wrong passphrase?)");
        }
        panic!("wallet file has neither sk nor enc");
    }
    panic!("identity required: --wallet or --key-seed");
}

fn load_genesis(args: &Args, store: &store::Store) -> Vec<i64> {
    if let Some(g) = store.read_genesis() {
        return g;                                   // durable once written
    }
    let g: Vec<i64> = if !args.genesis_file.is_empty() {
        let raw = std::fs::read(&args.genesis_file).expect("genesis file unreadable");
        raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<i64>>()
    } else if args.toy_dim > 0 {
        (0..args.toy_dim as i64).map(|i| i * 100).collect::<Vec<i64>>()
    } else {
        panic!("genesis required: --genesis-file or --toy-dim");
    };
    store.write_genesis(&g).expect("cannot persist genesis");
    g
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,libp2p=warn")))
        .init();
    let args = Args::parse();

    let seed = load_identity(&args);
    let consensus_key = core::Key::from_seed(seed);
    let p2p_key = libp2p::identity::Keypair::ed25519_from_bytes(seed)?;
    info!(miner = &consensus_key.pub_hex()[..12], "identity loaded");

    let store = store::Store::open(&args.data_dir)?;
    let _genesis = load_genesis(&args, &store);  // side effect: persists genesis.bin
    let dc = (!args.data_contributor.is_empty()).then(|| args.data_contributor.clone());

    // replay any existing chain from disk (validated)
    let (tree, blocks_full, payloads) = store
        .replay(dc.clone(), args.prune_depth)
        .expect("chain replay failed");

    // swarm with the full NAT stack: QUIC+TCP, Noise, relay client, AutoNAT,
    // DCUtR hole punching, optional relay server (seeds)
    let relay_server = args.relay_server;
    let mut swarm = SwarmBuilder::with_existing_identity(p2p_key)
        .with_tokio()
        .with_tcp(libp2p::tcp::Config::default(),
                  libp2p::noise::Config::new, libp2p::yamux::Config::default)?
        .with_quic()
        .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)?
        .with_behaviour(|key, relay_client| {
            node::behaviour(key, relay_client, relay_server)
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(300)))
        .build();

    let topic = libp2p::gossipsub::IdentTopic::new("palimpsest/v1");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
    swarm.listen_on(format!("/ip4/0.0.0.0/udp/{}/quic-v1", args.port).parse::<Multiaddr>()?)?;
    swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", args.port).parse::<Multiaddr>()?)?;
    if !args.external_address.is_empty() {
        match args.external_address.parse::<Multiaddr>() {
            Ok(a) => swarm.add_external_address(a),
            Err(e) => warn!("bad --external-address: {e}"),
        }
    }
    node::dial_peers(&mut swarm, &args.peers);

    // channels: api <-> node, bridge <-> node
    let (api_tx, api_rx) = mpsc::channel(64);
    let (bridge_cmd_tx, bridge_cmd_rx) = mpsc::channel::<bridge::ToBridge>(16);
    let (bridge_ev_tx, bridge_ev_rx) = mpsc::channel::<bridge::FromBridge>(16);
    tokio::spawn(api::run(args.api_port, api_tx));
    tokio::spawn(bridge::run(args.bridge_port, bridge_cmd_rx, bridge_ev_tx));

    let rotate = (!args.rotate.is_empty()).then(|| {
        let (n, id) = args.rotate.split_once(',').expect("--rotate n,id");
        (n.parse().unwrap(), id.parse().unwrap())
    });
    let n = node::Node {
        tree,
        store,
        key: consensus_key,
        blocks_full,
        payloads,
        delta_pool: Default::default(),
        account_pool: Default::default(),
        pending: Default::default(),
        seen: Default::default(),
        cfg: node::NodeConfig {
            produce: args.produce,
            interval: args.interval,
            rotate,
            seconds: args.seconds,
            data_contributor: dc,
            peers: args.peers.clone(),
        },
        topic,
        bridge_tx: bridge_cmd_tx,
        bridge_synced: false,
        train_inflight: false,
        t0: if args.t0 > 0.0 { args.t0 } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?.as_secs_f64()
        },
        last_proposed_round: -1,
        last_announced_round: -1,
        last_sync_req: Default::default(),
    };
    node::run(n, swarm, api_rx, bridge_ev_rx).await;
    Ok(())
}
