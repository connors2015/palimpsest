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
    key_seed: String,        // DEPRECATED: raw hex seed on argv (ps-visible!)
    #[arg(long, default_value = "")]
    key_file: String,        // path to a 0600 file holding a 32-byte hex seed
    #[arg(long, default_value = "")]
    genesis_file: String,    // raw i64-LE genesis vector (ceremony artifact)
    #[arg(long, default_value = "")]
    genesis_hash: String,    // published genesis id; a fresh node fetches +
                             // verifies the genesis from a peer against this
    #[arg(long, default_value_t = 0)]
    toy_dim: usize,          // devnet: deterministic toy genesis of this size
    #[arg(long, default_value_t = 7900)]
    port: u16,
    #[arg(long, default_value_t = 8090)]
    api_port: u16,
    #[arg(long, default_value = "0.0.0.0")]
    api_bind: String,        // interface for the HTTP API/dashboard
    #[arg(long, default_value = "0.0.0.0")]
    listen_bind: String,     // p2p listen interface; pin to one NIC (e.g. the
                             // LAN IP) on hosts with docker/k8s/libvirt bridges
                             // so libp2p stops advertising unreachable addrs
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
    #[arg(long, default_value = "")]
    data_refs: String,       // rev 5: comma-separated data_hashes of the staked
                             // corpora this miner trains on; named on every delta
                             // for provenance (empty deltas are rejected)
    /// PREFLIGHT: verify this machine can actually contribute — peer reachable,
    /// genesis id matches, identity/disk usable, and (with --interval) whether a
    /// training round can finish inside the block window. Prints a verdict and
    /// exits without touching the chain. Run this BEFORE mining for hours.
    #[arg(long, default_value_t = false)]
    check: bool,
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

/// Decode a 32-byte hex seed, wiping the hex text + decoded Vec afterwards so
/// transient key material doesn't linger in freed memory.
fn seed_from_hex(mut hexed: String) -> [u8; 32] {
    use zeroize::Zeroize;
    let mut raw = hex::decode(hexed.trim()).expect("key seed must be hex");
    hexed.zeroize();
    let out: [u8; 32] = raw.as_slice().try_into().expect("key seed must be 32 bytes");
    raw.zeroize();
    out
}

/// Load the node identity WITHOUT ever taking key material from argv (which is
/// world-readable via ps/proc). Preferred sources, in order: a key file (0600),
/// the PALIMPSEST_KEY_SEED env var, an (encrypted) wallet. --key-seed remains
/// only as a loud-deprecated fallback for local devnet.
fn load_identity(args: &Args) -> [u8; 32] {
    if !args.key_file.is_empty() {
        let hexed = std::fs::read_to_string(&args.key_file)
            .expect("--key-file unreadable");
        return seed_from_hex(hexed);
    }
    if let Ok(hexed) = std::env::var("PALIMPSEST_KEY_SEED") {
        if !hexed.is_empty() {
            std::env::remove_var("PALIMPSEST_KEY_SEED"); // don't leak to children
            return seed_from_hex(hexed);
        }
    }
    if !args.key_seed.is_empty() {
        warn!("--key-seed passes the private key on the command line, visible in \
               ps/proc to any local user; use --key-file or PALIMPSEST_KEY_SEED");
        return seed_from_hex(args.key_seed.clone());
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
    panic!("identity required: --key-file, PALIMPSEST_KEY_SEED, or --wallet");
}

/// Resolve the genesis weights: local disk (durable) -> --genesis-file ->
/// --toy-dim -> FETCH from a peer, verified against the published --genesis-hash.
/// The genesis is public + self-verifying, so a fresh node bootstraps from a
/// single peer address plus the (tiny) published genesis id.
async fn resolve_genesis(args: &Args, store: &store::Store,
                         swarm: &mut libp2p::Swarm<node::Behaviour>) -> Vec<i64> {
    if let Some(g) = store.read_genesis() {
        return g; // durable once written
    }
    let g: Vec<i64> = if !args.genesis_file.is_empty() {
        let raw = std::fs::read(&args.genesis_file).expect("genesis file unreadable");
        raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    } else if args.toy_dim > 0 {
        (0..args.toy_dim as i64).map(|i| i * 100).collect()
    } else if !args.genesis_hash.is_empty() {
        info!(id = %args.genesis_hash, "no local genesis — fetching it from the network");
        match fetch_genesis(swarm, &args.genesis_hash, &args.peers).await {
            Some(g) => g,
            // Peer-fetch only works for genesis vectors that fit the sync
            // response cap. The production model is ~650MB raw, far over it, so
            // this path fails by design there — don't leave the operator staring
            // at a 3-minute hang and a bare panic. The genesis is DETERMINISTIC,
            // so generating it locally is both faster and trustless.
            None => panic!("{}", [
                "could not fetch a genesis matching --genesis-hash from any peer.",
                "",
                "For the production model this is expected: the genesis (~650MB) is",
                "far larger than the sync transport can carry. Generate it locally —",
                "it is deterministic, so this is trustless, and faster than a download:",
                "",
                "  uv run --with torch --with numpy --with pynacl \\",
                "      python -m client.make_genesis --model small --seed 1337 --out genesis.bin",
                "",
                "then re-run with --genesis-file genesis.bin (the printed",
                "genesis_state_root must equal the published genesis id).",
            ].join("\n")),
        }
    } else {
        panic!("genesis required: --genesis-file, --toy-dim, or --genesis-hash + --peers");
    };
    store.write_genesis(&g).expect("cannot persist genesis");
    g
}

/// Fetch the genesis weights from a peer and verify they hash to the expected
/// genesis id before adopting them (so a malicious peer can't seed a wrong
/// genesis). Times out after a few minutes with no matching response.
/// PREFLIGHT (`--check`): answer "can this machine actually contribute?" before
/// the operator spends hours finding out it can't. Every check prints PASS/WARN/
/// FAIL with the concrete remedy — this exists because the failure modes here are
/// silent by nature (a too-slow trainer mines forever and earns nothing).
async fn preflight(args: &Args, key: &core::Key, store: &store::Store,
                   swarm: &mut libp2p::Swarm<node::Behaviour>)
                   -> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt;
    let (mut fails, mut warns) = (0u32, 0u32);
    let pass = |m: String| println!("  \x1b[32mPASS\x1b[0m  {m}");
    println!("\npalimpsest preflight — can this machine contribute?\n");

    // 1. identity + data dir (already opened above, so both are usable)
    pass(format!("identity loaded — miner {} / address {}",
                 &key.pub_hex()[..12], &core::token::address(&key.pub_hex())[..12]));
    pass(format!("data dir writable + exclusively locked ({})", args.data_dir));
    if store.read_genesis().is_some() {
        pass("existing chain on disk — will resume from it".into());
    }

    // 2. peer reachability: can we actually dial the bootstrap?
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut connected = 0usize;
    while std::time::Instant::now() < deadline {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                connected = swarm.connected_peers().count();
                if connected > 0 { break }
                node::dial_peers(swarm, &args.peers);
            }
            _ = swarm.select_next_some() => {
                connected = swarm.connected_peers().count();
                if connected > 0 { break }
            }
        }
    }
    if args.peers.is_empty() {
        println!("  \x1b[33mWARN\x1b[0m  no --peers given — this node will be alone \
                  (fine for a local devnet, not for joining)");
        warns += 1;
    } else if connected > 0 {
        pass(format!("bootstrap reachable — {connected} peer(s) connected"));
    } else {
        println!("  \x1b[31mFAIL\x1b[0m  cannot reach any peer in --peers within 30s.");
        println!("        Check the address/port and that outbound TCP is allowed.");
        fails += 1;
    }

    // 3. genesis agreement — the thing that silently forks you off the network
    if !args.genesis_hash.is_empty() {
        if connected > 0 {
            match fetch_genesis(swarm, &args.genesis_hash, &args.peers).await {
                Some(g) => pass(format!(
                    "genesis verified against the published id ({} params)", g.len())),
                None => {
                    println!("  \x1b[33mWARN\x1b[0m  no peer served a genesis \
                              matching the published id.");
                    println!("        For the production model this is EXPECTED: \
                              the genesis (~650MB) is too large to ship over the");
                    println!("        sync transport. Generate it locally instead \
                              — it is deterministic, so this is trustless:");
                    println!("          uv run --with torch --with numpy --with \
                              pynacl python -m client.make_genesis \\");
                    println!("              --model small --seed 1337 --out genesis.bin");
                    println!("        then run with --genesis-file genesis.bin \
                              (its printed state_root must equal the published id).");
                    warns += 1;
                }
            }
        }
    } else if args.genesis_file.is_empty() && args.toy_dim == 0
              && store.read_genesis().is_none() {
        println!("  \x1b[31mFAIL\x1b[0m  no genesis source: pass --genesis-hash \
                  (to fetch+verify from a peer), --genesis-file, or --toy-dim.");
        fails += 1;
    }

    // 4. the mining-viability check — the silent killer. A delta is includable
    //    only at base_height == head, so a round slower than the block interval
    //    is ALWAYS dropped. We can't time the GPU from here (that's the
    //    trainer's job, and it now auto-fits), but we can state the budget so
    //    the operator can compare it against their measured round time.
    if args.produce {
        let budget = args.interval * 0.6;
        pass(format!("producing with --interval {:.0}s → trainer budget ~{:.0}s \
                      per round (it auto-fits its steps to this)",
                     args.interval, budget));
        if args.interval < 60.0 {
            println!("  \x1b[33mWARN\x1b[0m  --interval {:.0}s is tight for a real \
                      network: multi-MB deltas may not propagate to the proposer \
                      in time, so your work would be orphaned. 120–180s is safer.",
                     args.interval);
            warns += 1;
        }
        if args.data_refs.is_empty() {
            println!("  \x1b[31mFAIL\x1b[0m  --produce without --data-refs: every \
                      delta you submit will be REJECTED (provenance is required).");
            println!("        Use --data-refs genesis, or the data_hash of a \
                      corpus you have staked.");
            fails += 1;
        } else {
            pass(format!("provenance set — deltas will name: {}", args.data_refs));
        }
    } else {
        pass("watch/serve mode (no --produce) — will sync and serve, not mine".into());
    }

    println!("\n{}\n", match (fails, warns) {
        (0, 0) => "\x1b[32mREADY\x1b[0m — this machine can contribute.".to_string(),
        (0, w) => format!("\x1b[33mREADY WITH {w} WARNING(S)\x1b[0m — see above."),
        (f, _) => format!("\x1b[31mNOT READY — {f} blocking problem(s)\x1b[0m."),
    });
    if fails > 0 { std::process::exit(1) }
    Ok(())
}

async fn fetch_genesis(swarm: &mut libp2p::Swarm<node::Behaviour>,
                       expected_hash: &str, peers: &str) -> Option<Vec<i64>> {
    use futures::StreamExt;
    use libp2p::{request_response, swarm::SwarmEvent};
    let start = std::time::Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_secs(3));
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let connected: Vec<_> = swarm.connected_peers().copied().collect();
                if connected.is_empty() {
                    node::dial_peers(swarm, peers);
                }
                for p in connected {
                    swarm.behaviour_mut().sync.send_request(&p, proto::SyncRequest {
                        from_height: 0, count: 0, want_genesis: true });
                }
                if start.elapsed().as_secs() > 180 {
                    return None;
                }
            }
            ev = swarm.select_next_some() => {
                if let SwarmEvent::Behaviour(node::BehaviourEvent::Sync(
                    request_response::Event::Message {
                        message: request_response::Message::Response { response, .. }, .. })) = ev
                {
                    if let Some(w) = response.genesis {
                        // Accept either published form of the genesis id: the
                        // genesis block hash (header-format dependent) or the
                        // genesis state_root (sha256 of the raw weight bytes —
                        // what make_genesis prints and the docs publish, stable
                        // across header revisions). Both pin the same weights.
                        if core::blocktree::genesis_block_hash(&w) == expected_hash
                            || core::state_root(&w) == expected_hash {
                            info!(dim = w.len(), "fetched + verified genesis from a peer");
                            return Some(w);
                        }
                        warn!("a peer served a genesis that doesn't match the published id");
                    }
                }
            }
        }
    }
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
    let dc = (!args.data_contributor.is_empty()).then(|| args.data_contributor.clone());

    // swarm with the full NAT stack: QUIC+TCP, Noise, relay client, AutoNAT,
    // DCUtR hole punching, optional relay server (seeds). Built BEFORE genesis
    // so a fresh node can fetch the genesis from a peer.
    let relay_server = args.relay_server;
    let mut swarm = SwarmBuilder::with_existing_identity(p2p_key)
        .with_tokio()
        .with_tcp(libp2p::tcp::Config::default(),
                  libp2p::noise::Config::new, libp2p::yamux::Config::default)?
        .with_quic_config(|mut cfg| {
            // libp2p-quic defaults (10s idle / 5s keepalive) are far too tight for
            // a NAT'd peer on a lossy internet path: a couple of dropped keepalives
            // and the connection idle-times-out, then redials — the ~30s connect/
            // drop cycle we saw. Give it a generous idle window with frequent
            // keepalives so NAT mappings stay warm and transient loss is survivable.
            cfg.max_idle_timeout = 120_000;                 // ms
            cfg.keep_alive_interval = Duration::from_secs(15);
            cfg
        })
        .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)?
        .with_behaviour(|key, relay_client| {
            node::behaviour(key, relay_client, relay_server)
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(300)))
        .build();

    let topic = libp2p::gossipsub::IdentTopic::new("palimpsest/v1");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
    // preflight binds an EPHEMERAL port: it must be runnable while your real
    // node is already up, otherwise the check you most want to run is the one
    // you can't ("Address already in use").
    let listen_port = if args.check { 0 } else { args.port };
    swarm.listen_on(format!("/ip4/{}/udp/{}/quic-v1", args.listen_bind, listen_port).parse::<Multiaddr>()?)?;
    swarm.listen_on(format!("/ip4/{}/tcp/{}", args.listen_bind, listen_port).parse::<Multiaddr>()?)?;
    if !args.external_address.is_empty() {
        match args.external_address.parse::<Multiaddr>() {
            Ok(a) => swarm.add_external_address(a),
            Err(e) => warn!("bad --external-address: {e}"),
        }
    }
    node::dial_peers(&mut swarm, &args.peers);

    if args.check {
        return preflight(&args, &consensus_key, &store, &mut swarm).await;
    }

    // genesis: local disk -> --genesis-file -> --toy-dim -> FETCH from a peer,
    // verified against the published --genesis-hash. The genesis is public and
    // self-verifying, so a fresh node bootstraps from one peer + the id.
    resolve_genesis(&args, &store, &mut swarm).await;

    // replay any existing chain from disk (validated)
    let (tree, blocks_full, payloads) = store
        .replay(dc.clone(), args.prune_depth)
        .expect("chain replay failed");

    // Guarantee a current-format snapshot at the replayed head, so the NEXT
    // boot is a fast-boot even for an idle watcher that never advances to a
    // SNAPSHOT_EVERY height. Skips the write if disk already has one at head.
    if !matches!(store.read_snapshot(), Some((h, ..)) if h == tree.head) {
        let head = tree.head.clone();
        let height = tree.blocks[&head].height;
        store.write_snapshot(&head, height, &tree.state[&head], tree.head_ledger());
        info!(height, "wrote boot snapshot for fast-boot");
    }

    // channels: api <-> node, bridge <-> node
    let (api_tx, api_rx) = mpsc::channel(64);
    let (bridge_cmd_tx, bridge_cmd_rx) = mpsc::channel::<bridge::ToBridge>(16);
    let (bridge_ev_tx, bridge_ev_rx) = mpsc::channel::<bridge::FromBridge>(16);
    let api_token = std::env::var("PALIMPSEST_API_TOKEN").ok().filter(|t| !t.is_empty());
    tokio::spawn(api::run(args.api_bind.clone(), args.api_port, api_token, api_tx));
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
        delta_scores: Default::default(),
        delta_sketches: Default::default(),
        omitted_deltas: Default::default(),
        account_pool: Default::default(),
        pending: Default::default(),
        seen: Default::default(),
        seen_order: Default::default(),
        cfg: node::NodeConfig {
            produce: args.produce,
            interval: args.interval,
            rotate,
            seconds: args.seconds,
            peers: args.peers.clone(),
            data_refs: args.data_refs.split(',')
                .map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        },
        topic,
        bridge_tx: bridge_cmd_tx,
        bridge_synced: false,
        train_inflight: false,
        train_deadline: 0.0,
        t0: if args.t0 > 0.0 { args.t0 } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?.as_secs_f64()
        },
        last_proposed_round: -1,
        last_announced_round: -1,
        last_sync_req: Default::default(),
        peers_connected: 0,
        chat_pending: Vec::new(),
        chat_inflight: false,
        stale_deltas: 0,
    };
    node::run(n, swarm, api_rx, bridge_ev_rx).await;
    Ok(())
}
