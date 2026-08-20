//! palimpsest-node — the runnable Rust node.
//!
//! libp2p (GossipSub over QUIC/TCP + Noise) carries blocks; palimpsest-core
//! (bit-exact against the Python reference via golden vectors) validates them
//! and runs fork choice. This binary is the network layer milestone: a devnet
//! of Rust nodes producing, gossiping, validating, and agreeing on a chain —
//! with the full rev-3 ledger (emissions, registry, data share) live.
//!
//!   # 3-node local devnet (run in three shells, or scripts/devnet.sh):
//!   palimpsest-node --id 0 --n 3 --port 7700 --produce --seconds 30
//!   palimpsest-node --id 1 --n 3 --port 7701 --peers /ip4/127.0.0.1/udp/7700/quic-v1 --produce --seconds 30
//!   palimpsest-node --id 2 --n 3 --port 7702 --peers /ip4/127.0.0.1/udp/7700/quic-v1 --produce --seconds 30
//!
//! Training is NOT here yet (the PyTorch bridge is the next ring): producers
//! mint small deterministic-shape random deltas so consensus, gossip, fork
//! choice, and the token ledger are exercised end to end at wire level.

use clap::Parser;
use libp2p::{
    futures::StreamExt,
    gossipsub, identify,
    swarm::{NetworkBehaviour, SwarmEvent},
    Multiaddr, SwarmBuilder,
};
use palimpsest_core::{
    self as core,
    blocktree::{Block, BlockTree},
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    id: u64,
    #[arg(long, default_value_t = 3)]
    n: u64, // producers in the leader rotation
    #[arg(long, default_value_t = 7700)]
    port: u16,
    #[arg(long, default_value = "")]
    peers: String, // comma-separated multiaddrs
    #[arg(long, default_value_t = false)]
    produce: bool,
    #[arg(long, default_value_t = 3.0)]
    interval: f64, // seconds per round
    #[arg(long, default_value_t = 30.0)]
    seconds: f64,
    #[arg(long, default_value_t = 32)]
    dim: usize, // toy state vector size
    #[arg(long, default_value = "")]
    data_contributor: String,
}

#[derive(NetworkBehaviour)]
struct Behaviour {
    gossipsub: gossipsub::Behaviour,
    identify: identify::Behaviour,
}

// ---------------------------------------------------------------------------
// Block <-> JSON (the same shape the golden vectors use)
// ---------------------------------------------------------------------------

fn header_to_json(h: &core::Header) -> Value {
    json!({"height": h.height, "prev_hash": h.prev_hash, "state_root": h.state_root,
           "txset_root": h.txset_root, "n_txs": h.n_txs, "work": h.work,
           "proposer": h.proposer, "transfer_root": h.transfer_root,
           "ledger_root": h.ledger_root, "data_root": h.data_root})
}

fn header_from_json(v: &Value) -> Option<core::Header> {
    Some(core::Header {
        height: v["height"].as_u64()?,
        prev_hash: v["prev_hash"].as_str()?.into(),
        state_root: v["state_root"].as_str()?.into(),
        txset_root: v["txset_root"].as_str()?.into(),
        n_txs: v["n_txs"].as_u64()?,
        work: v["work"].as_u64()?,
        proposer: v["proposer"].as_str()?.into(),
        transfer_root: v["transfer_root"].as_str()?.into(),
        ledger_root: v["ledger_root"].as_str()?.into(),
        data_root: v["data_root"].as_str()?.into(),
    })
}

fn block_to_json(b: &Block) -> Value {
    json!({
        "header": header_to_json(&b.header),
        "txs": b.txs.iter().map(|t| json!({
            "miner": t.miner, "base_height": t.base_height, "shard_id": t.shard_id,
            "delta_hash": t.delta_hash, "da_pointer": t.da_pointer,
            "sig": hex::encode(&t.sig)})).collect::<Vec<_>>(),
        "bodies": b.bodies.iter().map(|(k, v)| (k.clone(), json!(v)))
            .collect::<serde_json::Map<_, _>>(),
    })
}

fn block_from_json(v: &Value) -> Option<Block> {
    let header = header_from_json(&v["header"])?;
    let mut txs = Vec::new();
    for t in v["txs"].as_array()? {
        txs.push(core::BackpropTx {
            miner: t["miner"].as_str()?.into(),
            base_height: t["base_height"].as_u64()?,
            shard_id: t["shard_id"].as_u64()?,
            delta_hash: t["delta_hash"].as_str()?.into(),
            da_pointer: t["da_pointer"].as_str()?.into(),
            sig: hex::decode(t["sig"].as_str()?).ok()?,
        });
    }
    let mut bodies = HashMap::new();
    for (k, arr) in v["bodies"].as_object()? {
        bodies.insert(k.clone(),
                      arr.as_array()?.iter().map(|x| x.as_i64().unwrap()).collect());
    }
    Some(Block { header, txs, bodies, transfers: vec![], data_txs: vec![] })
}

// ---------------------------------------------------------------------------
// Producer: mint a signed delta + build a valid block on the current head
// ---------------------------------------------------------------------------

fn produce_block(tree: &BlockTree, key: &core::Key, dim: usize, round: u64) -> Block {
    // a deterministic-per-(round,node) pseudo-delta — stands in for the PyTorch
    // pseudo-gradient until the training bridge lands (next ring)
    let seed_bytes = format!("delta|{}|{}", round, key.pub_hex());
    let h = core::delta_hash(seed_bytes.as_bytes());
    let mut delta = vec![0i64; dim];
    for (i, d) in delta.iter_mut().enumerate() {
        let byte = u8::from_str_radix(&h[(i * 2) % 60..(i * 2) % 60 + 2], 16).unwrap();
        *d = byte as i64 - 128;
    }
    let parent = tree.head.clone();
    let parent_h = tree.blocks[&parent].height;
    let dh = core::delta_hash(&core::int64_bytes(&delta));
    let mut tx = core::BackpropTx {
        miner: key.pub_hex(),
        base_height: parent_h,
        shard_id: 0,
        delta_hash: dh.clone(),
        da_pointer: format!("da://{}", dh),
        sig: vec![],
    };
    tx.sig = key.sign(&tx.signing_bytes());

    let parent_w = &tree.state[&parent];
    let mean = core::trimmed_mean(&[delta.clone()], 0.2);
    let w: Vec<i64> = parent_w.iter().zip(&mean).map(|(a, b)| a + b).collect();
    let mut ledger = tree.ledger[&parent].clone();
    let mut header = core::Header {
        height: parent_h + 1,
        prev_hash: parent.clone(),
        state_root: core::state_root(&w),
        txset_root: core::txset_root(&[tx.txid()]),
        n_txs: 1,
        work: 1000,
        proposer: key.pub_hex(),
        transfer_root: core::token::transfer_root(&[]),
        ledger_root: String::new(),
        data_root: core::token::data_root(&[]),
    };
    ledger.resolve_expired_challenges(header.height);
    ledger.apply_reward(header.height, &[tx.miner.clone()], &header.proposer, &[]);
    header.ledger_root = ledger.root();
    let mut bodies = HashMap::new();
    bodies.insert(tx.da_pointer.clone(), delta);
    Block { header, txs: vec![tx], bodies, transfers: vec![], data_txs: vec![] }
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // deterministic devnet identity from --id (real nodes use a wallet key)
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&args.id.to_le_bytes());
    seed[8..16].copy_from_slice(b"palimpse");
    let consensus_key = core::Key::from_seed(seed);
    let p2p_key = libp2p::identity::Keypair::ed25519_from_bytes(seed)?;

    // the shared devnet genesis: a fixed small state vector
    let genesis_w: Vec<i64> = (0..args.dim as i64).map(|i| i * 100).collect();
    let dc = if args.data_contributor.is_empty() { None } else {
        Some(args.data_contributor.clone())
    };
    let mut tree = BlockTree::new(genesis_w, dc);

    let mut swarm = SwarmBuilder::with_existing_identity(p2p_key)
        .with_tokio()
        .with_tcp(libp2p::tcp::Config::default(),
                  libp2p::noise::Config::new, libp2p::yamux::Config::default)?
        .with_quic()
        .with_behaviour(|key| {
            let gs_cfg = gossipsub::ConfigBuilder::default()
                .max_transmit_size(4 * 1024 * 1024)
                .validation_mode(gossipsub::ValidationMode::Permissive)
                .build()
                .unwrap();
            Behaviour {
                gossipsub: gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(key.clone()), gs_cfg).unwrap(),
                identify: identify::Behaviour::new(identify::Config::new(
                    "/palimpsest/0.3.0".into(), key.public())),
            }
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(120)))
        .build();

    let topic = gossipsub::IdentTopic::new("palimpsest/blocks/v3");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
    swarm.listen_on(format!("/ip4/0.0.0.0/udp/{}/quic-v1", args.port).parse::<Multiaddr>()?)?;
    swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", args.port).parse::<Multiaddr>()?)?;
    for p in args.peers.split(',').filter(|s| !s.is_empty()) {
        swarm.dial(p.parse::<Multiaddr>()?)?;
    }

    let t0 = now() + 3.0;
    let end = now() + args.seconds;
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    let mut last_round: i64 = -1;
    let mut last_sync = now();
    // retained full blocks (JSON) for the devnet chain-sync gossip; the tree
    // itself keeps only headers + state
    let mut full_blocks: HashMap<String, Value> = HashMap::new();

    println!("node {} listening on {} (quic+tcp) — {}",
             args.id, args.port, if args.produce { "producer" } else { "observer" });

    while now() < end {
        tokio::select! {
            _ = tick.tick() => {
                let round = ((now() - t0) / args.interval).floor() as i64;
                if args.produce && round >= 0 && round != last_round
                    && (round as u64) % args.n == args.id {
                    last_round = round;
                    let block = produce_block(&tree, &consensus_key, args.dim, round as u64);
                    let bjson = block_to_json(&block);
                    let bhash = block.hash();
                    let msg = json!({"type": "block", "block": bjson.clone()});
                    let _ = swarm.behaviour_mut().gossipsub
                        .publish(topic.clone(), msg.to_string().into_bytes());
                    match tree.add_block(block) {
                        Ok(_) => {
                            full_blocks.insert(bhash, bjson);
                            println!("node {} PRODUCED h{} head={}",
                                args.id, tree.blocks[&tree.head].height, &tree.head[..10]);
                        }
                        Err(e) => println!("node {} own block rejected: {}", args.id, e.0),
                    }
                }
                // periodic full-chain sync gossip (devnet-crude IBD: chains are tiny)
                if now() - last_sync > 5.0 {
                    last_sync = now();
                    let mut chain = Vec::new();
                    let mut cur = tree.head.clone();
                    let mut hashes = Vec::new();
                    while cur != tree.genesis_hash {
                        hashes.push(cur.clone());
                        cur = tree.blocks[&cur].prev_hash.clone();
                    }
                    for h in hashes.iter().rev() {
                        if let Some(b) = full_blocks.get(h) {
                            chain.push(b.clone());
                        }
                    }
                    if !chain.is_empty() {
                        let msg = json!({"type": "chain", "blocks": chain});
                        let _ = swarm.behaviour_mut().gossipsub
                            .publish(topic.clone(), msg.to_string().into_bytes());
                    }
                }
            }
            ev = swarm.select_next_some() => {
                if let SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(
                        gossipsub::Event::Message { message, .. })) = ev {
                    if let Ok(v) = serde_json::from_slice::<Value>(&message.data) {
                        match v["type"].as_str() {
                            Some("block") => {
                                if let Some(b) = block_from_json(&v["block"]) {
                                    let bh = b.hash();
                                    if tree.add_block(b).is_ok() {
                                        full_blocks.insert(bh, v["block"].clone());
                                    }
                                }
                            }
                            Some("chain") => {
                                for bj in v["blocks"].as_array().unwrap_or(&vec![]) {
                                    if let Some(b) = block_from_json(bj) {
                                        let bh = b.hash();
                                        if tree.add_block(b).is_ok() {
                                            full_blocks.insert(bh, bj.clone());
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // final report — devnet convergence check greps these lines
    let head = tree.blocks[&tree.head].clone();
    let mut lineage = Vec::new();
    let mut cur = tree.head.clone();
    while cur != tree.genesis_hash {
        lineage.push(cur[..6].to_string());
        cur = tree.blocks[&cur].prev_hash.clone();
    }
    lineage.reverse();
    println!("node {} LINEAGE {}", args.id, lineage.join(">"));
    println!("node {} done — height {} head {} supply {} ledger {}",
             args.id, head.height, &tree.head[..16],
             tree.head_ledger().supply(), &tree.head_ledger().root()[..12]);
    Ok(())
}
