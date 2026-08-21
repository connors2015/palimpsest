//! The node loop — single owner of all chain state, driving:
//! gossip (delta txs + payloads, account txs, commitment-only blocks),
//! range sync (request-response), the trainer bridge, block production,
//! persistence, and the HTTP API. NAT traversal behaviours (AutoNAT, DCUtR,
//! relay client, optional relay server for seeds) ride the same swarm.

use crate::api::ApiCmd;
use crate::bridge::{FromBridge, ToBridge};
use crate::proto::*;
use crate::store::{Store, SNAPSHOT_EVERY};
use libp2p::{
    autonat, dcutr, gossipsub, identify, ping, relay,
    futures::StreamExt,
    request_response::{self, ProtocolSupport},
    swarm::{behaviour::toggle::Toggle, NetworkBehaviour, SwarmEvent},
    Multiaddr, PeerId, StreamProtocol, Swarm,
};
use palimpsest_core::{self as core, blocktree::BlockTree, token::AccountTx};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub const INCLUDE_K: usize = 8;

// --- mempool / cache bounds (DoS hardening) --------------------------------
// Every pool and dedup set is size-capped with eviction, and deltas/txs are
// admitted only within a narrow, includable window around the head — otherwise
// an unauthenticated peer can mint unlimited well-formed txs and grow memory +
// disk without bound. A delta is includable only when base_height == head
// height (validate_block requires it), so anything materially older can never
// be used and anything far in the future is spam.
const DELTA_STALE_SLACK: u64 = 2; // tolerate this many blocks of head lag
const DELTA_FUTURE_WINDOW: u64 = 4; // ...and this much look-ahead
const MAX_DELTA_POOL: usize = 512;
const MAX_ACCOUNT_POOL: usize = 4096;
const MAX_PENDING: usize = 256;
const MAX_SEEN: usize = 100_000;

/// A delta is worth holding only if its base_height sits in the includable
/// window around the current head. Pure + total so it can be unit-tested.
pub fn delta_in_window(base_height: u64, head_height: u64) -> bool {
    base_height + DELTA_STALE_SLACK >= head_height
        && base_height <= head_height + DELTA_FUTURE_WINDOW
}

#[cfg(test)]
mod mempool_bounds_tests {
    use super::*;

    #[test]
    fn delta_window_admits_near_head_rejects_far() {
        let head = 100;
        assert!(delta_in_window(head, head), "at-head delta is includable");
        assert!(delta_in_window(head - DELTA_STALE_SLACK, head), "within slack kept");
        assert!(!delta_in_window(head - DELTA_STALE_SLACK - 1, head), "too stale dropped");
        assert!(delta_in_window(head + DELTA_FUTURE_WINDOW, head), "near-future kept");
        assert!(!delta_in_window(head + DELTA_FUTURE_WINDOW + 1, head), "far-future dropped");
    }

    #[test]
    fn delta_window_safe_at_genesis() {
        // head 0 must not underflow / panic
        assert!(delta_in_window(0, 0));
        assert!(delta_in_window(3, 0));
        assert!(!delta_in_window(0 + DELTA_FUTURE_WINDOW + 1, 0));
    }
}

/// Length-prefixed JSON sync codec with explicit large limits — the stock JSON
/// codec caps responses ~10MB, but an 86M-model compressed payload is ~18MB.
#[derive(Clone, Default)]
pub struct BigJsonCodec;

const SYNC_REQ_MAX: u64 = 16 * 1024 * 1024;
const SYNC_RESP_MAX: u64 = 512 * 1024 * 1024;

#[async_trait::async_trait]
impl request_response::Codec for BigJsonCodec {
    type Protocol = StreamProtocol;
    type Request = SyncRequest;
    type Response = SyncResponse;

    async fn read_request<T>(&mut self, _: &StreamProtocol, io: &mut T)
        -> std::io::Result<SyncRequest>
    where T: futures::AsyncRead + Unpin + Send {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.take(SYNC_REQ_MAX).read_to_end(&mut buf).await?;
        serde_json::from_slice(&buf).map_err(std::io::Error::other)
    }

    async fn read_response<T>(&mut self, _: &StreamProtocol, io: &mut T)
        -> std::io::Result<SyncResponse>
    where T: futures::AsyncRead + Unpin + Send {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.take(SYNC_RESP_MAX).read_to_end(&mut buf).await?;
        serde_json::from_slice(&buf).map_err(std::io::Error::other)
    }

    async fn write_request<T>(&mut self, _: &StreamProtocol, io: &mut T, req: SyncRequest)
        -> std::io::Result<()>
    where T: futures::AsyncWrite + Unpin + Send {
        use futures::AsyncWriteExt;
        io.write_all(&serde_json::to_vec(&req)?).await?;
        io.close().await
    }

    async fn write_response<T>(&mut self, _: &StreamProtocol, io: &mut T, resp: SyncResponse)
        -> std::io::Result<()>
    where T: futures::AsyncWrite + Unpin + Send {
        use futures::AsyncWriteExt;
        io.write_all(&serde_json::to_vec(&resp)?).await?;
        io.close().await
    }
}

#[derive(NetworkBehaviour)]
pub struct Behaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub identify: identify::Behaviour,
    pub sync: request_response::Behaviour<BigJsonCodec>,
    pub autonat: autonat::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub relay_server: Toggle<relay::Behaviour>,
    pub ping: ping::Behaviour,
}

pub fn behaviour(
    key: &libp2p::identity::Keypair,
    relay_client: relay::client::Behaviour,
    relay_server: bool,
) -> Behaviour {
    let peer_id = key.public().to_peer_id();
    let gs_cfg = gossipsub::ConfigBuilder::default()
        .max_transmit_size(64 * 1024 * 1024)     // 86M compressed payloads fit
        .validation_mode(gossipsub::ValidationMode::Permissive)
        .build()
        .unwrap();
    Behaviour {
        gossipsub: gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(key.clone()), gs_cfg).unwrap(),
        identify: identify::Behaviour::new(identify::Config::new(
            "/palimpsest/1.0.0".into(), key.public())),
        sync: request_response::Behaviour::with_codec(
            BigJsonCodec,
            [(StreamProtocol::new("/palimpsest/sync/1"), ProtocolSupport::Full)],
            request_response::Config::default()
                .with_request_timeout(Duration::from_secs(300)),
        ),
        autonat: autonat::Behaviour::new(peer_id, autonat::Config::default()),
        dcutr: dcutr::Behaviour::new(peer_id),
        relay_client,
        relay_server: Toggle::from(relay_server.then(|| {
            relay::Behaviour::new(peer_id, relay::Config::default())
        })),
        ping: ping::Behaviour::default(),
    }
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

pub struct NodeConfig {
    pub produce: bool,
    pub interval: f64,
    pub rotate: Option<(u64, u64)>, // (n, id): deterministic devnet rotation
    pub seconds: f64,               // 0 = run forever
    pub data_contributor: Option<String>,
    pub peers: String,              // configured peers — re-dialed when lost
}

pub struct Node {
    pub tree: BlockTree,
    pub store: Store,
    pub key: core::Key,
    pub blocks_full: HashMap<String, StoredBlock>,
    pub payloads: HashMap<String, Payload>,       // txid -> compressed payload
    pub delta_pool: HashMap<String, core::BackpropTx>,
    pub account_pool: HashMap<String, AccountTx>,
    pub pending: HashMap<String, (StoredBlock, PeerId)>, // blocks awaiting payloads
    pub seen: HashSet<String>,
    /// insertion order for `seen`, so it can be bounded as a recency ring
    pub seen_order: VecDeque<String>,
    pub cfg: NodeConfig,
    pub topic: gossipsub::IdentTopic,
    pub bridge_tx: mpsc::Sender<ToBridge>,
    pub bridge_synced: bool,
    pub train_inflight: bool,
    pub t0: f64,
    pub last_proposed_round: i64,
    pub last_announced_round: i64,
    /// per-peer timestamp of the last sync we requested — heartbeat-triggered
    /// catch-up must not stack concurrent multi-hundred-MB transfers
    pub last_sync_req: HashMap<PeerId, f64>,
    pub peers_connected: usize,
    pub chat_pending: Vec<tokio::sync::oneshot::Sender<Value>>,
    pub chat_inflight: bool,
}

impl Node {
    fn head_height(&self) -> u64 {
        self.tree.blocks[&self.tree.head].height
    }

    fn publish(&mut self, swarm: &mut Swarm<Behaviour>, msg: &Gossip) {
        let bytes = serde_json::to_vec(msg).unwrap();
        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(self.topic.clone(), bytes) {
            debug!("publish: {e}");                // e.g. no peers yet — fine
        }
    }

    /// Record a txid as seen, bounding the set as an insertion-ordered ring so a
    /// peer streaming unique txids can't grow it without limit.
    fn mark_seen(&mut self, txid: String) {
        if self.seen.insert(txid.clone()) {
            self.seen_order.push_back(txid);
            while self.seen_order.len() > MAX_SEEN {
                if let Some(old) = self.seen_order.pop_front() {
                    self.seen.remove(&old);
                }
            }
        }
    }

    /// Drop a never-included delta from the mempool AND reclaim its disk payload
    /// (it was written on accept; if it's never mined it is pure garbage).
    fn drop_pool_delta(&mut self, txid: &str) {
        self.delta_pool.remove(txid);
        self.payloads.remove(txid);
        self.store.remove_payload(txid);
    }

    /// Evict stale/over-cap deltas. Stale = outside the includable window (can
    /// never be mined); over-cap = keep the freshest MAX_DELTA_POOL by height.
    fn evict_delta_pool(&mut self) {
        let head = self.head_height();
        let stale: Vec<String> = self.delta_pool.iter()
            .filter(|(_, t)| !delta_in_window(t.base_height, head))
            .map(|(id, _)| id.clone()).collect();
        for id in stale {
            self.drop_pool_delta(&id);
        }
        if self.delta_pool.len() > MAX_DELTA_POOL {
            let mut by_h: Vec<(String, u64)> = self.delta_pool.iter()
                .map(|(id, t)| (id.clone(), t.base_height)).collect();
            by_h.sort_by_key(|(_, h)| *h);                 // stalest first
            let excess = self.delta_pool.len() - MAX_DELTA_POOL;
            for (id, _) in by_h.into_iter().take(excess) {
                self.drop_pool_delta(&id);
            }
        }
    }

    /// Evict account txs whose nonce is now below the sender's ledger nonce
    /// (can never apply), then cap by dropping the most speculative (highest
    /// nonce) first.
    fn evict_account_pool(&mut self) {
        use palimpsest_core::token::address;
        let stale: Vec<String> = {
            let led = self.tree.head_ledger();
            self.account_pool.iter()
                .filter(|(_, t)| t.nonce()
                        < led.nonces.get(&address(t.sender_pub())).copied().unwrap_or(0))
                .map(|(id, _)| id.clone()).collect()
        };
        for id in stale {
            self.account_pool.remove(&id);
        }
        if self.account_pool.len() > MAX_ACCOUNT_POOL {
            let mut by_n: Vec<(String, u64)> = self.account_pool.iter()
                .map(|(id, t)| (id.clone(), t.nonce())).collect();
            by_n.sort_by_key(|(_, n)| std::cmp::Reverse(*n));  // most future first
            let excess = self.account_pool.len() - MAX_ACCOUNT_POOL;
            for (id, _) in by_n.into_iter().take(excess) {
                self.account_pool.remove(&id);
            }
        }
    }

    /// Queue an orphan/missing-payload block, bounded: when full, evict the
    /// lowest-height pending block (least likely to ever become live).
    fn queue_pending(&mut self, bh: String, sb: StoredBlock, peer: PeerId) {
        if self.pending.len() >= MAX_PENDING && !self.pending.contains_key(&bh) {
            if let Some(drop) = self.pending.iter()
                .min_by_key(|(_, (s, _))| s.header.height).map(|(h, _)| h.clone()) {
                self.pending.remove(&drop);
            }
        }
        self.pending.insert(bh, (sb, peer));
    }

    // ---- delta txs (from our bridge or from gossip) ----------------------
    fn accept_delta(&mut self, tx: core::BackpropTx, payload: Payload) -> bool {
        let txid = tx.txid();
        if self.seen.contains(&txid) || !tx.verify() {
            return false;
        }
        let Some(dense) = payload.dense() else { return false };
        if core::delta_hash(&core::int64_bytes(&dense)) != tx.delta_hash {
            warn!("delta payload hash mismatch from {}", &tx.miner[..8]);
            return false;
        }
        // height gate: only admit deltas that can plausibly be mined onto head
        if !delta_in_window(tx.base_height, self.head_height()) {
            return false;
        }
        self.mark_seen(txid.clone());
        self.store.put_payload(&txid, &payload);
        self.payloads.insert(txid.clone(), payload);
        self.delta_pool.insert(txid, tx);
        self.evict_delta_pool();
        true
    }

    fn accept_account_tx(&mut self, tx: AccountTx) -> Option<String> {
        use palimpsest_core::token::address;
        let txid = tx.txid();
        if self.seen.contains(&txid) || !tx.verify() {
            return None;
        }
        // nonce gate: a tx below the sender's current nonce can never apply
        let cur = self.tree.head_ledger().nonces
            .get(&address(tx.sender_pub())).copied().unwrap_or(0);
        if tx.nonce() < cur {
            return None;
        }
        self.mark_seen(txid.clone());
        self.account_pool.insert(txid.clone(), tx);
        self.evict_account_pool();
        Some(txid)
    }

    // ---- block production ------------------------------------------------
    fn build_candidate(&self) -> Option<(StoredBlock, palimpsest_core::blocktree::Block)> {
        let head = self.tree.head.clone();
        let hh = self.head_height();
        let mut cands: Vec<&core::BackpropTx> = self.delta_pool.values()
            .filter(|t| t.base_height == hh).collect();
        if cands.is_empty() {
            return None;
        }
        cands.sort_by_key(|t| t.txid());
        let mut chosen = Vec::new();
        let mut miners = HashSet::new();
        for t in cands {                            // one delta per miner
            if miners.insert(t.miner.clone()) {
                chosen.push((*t).clone());
                if chosen.len() >= INCLUDE_K {
                    break;
                }
            }
        }
        // weight-state transition
        let deltas: Vec<Vec<i64>> = chosen.iter()
            .map(|t| self.payloads[&t.txid()].dense().unwrap()).collect();
        let mean = core::trimmed_mean(&deltas, 0.2);
        let parent_w = &self.tree.state[&head];
        // wrapping_add mirrors numpy int64 (matches validate_block exactly)
        let w: Vec<i64> = parent_w.iter().zip(&mean).map(|(a, b)| a.wrapping_add(*b)).collect();
        // account lanes: dry-run in the validator's exact order
        let mut scratch = self.tree.ledger[&head].clone();
        scratch.resolve_expired_challenges(hh + 1);
        let miner_pubs: Vec<String> = chosen.iter().map(|t| t.miner.clone()).collect();
        scratch.apply_reward(hh + 1, &miner_pubs, &self.key.pub_hex(), &[]);
        let jurors = self.tree.recent_proposers(&head);
        let mut transfers = Vec::new();
        let mut data_txs = Vec::new();
        let pool: Vec<AccountTx> = self.account_pool.values().cloned().collect();
        for t in palimpsest_core::token::canonical_account_txs(&pool) {
            let ok = match &t {
                AccountTx::Transfer(x) => scratch.apply_transfer(x),
                _ => scratch.apply_data_tx(&t, hh + 1, &jurors),
            };
            if ok {
                match &t {
                    AccountTx::Transfer(_) =>
                        transfers.push(account_tx_to_json(&t)),
                    _ => data_txs.push(account_tx_to_json(&t)),
                }
            }
        }
        let core_transfers: Vec<_> = transfers.iter()
            .filter_map(|v| match account_tx_from_json(v) {
                Some(AccountTx::Transfer(x)) => Some(x),
                _ => None,
            }).collect();
        let core_data: Vec<AccountTx> = data_txs.iter()
            .filter_map(account_tx_from_json).collect();
        let header = core::Header {
            height: hh + 1,
            prev_hash: head.clone(),
            state_root: core::state_root(&w),
            txset_root: core::txset_root(
                &chosen.iter().map(|t| t.txid()).collect::<Vec<_>>()),
            n_txs: chosen.len() as u64,
            work: chosen.len() as u64 * 1000,
            proposer: self.key.pub_hex(),
            transfer_root: palimpsest_core::token::transfer_root(&core_transfers),
            ledger_root: scratch.root(),
            data_root: palimpsest_core::token::data_root(&core_data),
        };
        let stored = StoredBlock {
            header: WireHeader::from_core(&header),
            txs: chosen.iter().map(WireDeltaTx::from_core).collect(),
            transfers,
            data_txs,
        };
        let mut bodies = HashMap::new();
        for (t, d) in chosen.iter().zip(deltas) {
            bodies.insert(t.da_pointer.clone(), d);
        }
        let block = palimpsest_core::blocktree::Block {
            header, txs: chosen, bodies,
            transfers: core_transfers, data_txs: core_data,
        };
        Some((stored, block))
    }

    // ---- installation ----------------------------------------------------
    /// Try to install a stored block (bodies from the payload store). Returns
    /// true if installed; queues it as pending when payloads are missing.
    fn install(&mut self, sb: StoredBlock, from: Option<PeerId>,
               swarm: &mut Swarm<Behaviour>) -> bool {
        let bh = sb.hash();
        if self.blocks_full.contains_key(&bh) {
            return false;
        }
        let Some(block) = sb.to_core(&self.payloads) else {
            if let Some(peer) = from {
                let req = SyncRequest {
                    from_height: sb.header.height.saturating_sub(1),
                    count: 32,
                };
                swarm.behaviour_mut().sync.send_request(&peer, req);
                self.queue_pending(bh, sb, peer);
            }
            return false;
        };
        let old_head = self.tree.head.clone();
        match self.tree.add_block(block) {
            Ok(_) => {
                let _ = self.store.append_block(&sb);
                for t in &sb.txs {
                    if let Some(tc) = t.to_core() {
                        let id = tc.txid();
                        self.delta_pool.remove(&id);
                        // the payload is now persisted inside the block (on disk);
                        // drop the in-memory copy — sync/replay read it from disk.
                        self.payloads.remove(&id);
                    }
                }
                for v in sb.transfers.iter().chain(sb.data_txs.iter()) {
                    if let Some(t) = account_tx_from_json(v) {
                        self.account_pool.remove(&t.txid());
                        self.mark_seen(t.txid());
                    }
                }
                self.blocks_full.insert(bh.clone(), sb);
                if self.tree.head != old_head {
                    self.on_head_advance(&old_head);
                }
                true
            }
            Err(e) => {
                if e.0.contains("orphan") {
                    if let Some(peer) = from {
                        let req = SyncRequest {
                            from_height: self.head_height().saturating_sub(8),
                            count: 64,
                        };
                        swarm.behaviour_mut().sync.send_request(&peer, req);
                        self.queue_pending(bh, sb, peer);
                    }
                } else {
                    warn!("invalid block h{}: {}", sb.header.height, e.0);
                }
                false
            }
        }
    }

    fn retry_pending(&mut self, swarm: &mut Swarm<Behaviour>) {
        let ready: Vec<String> = self.pending.iter()
            .filter(|(_, (sb, _))| {
                sb.txs.iter().all(|t| t.to_core()
                    .map(|tc| self.payloads.contains_key(&tc.txid()))
                    .unwrap_or(false))
                && self.tree.blocks.contains_key(&sb.header.prev_hash)
            })
            .map(|(h, _)| h.clone()).collect();
        for h in ready {
            if let Some((sb, peer)) = self.pending.remove(&h) {
                self.install(sb, Some(peer), swarm);
            }
        }
    }

    fn on_head_advance(&mut self, old_head: &str) {
        let h = self.head_height();
        info!(height = h, head = &self.tree.head[..10],
              supply = self.tree.head_ledger().supply(), "head advanced");
        // keep the bridge synced with a sparse state diff
        if self.bridge_synced {
            if let (Some(new_w), Some(old_w)) =
                (self.tree.state.get(&self.tree.head), self.tree.state.get(old_head))
            {
                let diff: Vec<i64> = new_w.iter().zip(old_w)
                    .map(|(a, b)| a - b).collect();
                let sparse = Payload::from_dense_i64(&diff);
                let _ = self.bridge_tx.try_send(ToBridge::Advance { height: h, sparse });
            } else {
                // reorg past pruned state — bridge must resync from scratch
                self.send_bridge_state();
            }
        }
        if h % SNAPSHOT_EVERY == 0 {
            self.store.write_snapshot(&self.tree.head, h,
                                      &self.tree.state[&self.tree.head],
                                      self.tree.head_ledger());
        }
        // the head moved: prune mempools + pending against it
        self.evict_delta_pool();
        self.evict_account_pool();
        let drop_pending: Vec<String> = self.pending.iter()
            .filter(|(_, (s, _))| s.header.height + DELTA_STALE_SLACK < h)
            .map(|(k, _)| k.clone()).collect();
        for k in drop_pending {
            self.pending.remove(&k);
        }
    }

    fn send_bridge_state(&mut self) {
        let h = self.head_height();
        let state = self.tree.state[&self.tree.head].clone();
        if self.bridge_tx.try_send(ToBridge::State { height: h, state }).is_ok() {
            self.bridge_synced = true;
        }
    }

    // ---- api -------------------------------------------------------------
    fn api_status(&self) -> Value {
        let led = self.tree.head_ledger();
        json!({
            "height": self.head_height(),
            "head": &self.tree.head[..16],
            "supply": led.supply(),
            "delta_pool": self.delta_pool.len(),
            "account_pool": self.account_pool.len(),
            "pending_blocks": self.pending.len(),
            "producer": self.cfg.produce,
            "miner": self.key.pub_hex(),
            "peers": self.peers_connected,
            "model_attached": self.bridge_synced,
        })
    }

    fn api_balance(&self, addr: &str) -> Value {
        let led = self.tree.head_ledger();
        json!({"addr": addr, "grains": led.balance(addr),
               "nonce": led.nonces.get(addr).copied().unwrap_or(0),
               "supply": led.supply(), "height": self.head_height()})
    }

    fn api_registry(&self) -> Value {
        let led = self.tree.head_ledger();
        json!({"registry": led.registry, "challenges": led.challenges})
    }

    fn api_miners(&self) -> Value {
        // work accounting straight from chain history: for every miner ever
        // seen, blocks proposed, deltas contributed, tokens earned, last height
        use palimpsest_core::token::address;
        let mut stats: HashMap<String, (u64, u64, u64)> = HashMap::new(); // pub -> (proposed, deltas, last_h)
        for sb in self.blocks_full.values() {
            let h = sb.header.height;
            if sb.header.proposer != "genesis" {
                let e = stats.entry(sb.header.proposer.clone()).or_default();
                e.0 += 1;
                e.2 = e.2.max(h);
            }
            for t in &sb.txs {
                let e = stats.entry(t.miner.clone()).or_default();
                e.1 += 1;
                e.2 = e.2.max(h);
            }
        }
        let led = self.tree.head_ledger();
        let total_blocks = self.head_height().max(1);
        let mut miners: Vec<Value> = stats.into_iter().map(|(pub_hex, (p, d, lh))| {
            let addr = address(&pub_hex);
            json!({"miner": pub_hex, "address": addr,
                   "blocks_proposed": p, "deltas": d, "last_height": lh,
                   "balance": led.balance(&addr),
                   "share_pct": (p as f64 * 100.0 / total_blocks as f64).round(),
                   "is_me": pub_hex == self.key.pub_hex()})
        }).collect();
        miners.sort_by_key(|m| std::cmp::Reverse(m["blocks_proposed"].as_u64().unwrap_or(0)));
        json!({"miners": miners, "peers_connected": self.peers_connected,
               "head_height": self.head_height()})
    }

    fn api_upload(&mut self, bytes: Vec<u8>, stake: u64, media: String) -> (Value, Option<Gossip>) {
        use palimpsest_core::token::{address, AccountTx, DataSubmitTx};
        if bytes.is_empty() {
            return (json!({"ok": false, "error": "empty file"}), None);
        }
        let hash = core::delta_hash(&bytes);
        if let Err(e) = self.store.save_upload(&hash, &bytes) {
            return (json!({"ok": false, "error": format!("store: {e}")}), None);
        }
        let led = self.tree.head_ledger();
        let my_addr = address(&self.key.pub_hex());
        if led.balance(&my_addr) < stake {
            return (json!({"ok": false,
                "error": format!("node wallet balance {} < stake {}",
                                 led.balance(&my_addr), stake),
                "data_hash": hash,
                "hint": "file is custodied; submit on-chain from a funded wallet \
                         with: wallet submit-data"}), None);
        }
        let mut tx = DataSubmitTx {
            owner_pub: self.key.pub_hex(),
            data_hash: hash.clone(),
            size_bytes: bytes.len() as u64,
            media_type: media,
            stake,
            nonce: *led.nonces.get(&my_addr).unwrap_or(&0),
            sig: vec![],
        };
        tx.sig = self.key.sign(&AccountTx::DataSubmit(tx.clone()).signing_bytes());
        let atx = AccountTx::DataSubmit(tx);
        match self.accept_account_tx(atx.clone()) {
            Some(txid) => (json!({"ok": true, "txid": txid, "data_hash": hash,
                                  "bytes": bytes.len(),
                                  "status": "custodied + staked submission in mempool"}),
                           Some(Gossip::Atx { tx: account_tx_to_json(&atx) })),
            None => (json!({"ok": false, "error": "tx rejected (duplicate?)",
                            "data_hash": hash}), None),
        }
    }

    fn api_chain(&self) -> Value {
        // the last 16 headers along the head lineage, oldest first
        let mut out = Vec::new();
        let mut cur = self.tree.head.clone();
        for _ in 0..16 {
            if cur == self.tree.genesis_hash {
                break;
            }
            let h = &self.tree.blocks[&cur];
            out.push(json!({"height": h.height, "hash": cur,
                            "proposer": h.proposer, "n_txs": h.n_txs,
                            "work": h.work}));
            cur = h.prev_hash.clone();
        }
        out.reverse();
        json!({"blocks": out})
    }
}

// ---------------------------------------------------------------------------
// The main loop
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub async fn run(
    mut node: Node,
    mut swarm: Swarm<Behaviour>,
    mut api_rx: mpsc::Receiver<ApiCmd>,
    mut bridge_rx: mpsc::Receiver<FromBridge>,
) {
    let end = if node.cfg.seconds > 0.0 { now() + node.cfg.seconds } else { f64::MAX };
    let mut tick = tokio::time::interval(Duration::from_millis(400));
    let jitter: f64 = rand::random::<f64>() * 0.5;

    loop {
        if now() >= end {
            break;
        }
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("shutdown signal");
                break;
            }
            _ = tick.tick() => {
                let round = ((now() - node.t0 - jitter) / node.cfg.interval).floor() as i64;
                if round >= 0 && round != node.last_announced_round {
                    node.last_announced_round = round;
                    // the self-healing heartbeat: announce our head every round
                    let head_msg = Gossip::Head {
                        hash: node.tree.head.clone(),
                        height: node.head_height(),
                    };
                    node.publish(&mut swarm, &head_msg);
                    // …and re-dial configured peers when connections are lost
                    // (restarts on either side otherwise orphan the mesh forever)
                    let expected = node.cfg.peers.split(',')
                        .filter(|s| !s.is_empty()).count();
                    if swarm.network_info().num_peers() < expected {
                        dial_peers(&mut swarm, &node.cfg.peers.clone());
                    }
                }
                if node.cfg.produce && round >= 0 && round != node.last_proposed_round {
                    node.last_proposed_round = round;
                    // republish unconfirmed deltas for the current height: a
                    // publish can silently fail before the gossip mesh forms
                    // (InsufficientPeers), so retry each round until included
                    let hh = node.head_height();
                    let resend: Vec<(WireDeltaTx, Payload)> = node.delta_pool.values()
                        .filter(|t| t.base_height == hh)
                        .filter_map(|t| node.payloads.get(&t.txid())
                            .map(|p| (WireDeltaTx::from_core(t), p.clone())))
                        .collect();
                    for (tx, payload) in resend {
                        node.publish(&mut swarm, &Gossip::Dtx { tx, payload });
                    }
                    // train EVERY round (the delta gossips to whoever proposes);
                    // proposing itself may rotate (devnet) or be open (mainnet)
                    if node.bridge_synced && !node.train_inflight {
                        node.train_inflight = true;
                        let _ = node.bridge_tx.try_send(ToBridge::Train {
                            height: node.head_height(),
                            seed: round as u64,
                        });
                    }
                    let my_turn = match node.cfg.rotate {
                        Some((n, id)) => (round as u64) % n == id,
                        None => true,                   // open proposing; fork choice settles
                    };
                    if my_turn {
                        if let Some((stored, block)) = node.build_candidate() {
                            let bh = stored.hash();
                            match node.tree.add_block(block) {
                                Ok(_) => {
                                    let _ = node.store.append_block(&stored);
                                    for t in &stored.txs {
                                        if let Some(tc) = t.to_core() {
                                            node.delta_pool.remove(&tc.txid());
                                        }
                                    }
                                    for v in stored.transfers.iter()
                                            .chain(stored.data_txs.iter()) {
                                        if let Some(t) = account_tx_from_json(v) {
                                            node.account_pool.remove(&t.txid());
                                        }
                                    }
                                    node.blocks_full.insert(bh, stored.clone());
                                    let old = stored.header.prev_hash.clone();
                                    node.on_head_advance(&old);
                                    node.publish(&mut swarm, &Gossip::Blk { block: stored });
                                }
                                Err(e) => warn!("own block rejected: {}", e.0),
                            }
                        }
                    }
                }
            }
            Some(ev) = bridge_rx.recv() => match ev {
                FromBridge::Connected | FromBridge::NeedState => {
                    node.train_inflight = false;
                    node.chat_inflight = false;
                    for tx in node.chat_pending.drain(..) {
                        let _ = tx.send(json!({"ok": false,
                            "error": "model reconnected — try again"}));
                    }
                    node.send_bridge_state();
                }
                FromBridge::Generated { text, height } => {
                    node.chat_inflight = false;
                    if let Some(tx) = node.chat_pending.pop() {
                        let _ = tx.send(json!({"ok": true, "reply": text,
                                               "height": height}));
                    }
                }
                FromBridge::Delta { height, loss, payload } => {
                    node.train_inflight = false;
                    if height != node.head_height() {
                        debug!("stale delta (h{height} vs h{})", node.head_height());
                    } else {
                        let dense = payload.dense().unwrap_or_default();
                        let dh = core::delta_hash(&core::int64_bytes(&dense));
                        let mut tx = core::BackpropTx {
                            miner: node.key.pub_hex(),
                            base_height: height,
                            shard_id: 0,
                            delta_hash: dh.clone(),
                            da_pointer: format!("da://{dh}"),
                            sig: vec![],
                        };
                        tx.sig = node.key.sign(&tx.signing_bytes());
                        info!(height, loss, kb = payload.wire_bytes() / 1024,
                              "trained delta");
                        let wire = WireDeltaTx::from_core(&tx);
                        if node.accept_delta(tx, payload.clone()) {
                            node.publish(&mut swarm,
                                         &Gossip::Dtx { tx: wire, payload });
                        }
                    }
                }
            },
            Some(cmd) = api_rx.recv() => match cmd {
                ApiCmd::Status(o) => { let _ = o.send(node.api_status()); }
                ApiCmd::Balance(addr, o) => { let _ = o.send(node.api_balance(&addr)); }
                ApiCmd::Registry(o) => { let _ = o.send(node.api_registry()); }
                ApiCmd::Chain(o) => { let _ = o.send(node.api_chain()); }
                ApiCmd::Miners(o) => { let _ = o.send(node.api_miners()); }
                ApiCmd::Chat(prompt, o) => {
                    if !node.bridge_synced {
                        let _ = o.send(json!({"ok": false,
                            "error": "no model attached to this node yet"}));
                    } else if node.chat_inflight {
                        let _ = o.send(json!({"ok": false,
                            "error": "model is generating for someone else — try again"}));
                    } else {
                        node.chat_inflight = true;
                        node.chat_pending.push(o);
                        let _ = node.bridge_tx.try_send(ToBridge::Generate {
                            prompt, n: 120,
                        });
                    }
                }
                ApiCmd::Upload(bytes, stake, media, o) => {
                    let (reply, gossip) = node.api_upload(bytes, stake, media);
                    if let Some(msg) = gossip {
                        node.publish(&mut swarm, &msg);
                    }
                    let _ = o.send(reply);
                }
                ApiCmd::SubmitAccountTx(v, o) => {
                    let reply = match account_tx_from_json(&v) {
                        None => json!({"ok": false, "error": "malformed tx"}),
                        Some(tx) => match node.accept_account_tx(tx.clone()) {
                            None => json!({"ok": false,
                                           "error": "bad signature or duplicate"}),
                            Some(txid) => {
                                node.publish(&mut swarm,
                                             &Gossip::Atx { tx: account_tx_to_json(&tx) });
                                json!({"ok": true, "txid": txid,
                                       "status": "in mempool — settles in the next block"})
                            }
                        },
                    };
                    let _ = o.send(reply);
                }
            },
            ev = swarm.select_next_some() => match ev {
                SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(
                        gossipsub::Event::Message { message, propagation_source, .. })) => {
                    if let Ok(g) = serde_json::from_slice::<Gossip>(&message.data) {
                        match g {
                            Gossip::Dtx { tx, payload } => {
                                if let Some(t) = tx.to_core() {
                                    node.accept_delta(t, payload);
                                    node.retry_pending(&mut swarm);
                                }
                            }
                            Gossip::Atx { tx } => {
                                if let Some(t) = account_tx_from_json(&tx) {
                                    node.accept_account_tx(t);
                                }
                            }
                            Gossip::Blk { block } => {
                                node.install(block, Some(propagation_source), &mut swarm);
                                node.retry_pending(&mut swarm);
                            }
                            Gossip::Head { hash, height } => {
                                // unknown head -> pull the sender's recent chain,
                                // BUT at most one in-flight catch-up per peer per
                                // 90s — payload batches are tens of MB and stacked
                                // transfers saturate home uplinks without landing
                                let recent = node.last_sync_req
                                    .get(&propagation_source)
                                    .map(|t| now() - t < 90.0)
                                    .unwrap_or(false);
                                if !node.tree.blocks.contains_key(&hash) && !recent {
                                    node.last_sync_req
                                        .insert(propagation_source, now());
                                    let from = node.head_height()
                                        .min(height).saturating_sub(2);
                                    info!(peer = %propagation_source, their_h = height,
                                          from, "unknown head — requesting sync");
                                    let req = SyncRequest { from_height: from, count: 8 };
                                    swarm.behaviour_mut().sync
                                        .send_request(&propagation_source, req);
                                }
                            }
                        }
                    }
                }
                SwarmEvent::Behaviour(BehaviourEvent::Sync(
                        request_response::Event::Message { message, .. })) => {
                    match message {
                        request_response::Message::Request { request, channel, .. } => {
                            // serve blocks along OUR head chain in the range;
                            // cap the batch — payload-heavy responses at real
                            // model scale (~18MB/block) OOM small peers and
                            // drown slow uplinks
                            let count = request.count.min(2);
                            let mut chain = Vec::new();
                            let mut cur = node.tree.head.clone();
                            while cur != node.tree.genesis_hash {
                                let hdr = &node.tree.blocks[&cur];
                                if hdr.height < request.from_height {
                                    break;
                                }
                                if hdr.height < request.from_height + count {
                                    if let Some(sb) = node.blocks_full.get(&cur) {
                                        chain.push(sb.clone());
                                    }
                                }
                                cur = hdr.prev_hash.clone();
                            }
                            chain.reverse();
                            let mut payloads = HashMap::new();
                            for sb in &chain {
                                for t in &sb.txs {
                                    if let Some(tc) = t.to_core() {
                                        let txid = tc.txid();
                                        if let Some(p) = node.payloads.get(&txid)
                                            .cloned()
                                            .or_else(|| node.store.get_payload(&txid)) {
                                            payloads.insert(txid, p);
                                        }
                                    }
                                }
                            }
                            info!(from = request.from_height, served = chain.len(),
                                  "serving sync request");
                            let resp = SyncResponse {
                                blocks: chain, payloads,
                                head_height: node.head_height(),
                            };
                            let _ = swarm.behaviour_mut().sync
                                .send_response(channel, resp);
                        }
                        request_response::Message::Response { response, .. } => {
                            info!(blocks = response.blocks.len(),
                                  their_head = response.head_height,
                                  "sync response received");
                            for (txid, p) in response.payloads {
                                if !node.payloads.contains_key(&txid) {
                                    node.store.put_payload(&txid, &p);
                                    node.payloads.insert(txid, p);
                                }
                            }
                            for sb in response.blocks {
                                node.install(sb, None, &mut swarm);
                            }
                            node.retry_pending(&mut swarm);
                        }
                    }
                }
                SwarmEvent::NewListenAddr { address, .. } => {
                    info!(%address, "listening");
                }
                SwarmEvent::Behaviour(BehaviourEvent::Identify(
                        identify::Event::Received { peer_id, info, .. })) => {
                    debug!(%peer_id, agent = %info.agent_version, "peer identified");
                }
                SwarmEvent::ConnectionClosed { .. } => {
                    node.peers_connected = node.peers_connected.saturating_sub(1);
                }
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    node.peers_connected += 1;
                    info!(%peer_id, "peer connected");
                    // opportunistic catch-up from every new peer — anchor BELOW
                    // our head: an equal-height fork needs the peer's blocks at
                    // heights we already have, not just above them
                    let req = SyncRequest {
                        from_height: node.head_height().saturating_sub(8), count: 64,
                    };
                    swarm.behaviour_mut().sync.send_request(&peer_id, req);
                }
                _ => {}
            },
        }
    }

    // final report + snapshot
    let h = node.head_height();
    node.store.write_snapshot(&node.tree.head, h,
                              &node.tree.state[&node.tree.head], node.tree.head_ledger());
    let mut lineage = Vec::new();
    let mut cur = node.tree.head.clone();
    while cur != node.tree.genesis_hash {
        lineage.push(cur[..6].to_string());
        cur = node.tree.blocks[&cur].prev_hash.clone();
    }
    lineage.reverse();
    println!("LINEAGE {}", lineage.join(">"));
    println!("done — height {} head {} supply {} ledger {}",
             h, &node.tree.head[..16], node.tree.head_ledger().supply(),
             &node.tree.head_ledger().root()[..12]);
}

/// Dial the configured peers (multiaddrs, comma-separated).
pub fn dial_peers(swarm: &mut Swarm<Behaviour>, peers: &str) {
    for p in peers.split(',').filter(|s| !s.is_empty()) {
        match p.parse::<Multiaddr>() {
            Ok(addr) => {
                if let Err(e) = swarm.dial(addr) {
                    warn!("dial {p}: {e}");
                }
            }
            Err(e) => warn!("bad multiaddr {p}: {e}"),
        }
    }
}
