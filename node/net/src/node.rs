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
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub const INCLUDE_K: usize = 8;

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
        self.seen.insert(txid.clone());
        self.store.put_payload(&txid, &payload);
        self.payloads.insert(txid.clone(), payload);
        self.delta_pool.insert(txid, tx);
        true
    }

    fn accept_account_tx(&mut self, tx: AccountTx) -> Option<String> {
        let txid = tx.txid();
        if self.seen.contains(&txid) || !tx.verify() {
            return None;
        }
        self.seen.insert(txid.clone());
        self.account_pool.insert(txid.clone(), tx);
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
        let w: Vec<i64> = parent_w.iter().zip(&mean).map(|(a, b)| a + b).collect();
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
                self.pending.insert(bh, (sb, peer));
            }
            return false;
        };
        let old_head = self.tree.head.clone();
        match self.tree.add_block(block) {
            Ok(_) => {
                let _ = self.store.append_block(&sb);
                for t in &sb.txs {
                    if let Some(tc) = t.to_core() {
                        self.delta_pool.remove(&tc.txid());
                    }
                }
                for v in sb.transfers.iter().chain(sb.data_txs.iter()) {
                    if let Some(t) = account_tx_from_json(v) {
                        self.account_pool.remove(&t.txid());
                        self.seen.insert(t.txid());
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
                        self.pending.insert(bh, (sb, peer));
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
                                      &self.tree.state[&self.tree.head]);
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
                    node.send_bridge_state();
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
                            let resp = SyncResponse {
                                blocks: chain, payloads,
                                head_height: node.head_height(),
                            };
                            let _ = swarm.behaviour_mut().sync
                                .send_response(channel, resp);
                        }
                        request_response::Message::Response { response, .. } => {
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
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
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
    node.store.write_snapshot(&node.tree.head, h, &node.tree.state[&node.tree.head]);
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
