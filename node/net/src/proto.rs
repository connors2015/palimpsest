//! Wire types and JSON codecs — gossip messages, sync request/response, and
//! the commitment-only block representation (bodies never ride in blocks; they
//! are reconstructed from the compressed payload store, exactly as the Python
//! client does).

use base64::Engine;
use palimpsest_core::{
    self as core,
    blocktree::Block,
    token::{AccountTx, DataChallengeTx, DataSubmitTx, DataVoteTx, TransferTx},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn b64(v: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(v)
}

pub fn unb64(s: &str) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD.decode(s).ok()
}

// ---------------------------------------------------------------------------
// Compressed delta payloads (top-k sparse, the transmission form; densifies to
// the exact int64 vector the chain commits to)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Payload {
    pub n: usize,        // dense length
    pub idx: String,     // base64 of u32-LE indices
    pub val: String,     // base64 of i32-LE values
}

impl Payload {
    pub fn dense(&self) -> Option<Vec<i64>> {
        let idx = unb64(&self.idx)?;
        let val = unb64(&self.val)?;
        Some(core::decompress(self.n, &idx, &val))
    }

    pub fn wire_bytes(&self) -> usize {
        self.idx.len() * 3 / 4 + self.val.len() * 3 / 4
    }

    /// Sparse-encode an int64 vector (used for aggregate deltas to the bridge —
    /// values may exceed i32 so this variant carries i64 values).
    pub fn from_dense_i64(v: &[i64]) -> SparseI64 {
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (i, x) in v.iter().enumerate() {
            if *x != 0 {
                idx.extend_from_slice(&(i as u32).to_le_bytes());
                val.extend_from_slice(&x.to_le_bytes());
            }
        }
        SparseI64 { n: v.len(), idx: b64(&idx), val: b64(&val) }
    }
}

/// Sparse vector with full i64 values (aggregates; bridge advances).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SparseI64 {
    pub n: usize,
    pub idx: String,
    pub val: String,
}

// ---------------------------------------------------------------------------
// Commitment-only stored blocks (what gossips, persists, and syncs)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WireHeader {
    pub height: u64,
    pub prev_hash: String,
    pub state_root: String,
    pub txset_root: String,
    pub n_txs: u64,
    pub work: u64,
    pub proposer: String,
    pub transfer_root: String,
    pub ledger_root: String,
    pub data_root: String,
    #[serde(default)]
    pub vrf_proof: String,
}

impl WireHeader {
    pub fn to_core(&self) -> core::Header {
        core::Header {
            height: self.height,
            prev_hash: self.prev_hash.clone(),
            state_root: self.state_root.clone(),
            txset_root: self.txset_root.clone(),
            n_txs: self.n_txs,
            work: self.work,
            proposer: self.proposer.clone(),
            transfer_root: self.transfer_root.clone(),
            ledger_root: self.ledger_root.clone(),
            data_root: self.data_root.clone(),
            vrf_proof: self.vrf_proof.clone(),
        }
    }

    pub fn from_core(h: &core::Header) -> Self {
        WireHeader {
            height: h.height,
            prev_hash: h.prev_hash.clone(),
            state_root: h.state_root.clone(),
            txset_root: h.txset_root.clone(),
            n_txs: h.n_txs,
            work: h.work,
            proposer: h.proposer.clone(),
            transfer_root: h.transfer_root.clone(),
            ledger_root: h.ledger_root.clone(),
            data_root: h.data_root.clone(),
            vrf_proof: h.vrf_proof.clone(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WireDeltaTx {
    pub miner: String,
    pub base_height: u64,
    pub shard_id: u64,
    pub delta_hash: String,
    pub da_pointer: String,
    #[serde(default)]
    pub bond: u64,
    pub sig: String, // hex
}

impl WireDeltaTx {
    pub fn to_core(&self) -> Option<core::BackpropTx> {
        Some(core::BackpropTx {
            miner: self.miner.clone(),
            base_height: self.base_height,
            shard_id: self.shard_id,
            delta_hash: self.delta_hash.clone(),
            da_pointer: self.da_pointer.clone(),
            bond: self.bond,
            sig: hex::decode(&self.sig).ok()?,
        })
    }

    pub fn from_core(t: &core::BackpropTx) -> Self {
        WireDeltaTx {
            miner: t.miner.clone(),
            base_height: t.base_height,
            shard_id: t.shard_id,
            delta_hash: t.delta_hash.clone(),
            da_pointer: t.da_pointer.clone(),
            bond: t.bond,
            sig: hex::encode(&t.sig),
        }
    }
}

/// Account txs on the wire: tagged JSON.
pub fn account_tx_to_json(t: &AccountTx) -> Value {
    match t {
        AccountTx::Transfer(x) => json!({"kind": "transfer", "from_pub": x.from_pub,
            "to_addr": x.to_addr, "amount": x.amount, "nonce": x.nonce,
            "sig": hex::encode(&x.sig)}),
        AccountTx::DataSubmit(x) => json!({"kind": "data_submit", "owner_pub": x.owner_pub,
            "data_hash": x.data_hash, "size_bytes": x.size_bytes,
            "media_type": x.media_type, "stake": x.stake, "nonce": x.nonce,
            "sig": hex::encode(&x.sig)}),
        AccountTx::DataChallenge(x) => json!({"kind": "data_challenge",
            "challenger_pub": x.challenger_pub, "data_id": x.data_id,
            "stake": x.stake, "reason": x.reason, "nonce": x.nonce,
            "sig": hex::encode(&x.sig)}),
        AccountTx::DataVote(x) => json!({"kind": "data_vote", "voter_pub": x.voter_pub,
            "challenge_id": x.challenge_id, "support": x.support, "nonce": x.nonce,
            "sig": hex::encode(&x.sig)}),
    }
}

pub fn account_tx_from_json(v: &Value) -> Option<AccountTx> {
    let sig = hex::decode(v["sig"].as_str()?).ok()?;
    Some(match v["kind"].as_str()? {
        "transfer" => AccountTx::Transfer(TransferTx {
            from_pub: v["from_pub"].as_str()?.into(),
            to_addr: v["to_addr"].as_str()?.into(),
            amount: v["amount"].as_u64()?,
            nonce: v["nonce"].as_u64()?, sig,
        }),
        "data_submit" => AccountTx::DataSubmit(DataSubmitTx {
            owner_pub: v["owner_pub"].as_str()?.into(),
            data_hash: v["data_hash"].as_str()?.into(),
            size_bytes: v["size_bytes"].as_u64()?,
            media_type: v["media_type"].as_str()?.into(),
            stake: v["stake"].as_u64()?,
            nonce: v["nonce"].as_u64()?, sig,
        }),
        "data_challenge" => AccountTx::DataChallenge(DataChallengeTx {
            challenger_pub: v["challenger_pub"].as_str()?.into(),
            data_id: v["data_id"].as_str()?.into(),
            stake: v["stake"].as_u64()?,
            reason: v["reason"].as_str()?.into(),
            nonce: v["nonce"].as_u64()?, sig,
        }),
        "data_vote" => AccountTx::DataVote(DataVoteTx {
            voter_pub: v["voter_pub"].as_str()?.into(),
            challenge_id: v["challenge_id"].as_str()?.into(),
            support: v["support"].as_bool()?,
            nonce: v["nonce"].as_u64()?, sig,
        }),
        _ => return None,
    })
}

/// A block as gossiped/stored: commitments only, no bodies.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StoredBlock {
    pub header: WireHeader,
    pub txs: Vec<WireDeltaTx>,
    pub transfers: Vec<Value>, // account-tx JSON (kind=transfer)
    pub data_txs: Vec<Value>,  // account-tx JSON (data lane)
}

impl StoredBlock {
    pub fn hash(&self) -> String {
        self.header.to_core().block_hash()
    }

    /// Materialize a validatable core Block by densifying bodies from payloads.
    /// Returns None if any payload is missing or malformed.
    pub fn to_core(&self, payloads: &HashMap<String, Payload>) -> Option<Block> {
        let mut txs = Vec::new();
        let mut bodies = HashMap::new();
        for wt in &self.txs {
            let t = wt.to_core()?;
            let p = payloads.get(&t.txid())?;
            bodies.insert(t.da_pointer.clone(), p.dense()?);
            txs.push(t);
        }
        let mut transfers = Vec::new();
        for v in &self.transfers {
            match account_tx_from_json(v)? {
                AccountTx::Transfer(t) => transfers.push(t),
                _ => return None,
            }
        }
        let mut data_txs = Vec::new();
        for v in &self.data_txs {
            data_txs.push(account_tx_from_json(v)?);
        }
        Some(Block { header: self.header.to_core(), txs, bodies, transfers, data_txs })
    }
}

// ---------------------------------------------------------------------------
// Gossip envelope + sync protocol
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(tag = "t")]
pub enum Gossip {
    /// A miner's delta commitment + its compressed payload.
    Dtx { tx: WireDeltaTx, payload: Payload },
    /// An account tx (transfer or data lane).
    Atx { tx: Value },
    /// A block — commitments only.
    Blk { block: StoredBlock },
    /// Tiny per-round head announcement — the self-healing heartbeat: any node
    /// seeing an unknown head syncs from its sender, so divergence resolves
    /// within a round regardless of what earlier messages were lost.
    Head { hash: String, height: u64 },
}

/// Range chain sync over libp2p request-response (JSON codec).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SyncRequest {
    pub from_height: u64,
    pub count: u64,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SyncResponse {
    pub blocks: Vec<StoredBlock>,
    pub payloads: HashMap<String, Payload>, // txid -> payload for those blocks
    pub head_height: u64,
}

/// Data-availability shard exchange (§3.3). A node missing a body asks peers for
/// its erasure shards; each peer returns whatever shards it holds. The requester
/// gathers K across peers and reconstructs — so a body stays recoverable even
/// when no single node retains it whole.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ShardRequest {
    pub txids: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ShardResponse {
    pub bodies: Vec<BodyShards>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BodyShards {
    pub txid: String,
    pub k: u32,
    pub n: u32,
    pub orig_len: u64,
    pub shards: Vec<(u32, String)>, // (index, base64 shard bytes)
}
