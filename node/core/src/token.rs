//! The native token ledger — chain state, mirroring `rig/token.py` bit-exactly.
//!
//! State maps use BTreeMap (sorted keys) and the registry/challenge entries are
//! serde_json Values, so the canonical ledger root — Python's
//! `json.dumps(state, sort_keys=True, separators=(",",":"))` — falls out of
//! `serde_json::to_string` structurally (serde_json's default Map is a BTreeMap).

use crate::verify_sig;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const GRAIN: u64 = 1_000_000_000;
pub const BASE_REWARD: u64 = 50 * GRAIN;
pub const HALVING_BLOCKS: u64 = 100_000;
pub const SUNSET_HEIGHT: u64 = 1_000_000;
pub const SHARE_MINERS: u64 = 7_000;
pub const SHARE_PROPOSER: u64 = 1_000;
pub const SHARE_DATA: u64 = 2_000;
pub const CHALLENGE_WINDOW: u64 = 20;
pub const PROPOSER_LOOKBACK: usize = 32;
pub const GENESIS_DATA_WEIGHT: u64 = 1_000_000;
/// Minimum affirmative juror votes to uphold a challenge — one juror must never
/// be able to seize an owner's stake; below quorum the challenge is rejected.
pub const CHALLENGE_QUORUM: usize = 3;

/// Wallet address: sha256 of the raw pubkey bytes, first 20 bytes, hex.
pub fn address(pub_hex: &str) -> String {
    let bytes = hex::decode(pub_hex).unwrap_or_default();
    hex::encode(&Sha256::digest(&bytes)[..20])
}

/// Deterministic block reward: halves every HALVING_BLOCKS, zero at/after sunset.
pub fn emission(height: u64) -> u64 {
    if height < 1 || height >= SUNSET_HEIGHT {
        return 0;
    }
    BASE_REWARD >> ((height - 1) / HALVING_BLOCKS)
}

// ---------------------------------------------------------------------------
// Account transactions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TransferTx {
    pub from_pub: String,
    pub to_addr: String,
    pub amount: u64,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DataSubmitTx {
    pub owner_pub: String,
    pub data_hash: String,
    pub size_bytes: u64,
    pub media_type: String,
    pub stake: u64,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DataChallengeTx {
    pub challenger_pub: String,
    pub data_id: String,
    pub stake: u64,
    pub reason: String,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct DataVoteTx {
    pub voter_pub: String,
    pub challenge_id: String,
    pub support: bool,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

/// The merged account-tx lane: one nonce sequence per wallet totally orders
/// everything it does.
#[derive(Clone, Debug)]
pub enum AccountTx {
    Transfer(TransferTx),
    DataSubmit(DataSubmitTx),
    DataChallenge(DataChallengeTx),
    DataVote(DataVoteTx),
}

impl TransferTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        crate::frame(&[b"transfer", self.from_pub.as_bytes(), self.to_addr.as_bytes(),
                       self.amount.to_string().as_bytes(), self.nonce.to_string().as_bytes()])
    }
}

impl DataSubmitTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        crate::frame(&[b"data_submit", self.owner_pub.as_bytes(), self.data_hash.as_bytes(),
                       self.size_bytes.to_string().as_bytes(), self.media_type.as_bytes(),
                       self.stake.to_string().as_bytes(), self.nonce.to_string().as_bytes()])
    }
}

impl DataChallengeTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        crate::frame(&[b"data_challenge", self.challenger_pub.as_bytes(), self.data_id.as_bytes(),
                       self.stake.to_string().as_bytes(), self.reason.as_bytes(),
                       self.nonce.to_string().as_bytes()])
    }
}

impl DataVoteTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        let support = if self.support { 1u8 } else { 0u8 };
        crate::frame(&[b"data_vote", self.voter_pub.as_bytes(), self.challenge_id.as_bytes(),
                       support.to_string().as_bytes(), self.nonce.to_string().as_bytes()])
    }
}

impl AccountTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        match self {
            AccountTx::Transfer(t) => t.signing_bytes(),
            AccountTx::DataSubmit(t) => t.signing_bytes(),
            AccountTx::DataChallenge(t) => t.signing_bytes(),
            AccountTx::DataVote(t) => t.signing_bytes(),
        }
    }

    pub fn txid(&self) -> String {
        hex::encode(Sha256::digest(&self.signing_bytes()))
    }

    pub fn sender_pub(&self) -> &str {
        match self {
            AccountTx::Transfer(t) => &t.from_pub,
            AccountTx::DataSubmit(t) => &t.owner_pub,
            AccountTx::DataChallenge(t) => &t.challenger_pub,
            AccountTx::DataVote(t) => &t.voter_pub,
        }
    }

    pub fn nonce(&self) -> u64 {
        match self {
            AccountTx::Transfer(t) => t.nonce,
            AccountTx::DataSubmit(t) => t.nonce,
            AccountTx::DataChallenge(t) => t.nonce,
            AccountTx::DataVote(t) => t.nonce,
        }
    }

    pub fn sig(&self) -> &[u8] {
        match self {
            AccountTx::Transfer(t) => &t.sig,
            AccountTx::DataSubmit(t) => &t.sig,
            AccountTx::DataChallenge(t) => &t.sig,
            AccountTx::DataVote(t) => &t.sig,
        }
    }

    pub fn verify(&self) -> bool {
        verify_sig(self.sender_pub(), &self.signing_bytes(), self.sig())
    }
}

/// Consensus ordering of ALL account txs in a block: (sender address, nonce, txid).
pub fn canonical_account_txs(txs: &[AccountTx]) -> Vec<AccountTx> {
    let mut out = txs.to_vec();
    out.sort_by_key(|t| (address(t.sender_pub()), t.nonce(), t.txid()));
    out
}

fn set_root(txids: &mut Vec<String>) -> String {
    txids.sort();
    hex::encode(Sha256::digest(txids.join("|").as_bytes()))
}

/// Order-independent commitment to a transfer set.
pub fn transfer_root(transfers: &[TransferTx]) -> String {
    let mut ids: Vec<String> = transfers.iter()
        .map(|t| AccountTx::Transfer(t.clone()).txid()).collect();
    set_root(&mut ids)
}

/// Order-independent commitment to a data-lane tx set.
pub fn data_root(data_txs: &[AccountTx]) -> String {
    let mut ids: Vec<String> = data_txs.iter().map(|t| t.txid()).collect();
    set_root(&mut ids)
}

// ---------------------------------------------------------------------------
// The ledger
// ---------------------------------------------------------------------------

#[derive(Clone, Default, Debug)]
pub struct TokenLedger {
    pub balances: BTreeMap<String, u64>,
    pub nonces: BTreeMap<String, u64>,
    pub registry: BTreeMap<String, Value>,   // data_id -> entry object
    pub challenges: BTreeMap<String, Value>, // challenge_id -> challenge object
}

impl TokenLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seed_genesis_data(&mut self, owner_addr: &str) {
        self.registry.insert("genesis".into(), json!({
            "owner": owner_addr, "data_hash": "genesis", "size": 0,
            "media_type": "text", "stake": 0,
            "weight": GENESIS_DATA_WEIGHT, "status": "active"}));
    }

    pub fn balance(&self, addr: &str) -> u64 {
        *self.balances.get(addr).unwrap_or(&0)
    }

    fn credit(&mut self, addr: &str, amount: u64) {
        if amount > 0 {
            // saturating guard: total emission is capped far below u64::MAX
            // (SUNSET_HEIGHT × BASE_REWARD ≈ 5e16 ≪ 1.8e19), so a single balance
            // can never actually reach the ceiling — this is a can't-happen
            // defense that stays deterministic instead of panicking/wrapping if
            // the emission constants ever change.
            let bal = self.balances.entry(addr.to_string()).or_insert(0);
            *bal = bal.saturating_add(amount);
        }
    }

    /// Mint + split the block reward; data share across the weighted registry.
    ///
    /// The three shares (miners / proposer / data) and the per-recipient splits
    /// are floor divisions; the remainders (dust) are DELIBERATELY not minted, so
    /// realized supply is always ≤ the emission schedule, never above it. This is
    /// deterministic (identical on every node) and is the reason `supply()` runs a
    /// hair under the nominal curve — a known, documented property, not a leak.
    pub fn apply_reward(&mut self, height: u64, miner_pubs: &[String],
                        proposer_pub: &str, legacy_data_addrs: &[String]) {
        let total = emission(height);
        if total == 0 {
            return;
        }
        let miners_pool = total * SHARE_MINERS / 10_000;
        let proposer_cut = total * SHARE_PROPOSER / 10_000;
        let data_pool = total * SHARE_DATA / 10_000;
        if !miner_pubs.is_empty() {
            let each = miners_pool / miner_pubs.len() as u64;
            let mut sorted: Vec<&String> = miner_pubs.iter().collect();
            sorted.sort();
            for p in sorted {
                let a = address(p);
                self.credit(&a, each);
            }
        }
        if !proposer_pub.is_empty() && proposer_pub != "genesis" {
            let a = address(proposer_pub);
            self.credit(&a, proposer_cut);
        }
        let active: Vec<(String, u64)> = self.registry.iter()
            .filter(|(_, e)| e["status"] == "active"
                    && e["weight"].as_u64().unwrap_or(0) > 0)
            .map(|(_, e)| (e["owner"].as_str().unwrap().to_string(),
                           e["weight"].as_u64().unwrap()))
            .collect();
        if !active.is_empty() {
            let wsum: u128 = active.iter().map(|(_, w)| *w as u128).sum();
            for (owner, w) in active {
                // u128 intermediate: pool×weight can exceed u64 (Python bigints
                // don't overflow; the floor-divided result always fits u64)
                let share = (data_pool as u128 * w as u128 / wsum) as u64;
                self.credit(&owner, share);
            }
        } else if !legacy_data_addrs.is_empty() {
            let each = data_pool / legacy_data_addrs.len() as u64;
            let mut sorted: Vec<&String> = legacy_data_addrs.iter().collect();
            sorted.sort();
            for a in sorted {
                self.credit(a, each);
            }
        }
    }

    /// Settle every expired challenge (sorted id order) — FIRST step per block.
    pub fn resolve_expired_challenges(&mut self, height: u64) {
        let ids: Vec<String> = self.challenges.keys().cloned().collect();
        for cid in ids {
            let ch = self.challenges[&cid].clone();
            if ch["expiry"].as_u64().unwrap() > height {
                continue;
            }
            let data_id = ch["data_id"].as_str().unwrap().to_string();
            // QUORUM: upheld only with a strict majority AND at least
            // CHALLENGE_QUORUM affirmative juror votes; below quorum → rejected.
            let vf = ch["votes_for"].as_array().unwrap().len();
            let va = ch["votes_against"].as_array().unwrap().len();
            let upheld = vf >= CHALLENGE_QUORUM && vf > va;
            if let Some(entry) = self.registry.get_mut(&data_id) {
                if upheld {
                    let stake = entry["stake"].as_u64().unwrap();
                    entry["status"] = json!("revoked");
                    entry["stake"] = json!(0);
                    let challenger = ch["challenger"].as_str().unwrap().to_string();
                    self.credit(&challenger, stake.saturating_add(ch["stake"].as_u64().unwrap()));
                } else {
                    let owner = entry["owner"].as_str().unwrap().to_string();
                    self.credit(&owner, ch["stake"].as_u64().unwrap());
                }
            }
            self.challenges.remove(&cid);
        }
    }

    pub fn apply_transfer(&mut self, tx: &TransferTx) -> bool {
        let atx = AccountTx::Transfer(tx.clone());
        if !atx.verify() || tx.amount == 0 {
            return false;
        }
        let src = address(&tx.from_pub);
        if tx.nonce != *self.nonces.get(&src).unwrap_or(&0)
            || self.balance(&src) < tx.amount {
            return false;
        }
        *self.balances.get_mut(&src).unwrap() -= tx.amount;
        self.credit(&tx.to_addr, tx.amount);
        self.nonces.insert(src, tx.nonce + 1);
        true
    }

    pub fn apply_data_tx(&mut self, tx: &AccountTx, height: u64,
                         recent_proposers: &std::collections::HashSet<String>) -> bool {
        if !tx.verify() {
            return false;
        }
        let src = address(tx.sender_pub());
        if tx.nonce() != *self.nonces.get(&src).unwrap_or(&0) {
            return false;
        }
        match tx {
            AccountTx::DataSubmit(t) => {
                if t.stake == 0 || self.balance(&src) < t.stake
                    || self.registry.contains_key(&tx.txid()) {
                    return false;
                }
                *self.balances.get_mut(&src).unwrap() -= t.stake;
                self.registry.insert(tx.txid(), json!({
                    "owner": src, "data_hash": t.data_hash, "size": t.size_bytes,
                    "media_type": t.media_type, "stake": t.stake,
                    "weight": t.stake, "status": "active"}));
            }
            AccountTx::DataChallenge(t) => {
                let ok = self.registry.get(&t.data_id)
                    .map(|e| e["status"] == "active").unwrap_or(false);
                let already = self.challenges.values()
                    .any(|c| c["data_id"] == t.data_id.as_str());
                if !ok || already || t.stake == 0 || self.balance(&src) < t.stake {
                    return false;
                }
                *self.balances.get_mut(&src).unwrap() -= t.stake;
                self.challenges.insert(tx.txid(), json!({
                    "data_id": t.data_id, "challenger": src, "stake": t.stake,
                    "reason": t.reason, "expiry": height + CHALLENGE_WINDOW,
                    "votes_for": [], "votes_against": []}));
            }
            AccountTx::DataVote(t) => {
                if !recent_proposers.contains(&t.voter_pub) {
                    return false;
                }
                let Some(ch) = self.challenges.get(&t.challenge_id) else {
                    return false;
                };
                let voted = ch["votes_for"].as_array().unwrap().iter().any(|v| v == src.as_str())
                    || ch["votes_against"].as_array().unwrap().iter().any(|v| v == src.as_str());
                // DISINTERESTED JURORS ONLY: neither the challenger nor the data
                // owner may vote on their own challenge — both are interested
                // parties. Jurors are disinterested recent proposers.
                let is_challenger = ch["challenger"].as_str() == Some(src.as_str());
                let data_id = ch["data_id"].as_str().unwrap().to_string();
                if voted || is_challenger {
                    return false;
                }
                if let Some(entry) = self.registry.get(&data_id) {
                    if entry["owner"].as_str() == Some(src.as_str()) {
                        return false;
                    }
                }
                let ch = self.challenges.get_mut(&t.challenge_id).unwrap();
                let k = if t.support { "votes_for" } else { "votes_against" };
                let arr = ch[k].as_array_mut().unwrap();
                arr.push(json!(src));
                arr.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
            }
            AccountTx::Transfer(_) => return false,
        }
        self.nonces.insert(src, tx.nonce() + 1);
        true
    }

    /// Canonical root — byte-identical to the Python reference: compact JSON,
    /// all keys sorted (BTreeMaps + serde_json's sorted Map make it structural).
    pub fn root(&self) -> String {
        let state = json!({
            "balances": self.balances,
            "challenges": self.challenges,
            "nonces": self.nonces,
            "registry": self.registry,
        });
        hex::encode(Sha256::digest(serde_json::to_string(&state).unwrap().as_bytes()))
    }

    pub fn supply(&self) -> u64 {
        self.balances.values().sum()
    }

    /// Serialize the full ledger for a snapshot (fast-boot). Structural, not the
    /// hashed root form — this round-trips the state itself.
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::json!({
            "balances": self.balances, "nonces": self.nonces,
            "registry": self.registry, "challenges": self.challenges,
        })
    }

    /// Reconstruct a ledger from a snapshot value (inverse of to_value).
    ///
    /// Returns None if the value is malformed in ANY way — wrong top-level
    /// shape, a non-integer balance/nonce, or a registry/challenge entry missing
    /// a field (or of the wrong type) that a later ledger path unwraps. A
    /// snapshot is untrusted input on the fast-boot path; seeding a partially-
    /// built ledger from it would either corrupt balances or panic a
    /// snapshot-booted node inside validate_block while block-synced peers march
    /// on. On None the caller falls back to full validated replay from genesis.
    pub fn from_value(v: &serde_json::Value) -> Option<Self> {
        let mut led = TokenLedger::new();
        for (k, x) in v["balances"].as_object()? {
            led.balances.insert(k.clone(), x.as_u64()?);
        }
        for (k, x) in v["nonces"].as_object()? {
            led.nonces.insert(k.clone(), x.as_u64()?);
        }
        for (k, e) in v["registry"].as_object()? {
            if !valid_registry_entry(e) {
                return None;
            }
            led.registry.insert(k.clone(), e.clone());
        }
        for (k, c) in v["challenges"].as_object()? {
            if !valid_challenge(c) {
                return None;
            }
            led.challenges.insert(k.clone(), c.clone());
        }
        Some(led)
    }
}

/// Every field the reward/resolve/apply/root paths read off a registry entry,
/// with the type they unwrap — validated once at snapshot load so no later
/// `.unwrap()` can panic on a crafted snapshot.
fn valid_registry_entry(e: &serde_json::Value) -> bool {
    e["owner"].is_string()
        && e["data_hash"].is_string()
        && e["size"].is_u64()
        && e["media_type"].is_string()
        && e["stake"].is_u64()
        && e["weight"].is_u64()
        && e["status"].is_string()
}

/// Likewise for an open challenge, including vote lists that must be arrays of
/// address strings (resolve/apply iterate and compare them).
fn valid_challenge(c: &serde_json::Value) -> bool {
    let str_array = |x: &serde_json::Value| {
        x.as_array().is_some_and(|a| a.iter().all(|v| v.is_string()))
    };
    c["data_id"].is_string()
        && c["challenger"].is_string()
        && c["stake"].is_u64()
        && c["reason"].is_string()
        && c["expiry"].is_u64()
        && str_array(&c["votes_for"])
        && str_array(&c["votes_against"])
}
