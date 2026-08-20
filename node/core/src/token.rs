//! The native token ledger — chain state, mirroring `rig/token.py` bit-exactly.
//!
//! Balances and nonces live in BTreeMaps so canonical (sorted) serialization is
//! structural. The ledger root reproduces Python's
//! `json.dumps({"balances":…, "nonces":…}, sort_keys=True, separators=(",",":"))`
//! byte-for-byte.

use crate::verify_sig;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const GRAIN: u64 = 1_000_000_000;
pub const BASE_REWARD: u64 = 50 * GRAIN;
pub const HALVING_BLOCKS: u64 = 100_000;
pub const SUNSET_HEIGHT: u64 = 1_000_000;
pub const SHARE_MINERS: u64 = 7_000;
pub const SHARE_PROPOSER: u64 = 1_000;
pub const SHARE_DATA: u64 = 2_000;

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

#[derive(Clone, Debug)]
pub struct TransferTx {
    pub from_pub: String,
    pub to_addr: String,
    pub amount: u64,
    pub nonce: u64,
    pub sig: Vec<u8>,
}

impl TransferTx {
    pub fn signing_bytes(&self) -> Vec<u8> {
        format!(
            "transfer|{}|{}|{}|{}",
            self.from_pub, self.to_addr, self.amount, self.nonce
        )
        .into_bytes()
    }

    pub fn txid(&self) -> String {
        hex::encode(Sha256::digest(&self.signing_bytes()))
    }

    pub fn verify(&self) -> bool {
        verify_sig(&self.from_pub, &self.signing_bytes(), &self.sig)
    }
}

/// Consensus ordering of a block's transfers: (sender address, nonce, txid).
pub fn canonical_transfers(transfers: &[TransferTx]) -> Vec<TransferTx> {
    let mut out = transfers.to_vec();
    out.sort_by_key(|t| (address(&t.from_pub), t.nonce, t.txid()));
    out
}

/// Order-independent commitment to a block's transfer set.
pub fn transfer_root(transfers: &[TransferTx]) -> String {
    let mut ids: Vec<String> = transfers.iter().map(|t| t.txid()).collect();
    ids.sort();
    hex::encode(Sha256::digest(ids.join("|").as_bytes()))
}

#[derive(Clone, Default, Debug)]
pub struct TokenLedger {
    pub balances: BTreeMap<String, u64>,
    pub nonces: BTreeMap<String, u64>,
}

impl TokenLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn balance(&self, addr: &str) -> u64 {
        *self.balances.get(addr).unwrap_or(&0)
    }

    fn credit(&mut self, addr: &str, amount: u64) {
        if amount > 0 {
            *self.balances.entry(addr.to_string()).or_insert(0) += amount;
        }
    }

    /// Mint and split the block reward. Integer division; dust burned.
    pub fn apply_reward(
        &mut self,
        height: u64,
        miner_pubs: &[String],
        proposer_pub: &str,
        data_addrs: &[String],
    ) {
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
            for pub_hex in sorted {
                let a = address(pub_hex);
                self.credit(&a, each);
            }
        }
        if !proposer_pub.is_empty() && proposer_pub != "genesis" {
            let a = address(proposer_pub);
            self.credit(&a, proposer_cut);
        }
        if !data_addrs.is_empty() {
            let each = data_pool / data_addrs.len() as u64;
            let mut sorted: Vec<&String> = data_addrs.iter().collect();
            sorted.sort();
            for addr in sorted {
                self.credit(addr, each);
            }
        }
    }

    /// Validate + apply one transfer. False = invalid (caller treats block invalid).
    pub fn apply_transfer(&mut self, tx: &TransferTx) -> bool {
        if !tx.verify() || tx.amount == 0 {
            return false;
        }
        let src = address(&tx.from_pub);
        if tx.nonce != *self.nonces.get(&src).unwrap_or(&0) {
            return false;
        }
        if self.balance(&src) < tx.amount {
            return false;
        }
        *self.balances.get_mut(&src).unwrap() -= tx.amount;
        self.credit(&tx.to_addr, tx.amount);
        self.nonces.insert(src, tx.nonce + 1);
        true
    }

    /// Canonical root — byte-identical to the Python reference's json.dumps
    /// with sort_keys=True and separators (",", ":").
    pub fn root(&self) -> String {
        let ser_map = |m: &BTreeMap<String, u64>| -> String {
            let inner: Vec<String> =
                m.iter().map(|(k, v)| format!("\"{}\":{}", k, v)).collect();
            format!("{{{}}}", inner.join(","))
        };
        let blob = format!(
            "{{\"balances\":{},\"nonces\":{}}}",
            ser_map(&self.balances),
            ser_map(&self.nonces)
        );
        hex::encode(Sha256::digest(blob.as_bytes()))
    }

    pub fn supply(&self) -> u64 {
        self.balances.values().sum()
    }
}
