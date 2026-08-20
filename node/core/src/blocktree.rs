//! Blocks, first-principles validation, and Nakamoto fork choice — mirroring
//! `rig/blockchain.py`. A node holding no special trust validates every block
//! completely: signatures, DA bodies against their hashes, the weight-state
//! transition (trimmed mean), the tx-set root, and — rev 2 — the full token
//! transition (rewards + canonical transfers) against the committed ledger_root.

use crate::token::{canonical_transfers, transfer_root, TokenLedger, TransferTx};
use crate::{delta_hash, int64_bytes, state_root, trimmed_mean, txset_root, BackpropTx, Header};
use std::collections::HashMap;

pub struct Block {
    pub header: Header,
    pub txs: Vec<BackpropTx>,
    pub bodies: HashMap<String, Vec<i64>>, // da_pointer -> dense delta
    pub transfers: Vec<TransferTx>,
}

impl Block {
    pub fn hash(&self) -> String {
        self.header.block_hash()
    }
}

#[derive(Debug)]
pub struct ValidationError(pub String);

fn err(msg: &str) -> ValidationError {
    ValidationError(msg.to_string())
}

/// Full validation against the parent's state; returns (post-weights, post-ledger).
pub fn validate_block(
    block: &Block,
    parent_w: &[i64],
    parent_ledger: &TokenLedger,
    data_contributor: Option<&str>,
) -> Result<(Vec<i64>, TokenLedger), ValidationError> {
    let h = &block.header;
    // 1. every delta tx well-formed and signed; DA body matches its hash
    for tx in &block.txs {
        if !tx.verify() {
            return Err(err("bad signature on tx"));
        }
        if tx.base_height != h.height - 1 {
            return Err(err("tx base_height does not match parent"));
        }
        let body = block
            .bodies
            .get(&tx.da_pointer)
            .ok_or_else(|| err("missing DA body"))?;
        if delta_hash(&int64_bytes(body)) != tx.delta_hash {
            return Err(err("delta body hash mismatch"));
        }
    }
    // 2. tx-set root
    let ids: Vec<String> = block.txs.iter().map(|t| t.txid()).collect();
    if txset_root(&ids) != h.txset_root {
        return Err(err("txset_root mismatch"));
    }
    // 3. weight-state transition reproduces the committed root
    let w = if block.txs.is_empty() {
        parent_w.to_vec()
    } else {
        let deltas: Vec<Vec<i64>> = block
            .txs
            .iter()
            .map(|t| block.bodies[&t.da_pointer].clone())
            .collect();
        let mean = trimmed_mean(&deltas, 0.2);
        parent_w.iter().zip(&mean).map(|(a, b)| a + b).collect()
    };
    if state_root(&w) != h.state_root {
        return Err(err("state_root does not reproduce from txs"));
    }
    // 4. the transfer lane: transfer-set root + full token transition
    if transfer_root(&block.transfers) != h.transfer_root {
        return Err(err("transfer_root mismatch"));
    }
    let mut led = parent_ledger.clone();
    let miner_pubs: Vec<String> = block.txs.iter().map(|t| t.miner.clone()).collect();
    let data_addrs: Vec<String> = data_contributor.map(|d| vec![d.to_string()]).unwrap_or_default();
    led.apply_reward(h.height, &miner_pubs, &h.proposer, &data_addrs);
    for tx in canonical_transfers(&block.transfers) {
        if !led.apply_transfer(&tx) {
            return Err(err("invalid transfer (sig/nonce/balance)"));
        }
    }
    if led.root() != h.ledger_root {
        return Err(err("ledger_root does not reproduce from block"));
    }
    Ok((w, led))
}

/// All known blocks with heaviest-valid-chain (Nakamoto) fork choice.
pub struct BlockTree {
    pub blocks: HashMap<String, Header>, // header per hash (bodies not retained)
    pub state: HashMap<String, Vec<i64>>,
    pub ledger: HashMap<String, TokenLedger>,
    pub cum_work: HashMap<String, u64>,
    pub head: String,
    pub genesis_hash: String,
    pub data_contributor: Option<String>,
}

impl BlockTree {
    pub fn new(genesis_w: Vec<i64>, data_contributor: Option<String>) -> Self {
        let gh = Header {
            height: 0,
            prev_hash: "0".repeat(64),
            state_root: state_root(&genesis_w),
            txset_root: crate::sha256_hex_pub(b""),
            n_txs: 0,
            work: 0,
            proposer: "genesis".into(),
            transfer_root: String::new(),
            ledger_root: String::new(),
        };
        let ghash = gh.block_hash();
        let mut t = BlockTree {
            blocks: HashMap::new(),
            state: HashMap::new(),
            ledger: HashMap::new(),
            cum_work: HashMap::new(),
            head: ghash.clone(),
            genesis_hash: ghash.clone(),
            data_contributor,
        };
        t.blocks.insert(ghash.clone(), gh);
        t.state.insert(ghash.clone(), genesis_w);
        t.ledger.insert(ghash.clone(), TokenLedger::new()); // fair launch: empty
        t.cum_work.insert(ghash, 0);
        t
    }

    /// Validate + attach; returns Ok(true) if the block became the new head.
    pub fn add_block(&mut self, block: Block) -> Result<bool, ValidationError> {
        let bh = block.hash();
        if self.blocks.contains_key(&bh) {
            return Ok(false);
        }
        let parent = block.header.prev_hash.clone();
        let parent_w = self
            .state
            .get(&parent)
            .ok_or_else(|| err("orphan: parent unknown"))?
            .clone();
        let parent_led = self.ledger[&parent].clone();
        let (w, led) = validate_block(
            &block,
            &parent_w,
            &parent_led,
            self.data_contributor.as_deref(),
        )?;
        let work = self.cum_work[&parent] + block.header.work.max(1);
        self.blocks.insert(bh.clone(), block.header);
        self.state.insert(bh.clone(), w);
        self.ledger.insert(bh.clone(), led);
        self.cum_work.insert(bh.clone(), work);
        // heaviest chain wins; ties broken by lexicographically smaller hash
        let head_work = self.cum_work[&self.head];
        if work > head_work || (work == head_work && bh < self.head) {
            self.head = bh;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn head_state(&self) -> &Vec<i64> {
        &self.state[&self.head]
    }

    pub fn head_ledger(&self) -> &TokenLedger {
        &self.ledger[&self.head]
    }
}
