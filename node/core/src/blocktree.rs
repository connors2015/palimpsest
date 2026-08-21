//! Blocks, first-principles validation, and Nakamoto fork choice — mirroring
//! `rig/blockchain.py`. A node holding no special trust validates every block
//! completely: signatures, DA bodies against their hashes, the weight-state
//! transition (trimmed mean), the tx-set root, and — rev 2 — the full token
//! transition (rewards + canonical transfers) against the committed ledger_root.

use crate::token::{
    canonical_account_txs, data_root, transfer_root, AccountTx, TokenLedger,
    TransferTx, PROPOSER_LOOKBACK,
};
use crate::{delta_hash, int64_bytes, state_root, trimmed_mean, txset_root, BackpropTx, Header};
use std::collections::{HashMap, HashSet};

pub struct Block {
    pub header: Header,
    pub txs: Vec<BackpropTx>,
    pub bodies: HashMap<String, Vec<i64>>, // da_pointer -> dense delta
    pub transfers: Vec<TransferTx>,
    pub data_txs: Vec<AccountTx>,          // rev 3: Data{Submit,Challenge,Vote}
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
    parent_height: u64,
    parent_ledger: &TokenLedger,
    data_contributor: Option<&str>,
    recent_proposers: &HashSet<String>,
) -> Result<(Vec<i64>, TokenLedger), ValidationError> {
    let h = &block.header;
    let dim = parent_w.len();
    // 0. STRUCTURAL invariants binding the header to its parent and body.
    //    height must advance by exactly one — otherwise a miner could pin a low
    //    height on every block and mint the height-keyed reward forever (halving
    //    /sunset are only meaningful if height is monotone), and a height-0
    //    non-genesis block would underflow `h.height - 1` below.
    if h.height != parent_height + 1 {
        return Err(err("height must be parent height + 1"));
    }
    //    n_txs must match the real tx count (it is committed in the block hash).
    if h.n_txs as usize != block.txs.len() {
        return Err(err("n_txs does not match tx count"));
    }
    // PROPOSER LOTTERY (rev 4): the VRF proof must verify as a signature by the
    // proposer over this height's seed, and header.work must be the vrf_work
    // derived from it — so work is NON-FORGEABLE (a peer cannot claim an
    // arbitrary fork-choice weight). Genesis is constructed directly and exempt.
    if h.proposer != "genesis" {
        let proof = hex::decode(&h.vrf_proof).unwrap_or_default();
        let seed = crate::lottery::seed(&h.prev_hash, h.height);
        if !crate::verify_sig(&h.proposer, &seed, &proof) {
            return Err(err("invalid proposer VRF proof"));
        }
        if h.work != crate::lottery::vrf_work(&proof) {
            return Err(err("header.work is not the VRF-derived weight"));
        }
    }
    // 1. every delta tx well-formed and signed; its DA body must have the model
    //    dimension (so aggregation can't be made to panic/diverge by a short or
    //    long body) and match its hash.
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
        if body.len() != dim {
            return Err(err("delta body length != model dimension"));
        }
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
        // numpy int64 add wraps on overflow; mirror it so the state transition
        // is bit-identical to the reference (and never panics a debug build).
        parent_w.iter().zip(&mean).map(|(a, b)| a.wrapping_add(*b)).collect()
    };
    if state_root(&w) != h.state_root {
        return Err(err("state_root does not reproduce from txs"));
    }
    // 4. the transfer + data lanes: set roots + full token transition, in the
    //    exact reference order (resolve expired -> rewards -> merged canonical
    //    account txs)
    if transfer_root(&block.transfers) != h.transfer_root {
        return Err(err("transfer_root mismatch"));
    }
    if data_root(&block.data_txs) != h.data_root {
        return Err(err("data_root mismatch"));
    }
    let mut led = parent_ledger.clone();
    led.resolve_expired_challenges(h.height);
    let miner_pubs: Vec<String> = block.txs.iter().map(|t| t.miner.clone()).collect();
    let data_addrs: Vec<String> = data_contributor.map(|d| vec![d.to_string()]).unwrap_or_default();
    led.apply_reward(h.height, &miner_pubs, &h.proposer, &data_addrs);
    let mut merged: Vec<AccountTx> = block.data_txs.clone();
    merged.extend(block.transfers.iter().cloned().map(AccountTx::Transfer));
    for tx in canonical_account_txs(&merged) {
        let ok = match &tx {
            AccountTx::Transfer(t) => led.apply_transfer(t),
            _ => led.apply_data_tx(&tx, h.height, recent_proposers),
        };
        if !ok {
            return Err(err("invalid account tx (sig/nonce/balance/gating)"));
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
    /// keep full state vectors only this many blocks below the head (plus
    /// genesis) — an 86M state is ~0.7GB, so retaining one per block OOMs.
    /// Headers/ledgers/cum_work are kept forever (fork choice needs them).
    pub prune_depth: Option<u64>,
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
            data_root: String::new(),
            vrf_proof: String::new(),
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
            prune_depth: None,
        };
        t.blocks.insert(ghash.clone(), gh);
        t.state.insert(ghash.clone(), genesis_w);
        // fair launch: empty balances; the founding corpus is registry entry zero
        let mut genesis_ledger = TokenLedger::new();
        if let Some(dc) = &t.data_contributor {
            genesis_ledger.seed_genesis_data(dc);
        }
        t.ledger.insert(ghash.clone(), genesis_ledger);
        t.cum_work.insert(ghash, 0);
        t
    }

    /// Proposer pubkeys of the last PROPOSER_LOOKBACK blocks ending at `tip` —
    /// the deterministic juror set for data-challenge votes.
    pub fn recent_proposers(&self, tip: &str) -> HashSet<String> {
        let mut out = HashSet::new();
        let mut cur = tip.to_string();
        for _ in 0..PROPOSER_LOOKBACK {
            if cur == self.genesis_hash {
                break;
            }
            let Some(h) = self.blocks.get(&cur) else { break };
            out.insert(h.proposer.clone());
            cur = h.prev_hash.clone();
        }
        out
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
        let jurors = self.recent_proposers(&parent);
        let parent_height = self.blocks[&parent].height;
        let (w, led) = validate_block(
            &block,
            &parent_w,
            parent_height,
            &parent_led,
            self.data_contributor.as_deref(),
            &jurors,
        )?;
        // saturating: header.work is not yet cryptographically constrained
        // (see the proposer-eligibility work), so a peer could claim a huge
        // value; saturating_add keeps fork-choice bookkeeping panic-free and
        // deterministic until work is validated at its source.
        let work = self.cum_work[&parent].saturating_add(block.header.work.max(1));
        self.blocks.insert(bh.clone(), block.header);
        self.state.insert(bh.clone(), w);
        self.ledger.insert(bh.clone(), led);
        self.cum_work.insert(bh.clone(), work);
        // heaviest chain wins; ties broken by lexicographically smaller hash
        let head_work = self.cum_work[&self.head];
        let became = work > head_work || (work == head_work && bh < self.head);
        if became {
            self.head = bh;
        }
        self.prune_deep();
        Ok(became)
    }

    /// Drop heavy state vectors more than prune_depth below the head.
    fn prune_deep(&mut self) {
        let Some(depth) = self.prune_depth else { return };
        let head_h = self.blocks[&self.head].height;
        let floor = head_h.saturating_sub(depth);
        let doomed: Vec<String> = self.state.keys()
            .filter(|h| **h != self.genesis_hash
                    && self.blocks[*h].height < floor)
            .cloned().collect();
        for h in doomed {
            self.state.remove(&h);
        }
    }

    pub fn head_state(&self) -> &Vec<i64> {
        &self.state[&self.head]
    }

    pub fn head_ledger(&self) -> &TokenLedger {
        &self.ledger[&self.head]
    }
}
