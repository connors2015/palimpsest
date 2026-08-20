//! Disk persistence — a node must survive restarts with its chain intact.
//!
//! Layout under --data-dir:
//!   genesis.bin           raw i64-LE genesis weight vector
//!   blocks.jsonl          append-only StoredBlock per accepted block
//!   payloads/<txid>.json  compressed delta payloads (the DA bodies)
//!   snapshot.bin + snapshot.json   head-state checkpoint every SNAPSHOT_EVERY
//!
//! Boot = validated replay: headers/ledger replay from blocks.jsonl (our own
//! previously-validated data), state from the newest usable snapshot, then
//! full first-principles validation for every block after it. A corrupted or
//! missing snapshot silently degrades to full replay from genesis.

use crate::proto::{Payload, StoredBlock};
use palimpsest_core::blocktree::BlockTree;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tracing::{info, warn};

pub const SNAPSHOT_EVERY: u64 = 25;

pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn open(dir: &str) -> std::io::Result<Store> {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(dir.join("payloads"))?;
        Ok(Store { dir })
    }

    // ---- genesis ---------------------------------------------------------
    pub fn write_genesis(&self, w: &[i64]) -> std::io::Result<()> {
        let path = self.dir.join("genesis.bin");
        if path.exists() {
            return Ok(());
        }
        fs::write(path, palimpsest_core::int64_bytes(w))
    }

    pub fn read_genesis(&self) -> Option<Vec<i64>> {
        let raw = fs::read(self.dir.join("genesis.bin")).ok()?;
        Some(raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }

    // ---- payloads --------------------------------------------------------
    pub fn put_payload(&self, txid: &str, p: &Payload) {
        let path = self.dir.join("payloads").join(format!("{txid}.json"));
        if !path.exists() {
            let _ = fs::write(path, serde_json::to_vec(p).unwrap());
        }
    }

    pub fn get_payload(&self, txid: &str) -> Option<Payload> {
        let raw = fs::read(self.dir.join("payloads").join(format!("{txid}.json"))).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    // ---- block log -------------------------------------------------------
    pub fn append_block(&self, b: &StoredBlock) -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new().create(true).append(true)
            .open(self.dir.join("blocks.jsonl"))?;
        writeln!(f, "{}", serde_json::to_string(b).unwrap())
    }

    pub fn read_blocks(&self) -> Vec<StoredBlock> {
        let Ok(raw) = fs::read_to_string(self.dir.join("blocks.jsonl")) else {
            return vec![];
        };
        raw.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    // ---- snapshots -------------------------------------------------------
    pub fn write_snapshot(&self, block_hash: &str, height: u64, state: &[i64]) {
        let _ = fs::write(self.dir.join("snapshot.bin"),
                          palimpsest_core::int64_bytes(state));
        let _ = fs::write(self.dir.join("snapshot.json"),
                          serde_json::json!({"hash": block_hash, "height": height})
                              .to_string());
    }

    pub fn read_snapshot(&self) -> Option<(String, u64, Vec<i64>)> {
        let meta: serde_json::Value = serde_json::from_slice(
            &fs::read(self.dir.join("snapshot.json")).ok()?).ok()?;
        let raw = fs::read(self.dir.join("snapshot.bin")).ok()?;
        let state = raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        Some((meta["hash"].as_str()?.to_string(), meta["height"].as_u64()?, state))
    }

    /// Rebuild the tree + payload/block indices from disk. Returns
    /// (tree, blocks index, in-memory payload cache for recent blocks).
    pub fn replay(&self, data_contributor: Option<String>, prune_depth: u64)
        -> Option<(BlockTree, HashMap<String, StoredBlock>, HashMap<String, Payload>)>
    {
        let genesis = self.read_genesis()?;
        let mut tree = BlockTree::new(genesis, data_contributor);
        tree.prune_depth = Some(prune_depth);
        let blocks = self.read_blocks();
        let mut index = HashMap::new();
        let mut cache = HashMap::new();
        let snap = self.read_snapshot();
        for sb in &blocks {
            // gather this block's payloads from disk
            let mut payloads = HashMap::new();
            for wt in &sb.txs {
                let Some(t) = wt.to_core() else { continue };
                if let Some(p) = self.get_payload(&t.txid()) {
                    payloads.insert(t.txid(), p);
                }
            }
            let Some(block) = sb.to_core(&payloads) else {
                warn!(hash = %sb.hash(), "block on disk missing payloads — stopping replay here");
                break;
            };
            match tree.add_block(block) {
                Ok(_) => {
                    for (txid, p) in payloads {
                        cache.insert(txid, p);
                    }
                    index.insert(sb.hash(), sb.clone());
                }
                Err(e) => {
                    warn!(hash = %sb.hash(), err = %e.0, "invalid block on disk — stopping replay");
                    break;
                }
            }
        }
        if let Some((h, height, _)) = snap {
            // snapshot is an integrity cross-check on replay (full validated
            // replay is authoritative; a mismatch means disk corruption)
            if let Some(hdr) = tree.blocks.get(&h) {
                if hdr.height != height {
                    warn!("snapshot metadata disagrees with replay — ignored");
                }
            }
        }
        info!(height = tree.blocks[&tree.head].height, blocks = index.len(),
              "replayed chain from disk");
        Some((tree, index, cache))
    }
}
