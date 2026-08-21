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
use palimpsest_core::token::TokenLedger;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use tracing::{info, warn};

pub const SNAPSHOT_EVERY: u64 = 25;

/// (tree, block-index for serving, in-memory payload cache for recent blocks)
type Rebuilt = (BlockTree, HashMap<String, StoredBlock>, HashMap<String, Payload>);

pub struct Store {
    dir: PathBuf,
    /// Held for the process lifetime: an advisory flock on the data dir. The
    /// kernel releases it automatically when the process exits (fd closed), so a
    /// crash never leaves a stale lock. Two processes on one --data-dir would
    /// interleave appends into blocks.jsonl and corrupt the chain.
    _lock: fs::File,
}

impl Store {
    pub fn open(dir: &str) -> std::io::Result<Store> {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(dir.join("payloads"))?;
        let lock = Self::acquire_lock(&dir)?;
        Ok(Store { dir, _lock: lock })
    }

    /// Take an exclusive, non-blocking advisory lock on `<dir>/.lock`. Fails
    /// with a clear error if another node process already holds it.
    fn acquire_lock(dir: &PathBuf) -> std::io::Result<fs::File> {
        let path = dir.join(".lock");
        let file = fs::OpenOptions::new().create(true).write(true).truncate(false).open(&path)?;
        // SAFETY: valid fd for the lifetime of `file`; LOCK_NB never blocks.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!("another palimpsest-node already holds {}", path.display()),
            ));
        }
        Ok(file)
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

    // ---- uploaded corpus files (DA custody for /upload submissions) ------
    pub fn save_upload(&self, hash: &str, bytes: &[u8]) -> std::io::Result<()> {
        let dir = self.dir.join("uploads");
        fs::create_dir_all(&dir)?;
        let path = dir.join(hash);
        if !path.exists() {
            fs::write(path, bytes)?;
        }
        Ok(())
    }

    // ---- payloads --------------------------------------------------------
    /// Persist a payload (idempotent). Written to a temp file + rename so a
    /// crash can't leave a half-written body that get_payload would accept, and
    /// fsync'd for durability. Returns whether it is now on disk — the caller
    /// (which drops the in-memory copy once a delta is mined) must not discard
    /// memory if this reports false.
    pub fn put_payload(&self, txid: &str, p: &Payload) -> bool {
        let path = self.dir.join("payloads").join(format!("{txid}.json"));
        if path.exists() {
            return true;
        }
        let tmp = self.dir.join("payloads").join(format!("{txid}.json.tmp"));
        let write = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&serde_json::to_vec(p).unwrap())?;
            f.sync_all()?;
            fs::rename(&tmp, &path)
        })();
        if let Err(e) = write {
            warn!(%txid, "failed to persist payload: {e}");
            let _ = fs::remove_file(&tmp);
            return false;
        }
        true
    }

    pub fn get_payload(&self, txid: &str) -> Option<Payload> {
        let raw = fs::read(self.dir.join("payloads").join(format!("{txid}.json"))).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    /// Delete a payload that will never be part of a block (a mempool delta
    /// evicted before inclusion) — reclaims disk from spammed/never-mined deltas.
    pub fn remove_payload(&self, txid: &str) {
        let _ = fs::remove_file(self.dir.join("payloads").join(format!("{txid}.json")));
    }

    // ---- block log -------------------------------------------------------
    /// Append + fsync a block. fsync makes the record durable across power loss;
    /// the caller MUST treat an Err as fatal (a dropped write silently truncates
    /// the chain on the next boot).
    pub fn append_block(&self, b: &StoredBlock) -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new().create(true).append(true)
            .open(self.dir.join("blocks.jsonl"))?;
        writeln!(f, "{}", serde_json::to_string(b).unwrap())?;
        f.sync_all()
    }

    /// Read the block log, tolerating a torn final record (a crash mid-append)
    /// by self-healing: truncate to the last good line. A corrupt record in the
    /// MIDDLE is real corruption — stop there and warn loudly rather than
    /// silently skipping it (skipping would orphan every block after it).
    pub fn read_blocks(&self) -> Vec<StoredBlock> {
        let Ok(raw) = fs::read_to_string(self.dir.join("blocks.jsonl")) else {
            return vec![];
        };
        let lines: Vec<&str> = raw.split('\n').collect();
        let mut out = Vec::new();
        let mut good_bytes = 0usize; // byte length of the fully-valid prefix
        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() {
                if i + 1 == lines.len() {
                    break; // trailing "" after the final newline — clean EOF
                }
                good_bytes += 1; // a blank line's '\n'
                continue;
            }
            match serde_json::from_str::<StoredBlock>(line) {
                Ok(b) => {
                    out.push(b);
                    good_bytes += line.len() + 1; // + the '\n'
                }
                Err(e) => {
                    let more_after = lines[i + 1..].iter().any(|l| !l.is_empty());
                    if more_after {
                        warn!(line = i, err = %e,
                              "corrupt block record mid-log — replay stops here");
                    } else {
                        warn!(line = i, "torn final block record — truncating to last good line");
                        let _ = self.truncate_blocks(good_bytes);
                    }
                    break;
                }
            }
        }
        out
    }

    fn truncate_blocks(&self, len: usize) -> std::io::Result<()> {
        let f = fs::OpenOptions::new().write(true)
            .open(self.dir.join("blocks.jsonl"))?;
        f.set_len(len as u64)?;
        f.sync_all()
    }

    // ---- snapshots -------------------------------------------------------
    /// Checkpoint the full head state AND ledger, written atomically (temp +
    /// rename) so a crash mid-write can't leave a torn snapshot the fast path
    /// would trust. The state goes to a binary blob; hash/height/ledger to JSON.
    pub fn write_snapshot(&self, block_hash: &str, height: u64, state: &[i64],
                          ledger: &TokenLedger) {
        let bin_tmp = self.dir.join("snapshot.bin.tmp");
        if fs::write(&bin_tmp, palimpsest_core::int64_bytes(state)).is_err() {
            return;
        }
        let _ = fs::rename(&bin_tmp, self.dir.join("snapshot.bin"));
        let meta = serde_json::json!({"hash": block_hash, "height": height,
                                      "ledger": ledger.to_value()});
        let json_tmp = self.dir.join("snapshot.json.tmp");
        if fs::write(&json_tmp, meta.to_string()).is_ok() {
            let _ = fs::rename(&json_tmp, self.dir.join("snapshot.json"));
        }
    }

    pub fn read_snapshot(&self) -> Option<(String, u64, Vec<i64>, TokenLedger)> {
        let meta: serde_json::Value = serde_json::from_slice(
            &fs::read(self.dir.join("snapshot.json")).ok()?).ok()?;
        // reject pre-ledger snapshots (older format) — seeding an empty ledger
        // would corrupt balances, and fast_replay would NOT fall back since it
        // "succeeded". No ledger field => None => full validated replay.
        if !meta["ledger"].is_object() {
            warn!("snapshot has no ledger (old format) — ignoring, will full-replay");
            return None;
        }
        let raw = fs::read(self.dir.join("snapshot.bin")).ok()?;
        let state = raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        // a malformed ledger => reject the whole snapshot => full validated replay
        let ledger = match TokenLedger::from_value(&meta["ledger"]) {
            Some(l) => l,
            None => {
                warn!("snapshot ledger is malformed — ignoring, will full-replay");
                return None;
            }
        };
        Some((meta["hash"].as_str()?.to_string(), meta["height"].as_u64()?, state, ledger))
    }

    /// Rebuild the tree + indices from disk. Tries FAST-BOOT from the newest
    /// snapshot (trust the checkpointed state/ledger, validate only the blocks
    /// after it); any problem falls back to full validated replay from genesis.
    pub fn replay(&self, data_contributor: Option<String>, prune_depth: u64)
        -> Option<Rebuilt>
    {
        let genesis = self.read_genesis()?;
        let blocks = self.read_blocks();
        if let Some((h, height, state, ledger)) = self.read_snapshot() {
            if let Some(r) = self.fast_replay(&genesis, &blocks, &data_contributor,
                                              prune_depth, &h, height, state, ledger) {
                return Some(r);
            }
            warn!("fast-boot unusable — falling back to full validated replay");
        }
        self.full_replay(genesis, &blocks, data_contributor, prune_depth)
    }

    /// Fast path: seed the tree with the snapshot's TRUSTED state+ledger at its
    /// block, cheaply index all headers (no payloads, no trimmed-mean), then run
    /// full validation forward from the snapshot only. Returns None (→ fallback)
    /// if the snapshot block isn't in the log or nothing validates past it.
    #[allow(clippy::too_many_arguments)]
    fn fast_replay(&self, genesis: &[i64], blocks: &[StoredBlock],
                   dc: &Option<String>, prune_depth: u64, snap_hash: &str,
                   snap_h: u64, snap_state: Vec<i64>, snap_ledger: TokenLedger)
        -> Option<Rebuilt>
    {
        let mut tree = BlockTree::new(genesis.to_vec(), dc.clone());
        tree.prune_depth = Some(prune_depth);

        // 1. headers + cum_work for every block up to the snapshot height, in
        //    height order so parents precede children (cheap — headers only).
        let mut sorted: Vec<&StoredBlock> = blocks.iter().collect();
        sorted.sort_by_key(|b| b.header.height);
        for sb in &sorted {
            if sb.header.height > snap_h {
                break;
            }
            let hdr = sb.header.to_core();
            if let Some(pw) = tree.cum_work.get(&hdr.prev_hash).copied() {
                let h = sb.hash();
                tree.blocks.insert(h.clone(), hdr.clone());
                tree.cum_work.insert(h, pw + hdr.work.max(1));
            }
        }
        if !tree.blocks.contains_key(snap_hash) {
            return None; // snapshot block not on disk — can't trust it
        }
        // 2. seed the checkpointed state + ledger at the snapshot block
        tree.state.insert(snap_hash.to_string(), snap_state);
        tree.ledger.insert(snap_hash.to_string(), snap_ledger);
        tree.head = snap_hash.to_string();

        // 3. index every stored block so we can still serve old ones; validate
        //    FORWARD only the blocks after the snapshot (add_block validates
        //    fully + runs fork choice, reconstructing their state/ledger).
        let mut index: HashMap<String, StoredBlock> =
            blocks.iter().map(|b| (b.hash(), b.clone())).collect();
        let mut cache = HashMap::new();
        let mut validated = 0u64;
        for sb in &sorted {
            if sb.header.height <= snap_h
                || !tree.state.contains_key(&sb.header.prev_hash) {
                continue;
            }
            let mut payloads = HashMap::new();
            for wt in &sb.txs {
                if let Some(t) = wt.to_core() {
                    if let Some(p) = self.get_payload(&t.txid()) {
                        payloads.insert(t.txid(), p);
                    }
                }
            }
            let Some(block) = sb.to_core(&payloads) else { continue };
            if tree.add_block(block).is_ok() {
                for (txid, p) in payloads { cache.insert(txid, p); }
                validated += 1;
            }
        }
        index.retain(|_, sb| sb.header.height <= tree.blocks[&tree.head].height);
        info!(from = snap_h, to = tree.blocks[&tree.head].height,
              validated, "FAST-BOOT from snapshot");
        Some((tree, index, cache))
    }

    fn full_replay(&self, genesis: Vec<i64>, blocks: &[StoredBlock],
                   dc: Option<String>, prune_depth: u64) -> Option<Rebuilt> {
        let mut tree = BlockTree::new(genesis, dc);
        tree.prune_depth = Some(prune_depth);
        let mut index = HashMap::new();
        let mut cache = HashMap::new();
        for sb in blocks {
            let mut payloads = HashMap::new();
            for wt in &sb.txs {
                let Some(t) = wt.to_core() else { continue };
                if let Some(p) = self.get_payload(&t.txid()) {
                    payloads.insert(t.txid(), p);
                }
            }
            let Some(block) = sb.to_core(&payloads) else {
                warn!(hash = %sb.hash(), "block missing payloads — stopping replay");
                break;
            };
            match tree.add_block(block) {
                Ok(_) => {
                    for (txid, p) in payloads { cache.insert(txid, p); }
                    index.insert(sb.hash(), sb.clone());
                }
                Err(e) => {
                    warn!(hash = %sb.hash(), err = %e.0, "invalid block — stopping replay");
                    break;
                }
            }
        }
        info!(height = tree.blocks[&tree.head].height, blocks = index.len(),
              "full validated replay from genesis");
        Some((tree, index, cache))
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("palimpsest-store-test-{}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&p);
        p
    }

    fn block_line(height: u64) -> String {
        let hdr = format!(
            "{{\"height\":{height},\"prev_hash\":\"00\",\"state_root\":\"aa\",\
\"txset_root\":\"bb\",\"n_txs\":0,\"work\":1,\"proposer\":\"p\",\
\"transfer_root\":\"\",\"ledger_root\":\"\",\"data_root\":\"\"}}");
        format!("{{\"header\":{hdr},\"txs\":[],\"transfers\":[],\"data_txs\":[]}}")
    }

    #[test]
    fn data_dir_lock_is_exclusive() {
        let dir = tmpdir("lock");
        let s1 = Store::open(dir.to_str().unwrap()).expect("first open ok");
        assert!(Store::open(dir.to_str().unwrap()).is_err(),
                "a second open of the same data-dir must be rejected");
        drop(s1); // releasing the lock lets a fresh process take it
        assert!(Store::open(dir.to_str().unwrap()).is_ok(),
                "open after release must succeed");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_blocks_recovers_from_torn_final_line() {
        let dir = tmpdir("torn");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        // two good records + a torn final line, exactly as a crash mid-append leaves
        let path = dir.join("blocks.jsonl");
        let content = format!("{}\n{}\n{}",
            block_line(1), block_line(2), r#"{"ZZZTORN":{"height":3,"#);
        fs::write(&path, &content).unwrap();
        let blocks = store.read_blocks();
        assert_eq!(blocks.len(), 2, "recovers the two good records");
        assert_eq!(blocks[0].header.height, 1);
        assert_eq!(blocks[1].header.height, 2);
        // the torn tail was self-healed off disk
        assert!(!fs::read_to_string(&path).unwrap().contains("ZZZTORN"),
                "torn line must be truncated from disk");
        assert_eq!(store.read_blocks().len(), 2, "re-read is clean");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_blocks_stops_at_midfile_corruption() {
        let dir = tmpdir("corrupt");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        // a corrupt record with a VALID record after it => real corruption:
        // stop at the corruption rather than silently skipping it.
        let content = format!("{}\n{}\n{}\n",
            block_line(1), "{not json}", block_line(3));
        fs::write(dir.join("blocks.jsonl"), &content).unwrap();
        let blocks = store.read_blocks();
        assert_eq!(blocks.len(), 1, "stops at the corrupt middle line");
        let _ = fs::remove_dir_all(&dir);
    }
}
