//! Negative validation tests: a validating node MUST reject malformed blocks.
//!
//! These pin the structural guards added for the production-readiness audit —
//! height linkage (task 90), n_txs / height-0 (task 95), and delta body length
//! (task 91). Each guard closes a concrete break: an unconstrained height lets a
//! miner mint the height-keyed reward forever; a mismatched-length delta panics
//! `trimmed_mean` on every validator (remote chain-halt). The positive path is
//! covered by the golden vectors; this file covers rejection.

use palimpsest_core as core;
use palimpsest_core::blocktree::{Block, BlockTree};
use std::collections::HashMap;

const DIM: usize = 16;

fn genesis_tree() -> BlockTree {
    BlockTree::new(vec![0i64; DIM], None)
}

/// A header whose roots are left blank — every guard under test fires before
/// the state/root checks, so a valid root is unnecessary to prove rejection.
fn header(tree: &BlockTree, height: u64, n_txs: u64) -> core::Header {
    core::Header {
        height,
        prev_hash: tree.genesis_hash.clone(),
        state_root: String::new(),
        txset_root: String::new(),
        n_txs,
        work: 1,
        proposer: "test".into(),
        transfer_root: String::new(),
        ledger_root: String::new(),
        data_root: String::new(),
    }
}

fn empty_block(header: core::Header) -> Block {
    Block { header, txs: vec![], bodies: HashMap::new(), transfers: vec![], data_txs: vec![] }
}

#[test]
fn rejects_wrong_height() {
    let mut tree = genesis_tree();
    // parent is genesis (height 0); a block claiming height 5 breaks the chain's
    // monotone height and must be rejected.
    let e = tree.add_block(empty_block(header(&tree, 5, 0))).unwrap_err();
    assert!(e.0.contains("height must be parent height + 1"), "got: {}", e.0);
}

#[test]
fn rejects_height_zero_nongenesis() {
    let mut tree = genesis_tree();
    // height 0 on a non-genesis block would underflow `h.height - 1`.
    let e = tree.add_block(empty_block(header(&tree, 0, 0))).unwrap_err();
    assert!(e.0.contains("height must be parent height + 1"), "got: {}", e.0);
}

#[test]
fn rejects_ntxs_mismatch() {
    let mut tree = genesis_tree();
    // header claims one tx but the block carries none.
    let e = tree.add_block(empty_block(header(&tree, 1, 1))).unwrap_err();
    assert!(e.0.contains("n_txs does not match"), "got: {}", e.0);
}

#[test]
fn rejects_wrong_length_delta_body() {
    let mut tree = genesis_tree();
    let key = core::Key::from_seed([7u8; 32]);
    // a correctly-signed, correctly-hashed delta whose body is the WRONG
    // dimension (3, not DIM=16) — this is exactly what would panic trimmed_mean.
    let body: Vec<i64> = vec![1, 2, 3];
    let dh = core::delta_hash(&core::int64_bytes(&body));
    let mut tx = core::BackpropTx {
        miner: key.pub_hex(),
        base_height: 0,
        shard_id: 0,
        delta_hash: dh.clone(),
        da_pointer: format!("da://{dh}"),
        sig: vec![],
    };
    tx.sig = key.sign(&tx.signing_bytes());
    assert!(tx.verify(), "test tx must be a valid signature to reach the length check");
    let mut bodies = HashMap::new();
    bodies.insert(tx.da_pointer.clone(), body);
    let blk = Block {
        header: header(&tree, 1, 1), // height + n_txs both valid; only length is wrong
        txs: vec![tx],
        bodies,
        transfers: vec![],
        data_txs: vec![],
    };
    let e = tree.add_block(blk).unwrap_err();
    assert!(e.0.contains("delta body length"), "got: {}", e.0);
}
