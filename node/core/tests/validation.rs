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

/// A header whose roots are left blank but which carries a VALID proposer VRF
/// proof + matching work, so validation passes the lottery gate and reaches the
/// specific guard under test (height / n_txs / body length).
fn header(tree: &BlockTree, height: u64, n_txs: u64) -> core::Header {
    let key = core::Key::from_seed([9u8; 32]);
    let prev = tree.genesis_hash.clone();
    let proof = core::lottery::vrf_prove(&key, &prev, height);
    core::Header {
        height,
        prev_hash: prev,
        state_root: String::new(),
        txset_root: String::new(),
        n_txs,
        work: core::lottery::vrf_work(&proof),
        proposer: key.pub_hex(),
        transfer_root: String::new(),
        ledger_root: String::new(),
        data_root: String::new(),
        vrf_proof: hex::encode(&proof),
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
fn rejects_forged_work() {
    let mut tree = genesis_tree();
    let mut h = header(&tree, 1, 0); // valid VRF proof + correct vrf_work
    h.work = 999_999; // ...but claim an inflated fork-choice weight
    let e = tree.add_block(empty_block(h)).unwrap_err();
    assert!(e.0.contains("VRF-derived weight"), "got: {}", e.0);
}

#[test]
fn rejects_invalid_vrf_proof() {
    let mut tree = genesis_tree();
    let mut h = header(&tree, 1, 0);
    h.vrf_proof = "00".repeat(64); // not a valid signature by the proposer
    let e = tree.add_block(empty_block(h)).unwrap_err();
    assert!(e.0.contains("invalid proposer VRF proof"), "got: {}", e.0);
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

// --- data-challenge market: quorum + disinterested jurors (task 93) ---------

use palimpsest_core::token::{
    address, AccountTx, DataChallengeTx, DataSubmitTx, DataVoteTx, TokenLedger,
};
use std::collections::HashSet;

fn signed(mut tx: AccountTx, key: &core::Key) -> AccountTx {
    let sig = key.sign(&tx.signing_bytes());
    match &mut tx {
        AccountTx::Transfer(t) => t.sig = sig,
        AccountTx::DataSubmit(t) => t.sig = sig,
        AccountTx::DataChallenge(t) => t.sig = sig,
        AccountTx::DataVote(t) => t.sig = sig,
    }
    tx
}

/// Fund an owner, register a staked entry, fund a challenger, open a challenge.
/// Returns (ledger, owner, challenger, jurors, data_id, challenge_id).
fn open_challenge() -> (TokenLedger, core::Key, core::Key, Vec<core::Key>, String, String) {
    let owner = core::Key::from_seed([1u8; 32]);
    let challenger = core::Key::from_seed([2u8; 32]);
    let jurors: Vec<core::Key> = (10u8..13).map(|i| core::Key::from_seed([i; 32])).collect();
    let mut led = TokenLedger::new();
    // fund owner + challenger via block rewards
    led.apply_reward(1, &[owner.pub_hex()], &owner.pub_hex(), &[]);
    let sub = signed(AccountTx::DataSubmit(DataSubmitTx {
        owner_pub: owner.pub_hex(), data_hash: "aa".repeat(32), size_bytes: 8,
        media_type: "text".into(), stake: 1_000_000, nonce: 0, sig: vec![],
    }), &owner);
    assert!(led.apply_data_tx(&sub, 1, &HashSet::new()));
    let data_id = sub.txid();
    led.apply_reward(2, &[challenger.pub_hex()], &challenger.pub_hex(), &[]);
    let ch = signed(AccountTx::DataChallenge(DataChallengeTx {
        challenger_pub: challenger.pub_hex(), data_id: data_id.clone(), stake: 500_000,
        reason: "validity".into(), nonce: 0, sig: vec![],
    }), &challenger);
    assert!(led.apply_data_tx(&ch, 2, &HashSet::new()));
    let challenge_id = ch.txid();
    (led, owner, challenger, jurors, data_id, challenge_id)
}

#[test]
fn challenger_cannot_vote_on_own_challenge() {
    let (mut led, _owner, challenger, _jurors, _data_id, challenge_id) = open_challenge();
    let jset: HashSet<String> = [challenger.pub_hex()].into_iter().collect();
    let vote = signed(AccountTx::DataVote(DataVoteTx {
        voter_pub: challenger.pub_hex(), challenge_id, support: true, nonce: 1, sig: vec![],
    }), &challenger);
    // even though the challenger is a "recent proposer", they are an interested
    // party and must be rejected as a juror.
    assert!(!led.apply_data_tx(&vote, 3, &jset), "challenger self-vote must be rejected");
}

#[test]
fn owner_cannot_vote_on_challenge_of_own_entry() {
    let (mut led, owner, _challenger, _jurors, _data_id, challenge_id) = open_challenge();
    let jset: HashSet<String> = [owner.pub_hex()].into_iter().collect();
    let vote = signed(AccountTx::DataVote(DataVoteTx {
        voter_pub: owner.pub_hex(), challenge_id, support: false, nonce: 1, sig: vec![],
    }), &owner);
    assert!(!led.apply_data_tx(&vote, 3, &jset), "owner defending own entry must be rejected");
}

#[test]
fn challenge_below_quorum_is_rejected_and_refunds_owner() {
    use palimpsest_core::token::CHALLENGE_QUORUM;
    let (mut led, owner, challenger, jurors, data_id, challenge_id) = open_challenge();
    let owner_addr = address(&owner.pub_hex());
    let bal_before = led.balance(&owner_addr);
    // only (QUORUM - 1) jurors uphold — below quorum
    let jset: HashSet<String> = jurors.iter().map(|k| k.pub_hex()).collect();
    for jk in jurors.iter().take(CHALLENGE_QUORUM - 1) {
        let vote = signed(AccountTx::DataVote(DataVoteTx {
            voter_pub: jk.pub_hex(), challenge_id: challenge_id.clone(), support: true,
            nonce: 0, sig: vec![],
        }), jk);
        assert!(led.apply_data_tx(&vote, 3, &jset));
    }
    let chal_addr = address(&challenger.pub_hex());
    let chal_before = led.balance(&chal_addr); // stake already escrowed out
    led.resolve_expired_challenges(2 + palimpsest_core::token::CHALLENGE_WINDOW);
    // below quorum => NOT upheld: entry stays active, and the challenger's stake
    // is forfeited to the owner (lying/failed challenge costs).
    assert_eq!(led.registry[&data_id]["status"].as_str(), Some("active"),
               "sub-quorum challenge must not revoke the entry");
    assert_eq!(led.balance(&owner_addr), bal_before + 500_000,
               "owner must be refunded exactly the challenger's forfeited stake");
    // the challenger seized nothing back — its escrowed stake is gone to the owner
    assert_eq!(led.balance(&chal_addr), chal_before,
               "rejected challenger recovers nothing");
}

// --- snapshot (de)serialization hardening (task 94) -------------------------

#[test]
fn snapshot_roundtrips_and_rejects_malformed() {
    use serde_json::json;
    // a populated ledger (balances, nonces, a registry entry, an open challenge)
    let (led, _o, _c, _j, _d, _cid) = open_challenge();

    // a well-formed snapshot round-trips to the exact same root
    let good = led.to_value();
    let back = TokenLedger::from_value(&good).expect("valid snapshot must round-trip");
    assert_eq!(back.root(), led.root(), "round-trip must preserve the ledger root");

    // a non-integer balance is rejected (would corrupt supply / panic on math)
    let mut bad = led.to_value();
    let addr = bad["balances"].as_object().unwrap().keys().next().unwrap().clone();
    bad["balances"][addr.as_str()] = json!("not-a-number");
    assert!(TokenLedger::from_value(&bad).is_none(), "string balance must be rejected");

    // a registry entry missing a field is rejected (would panic apply_reward)
    let mut bad = led.to_value();
    let did = bad["registry"].as_object().unwrap().keys().next().unwrap().clone();
    bad["registry"][did.as_str()].as_object_mut().unwrap().remove("stake");
    assert!(TokenLedger::from_value(&bad).is_none(), "registry entry missing stake must be rejected");

    // a challenge whose vote list isn't an array of strings is rejected
    let mut bad = led.to_value();
    let cid = bad["challenges"].as_object().unwrap().keys().next().unwrap().clone();
    bad["challenges"][cid.as_str()]["votes_for"] = json!([1, 2, 3]);
    assert!(TokenLedger::from_value(&bad).is_none(), "non-string vote list must be rejected");

    // a missing top-level section is rejected
    let mut bad = led.to_value();
    bad.as_object_mut().unwrap().remove("challenges");
    assert!(TokenLedger::from_value(&bad).is_none(), "missing section must be rejected");
}

// --- signing-preimage framing is injective (task 96) ------------------------

#[test]
fn frame_resists_delimiter_injection() {
    // The classic collision the old '|'-joined signing strings allowed:
    // join(["a", "b|c"]) == "a|b|c" == join(["a|b", "c"]). Length-prefix framing
    // must keep these distinct, so a field's contents can never be re-parsed as
    // a different field split (which would give two txs the same txid).
    assert_ne!(core::frame(&[b"a", b"b|c"]), core::frame(&[b"a|b", b"c"]));
    // and empty-field boundaries are unambiguous too
    assert_ne!(core::frame(&[b"", b"ab"]), core::frame(&[b"a", b"b"]));
    // identical inputs still frame identically (determinism)
    assert_eq!(core::frame(&[b"x", b"yz"]), core::frame(&[b"x", b"yz"]));
}
