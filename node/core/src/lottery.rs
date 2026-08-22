//! Verifiable proposer sortition (§7.4, interim) — mirrors `rig/lottery.py`.
//!
//! Replaces fixed round-robin rotation with per-height, stake-weighted,
//! verifiable eligibility. The per-height seed binds to the parent hash and
//! height; a proposer's VRF proof is a DETERMINISTIC Ed25519 signature over it
//! (only the key holder can produce it, anyone can verify it, unique per height);
//! eligibility requires the proof's hash, as a fraction of 2^256, to fall below
//! TARGET_PROPOSERS × stake/total_stake. Multiple or zero nodes may qualify in a
//! slot — heaviest-valid-chain fork choice settles it and no single node's
//! absence stalls the chain. The threshold-BLS beacon is the unbiasable upgrade.

use crate::{verify_sig, Key};
use sha2::{Digest, Sha256};

pub const TARGET_PROPOSERS: u64 = 2;

/// The per-height randomness seed, bound to the parent and height.
pub fn seed(prev_hash: &str, height: u64) -> [u8; 32] {
    let mut m = Sha256::new();
    m.update(format!("sestrian-lottery|{prev_hash}|{height}").as_bytes());
    m.finalize().into()
}

/// The proposer's VRF proof: a deterministic signature over the seed.
pub fn vrf_prove(key: &Key, prev_hash: &str, height: u64) -> Vec<u8> {
    key.sign(&seed(prev_hash, height))
}

/// A uniform 256-bit value only the key holder could have produced, as bytes
/// (big-endian) so it can be compared against the threshold without bignum deps.
pub fn vrf_output(proof: &[u8]) -> [u8; 32] {
    Sha256::digest(proof).into()
}

/// Fork-choice weight: leading zero bits of the VRF output + 1 (>= 1). Luckier
/// (smaller) output => more work, so the luckiest eligible proposer wins — and
/// work is non-forgeable (one VRF per proposer per height).
pub fn vrf_work(proof: &[u8]) -> u64 {
    let out = vrf_output(proof);
    let mut lz = 0u64;
    for &b in out.iter() {
        if b == 0 {
            lz += 8;
        } else {
            lz += b.leading_zeros() as u64;
            break;
        }
    }
    lz + 1
}

/// Eligible iff the VRF proof verifies for `pub_hex` at this height AND its
/// output falls under the stake-weighted threshold
/// (output < 2^256 · TARGET · stake/total_stake).
pub fn eligible(
    pub_hex: &str,
    proof: &[u8],
    prev_hash: &str,
    height: u64,
    stake: u64,
    total_stake: u64,
) -> bool {
    if !verify_sig(pub_hex, &seed(prev_hash, height), proof) {
        return false;
    }
    below_threshold(&vrf_output(proof), stake, total_stake)
}

/// Compare a 256-bit big-endian output against 2^256 · TARGET · stake/total.
/// Done with 512-bit integer math over u128 limbs so there are NO floats and no
/// bignum dependency — bit-identical to the Python reference's big-int compare.
fn below_threshold(output: &[u8; 32], stake: u64, total_stake: u64) -> bool {
    if total_stake == 0 || stake == 0 {
        return false; // threshold 0 — nothing is below it
    }
    // threshold = floor(2^256 * TARGET * stake / total_stake), capped at 2^256.
    // Compute numerator = TARGET * stake as u128 (no overflow: both <= supply),
    // then threshold = (2^256 * numerator) / total_stake. We avoid materializing
    // 2^256 by long-dividing the 320-bit value (numerator << 256) by total_stake
    // into a 256-bit quotient, then compare bytewise with `output`.
    let numerator = (TARGET_PROPOSERS as u128) * (stake as u128);
    // If numerator >= total_stake, threshold >= 2^256 (cap) => everything below.
    if numerator >= total_stake as u128 {
        return true;
    }
    // Long division: dividend is numerator followed by 32 zero bytes (i.e.
    // numerator * 2^256), divisor is total_stake. We only need the top 32 bytes
    // of the quotient (the value in [0, 2^256)).
    let q = mul2pow256_div(numerator, total_stake as u128); // 32-byte big-endian
    // eligible iff output < threshold(q)
    output.as_slice() < q.as_slice()
}

/// floor(numerator * 2^256 / divisor) as a 32-byte big-endian value, for
/// numerator < divisor (so the result is < 2^256). Schoolbook long division,
/// one output byte at a time, remainder carried in u128.
fn mul2pow256_div(numerator: u128, divisor: u128) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut rem = numerator; // numerator < divisor, so rem starts < divisor
    for byte in out.iter_mut() {
        // bring down 8 bits: rem = rem*256 (+ 0, since the low bytes are zero)
        let cur = rem << 8; // rem < divisor <= u128::MAX/256? divisor is u64 so fits
        *byte = (cur / divisor) as u8;
        rem = cur % divisor;
    }
    out
}
