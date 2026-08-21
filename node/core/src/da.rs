//! Data-availability layer: erasure coding + availability sampling (§3.3).
//!
//! A faithful port of the Python reference (`rig/da.py` + `rig/merkle.py`) — the
//! golden vectors pin this to it byte-for-byte. A delta body is Reed-Solomon
//! coded over GF(256) into `n` shards such that ANY `k` reconstruct it, the
//! shards are Merkle-committed (the root IS the tx's DA pointer), and a verifier
//! samples a few random shards with inclusion proofs. Because unrecoverability
//! needs more than `n-k` shards missing, a handful of samples detects a
//! withholding attack with high probability — without downloading the body.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// GF(256) — the AES field: generator 0x03, modulus x^8+x^4+x^3+x+1 (0x11b)
// ---------------------------------------------------------------------------

struct Gf {
    exp: [u8; 512],
    log: [u8; 256],
}

fn gf() -> &'static Gf {
    static GF: OnceLock<Gf> = OnceLock::new();
    GF.get_or_init(|| {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u16 = 1;
        for i in 0..255 {
            exp[i] = x as u8;
            log[x as usize] = i as u8;
            x ^= x << 1;
            if x & 0x100 != 0 {
                x ^= 0x11b;
            }
        }
        for i in 255..512 {
            exp[i] = exp[i - 255];
        }
        Gf { exp, log }
    })
}

fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let g = gf();
    // log a + log b < 512, so the 512-entry exp table needs no modulo (matches
    // the Python `_EXP[_LOG[a] + _LOG[b]]`).
    g.exp[g.log[a as usize] as usize + g.log[b as usize] as usize]
}

fn gf_inv(a: u8) -> u8 {
    let g = gf();
    g.exp[255 - g.log[a as usize] as usize]
}

/// n×k Vandermonde: row i uses evaluation point (i+1); any k rows are invertible.
fn vandermonde(n: usize, k: usize) -> Vec<Vec<u8>> {
    let g = gf();
    (0..n)
        .map(|i| {
            (0..k)
                .map(|j| {
                    // V[i][j] = EXP[(LOG[i+1] * j) % 255]; i+1 >= 1 always.
                    let idx = (g.log[i + 1] as usize * j) % 255;
                    g.exp[idx]
                })
                .collect()
        })
        .collect()
}

/// Invert a k×k GF(256) matrix by Gauss-Jordan; None if singular.
fn mat_inv(m: &[Vec<u8>]) -> Option<Vec<Vec<u8>>> {
    let k = m.len();
    // augmented [m | I]
    let mut a: Vec<Vec<u8>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..k).map(|j| if i == j { 1u8 } else { 0u8 }));
            r
        })
        .collect();
    for col in 0..k {
        let piv = (col..k).find(|&r| a[r][col] != 0)?;
        a.swap(col, piv);
        let inv = gf_inv(a[col][col]);
        for v in a[col].iter_mut() {
            *v = gf_mul(*v, inv);
        }
        for r in 0..k {
            if r != col && a[r][col] != 0 {
                let f = a[r][col];
                let pivot_row = a[col].clone();
                for (x, y) in a[r].iter_mut().zip(pivot_row.iter()) {
                    *x ^= gf_mul(f, *y);
                }
            }
        }
    }
    Some(a.into_iter().map(|row| row[k..].to_vec()).collect())
}

/// One output row = XOR_r ( coeffs[r] · byte_rows[r] ), over GF(256).
fn gf_combine(coeffs: &[u8], byte_rows: &[Vec<u8>]) -> Vec<u8> {
    let len = byte_rows[0].len();
    let mut acc = vec![0u8; len];
    for (&c, row) in coeffs.iter().zip(byte_rows) {
        if c != 0 {
            for (a, &r) in acc.iter_mut().zip(row) {
                *a ^= gf_mul(c, r);
            }
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Erasure coding
// ---------------------------------------------------------------------------

/// Split `body` into k data rows and expand to n shards (any k reconstruct).
pub fn encode(body: &[u8], k: usize, n: usize) -> Vec<Vec<u8>> {
    assert!(0 < k && k <= n && n <= 255, "require 0 < k <= n <= 255");
    let pad = (k - body.len() % k) % k;
    let mut data = body.to_vec();
    data.extend(std::iter::repeat(0u8).take(pad));
    let l = data.len() / k;
    let rows: Vec<Vec<u8>> = (0..k).map(|r| data[r * l..(r + 1) * l].to_vec()).collect();
    let v = vandermonde(n, k);
    (0..n).map(|i| gf_combine(&v[i], &rows)).collect()
}

/// Recover the body from any k shards ({index: bytes}); None if fewer than k or
/// the chosen shard rows are singular (never happens for distinct Vandermonde
/// rows).
pub fn reconstruct(shards: &BTreeMap<usize, Vec<u8>>, k: usize, orig_len: usize) -> Option<Vec<u8>> {
    if shards.len() < k {
        return None;
    }
    let idx: Vec<usize> = shards.keys().copied().take(k).collect();
    let v = vandermonde(idx.iter().max().copied().unwrap() + 1, k);
    let sub: Vec<Vec<u8>> = idx.iter().map(|&i| v[i].clone()).collect();
    let inv = mat_inv(&sub)?;
    let cols: Vec<Vec<u8>> = idx.iter().map(|i| shards[i].clone()).collect();
    let mut out = Vec::with_capacity(k * cols[0].len());
    for r in 0..k {
        out.extend(gf_combine(&inv[r], &cols));
    }
    out.truncate(orig_len);
    Some(out)
}

// ---------------------------------------------------------------------------
// Merkle commitment over shards (mirrors rig/merkle.py exactly)
// ---------------------------------------------------------------------------

fn leaf_hash(data: &[u8]) -> [u8; 32] {
    let mut m = Sha256::new();
    m.update([0u8]); // domain-separated leaf
    m.update(data);
    m.finalize().into()
}

fn node_hash(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut m = Sha256::new();
    m.update([1u8]); // domain-separated internal node
    m.update(a);
    m.update(b);
    m.finalize().into()
}

/// Build the tree as levels[0]=leaf hashes .. levels[last]=[root]. An odd node
/// is promoted (hashed with itself).
pub fn merkle_build(leaves: &[Vec<u8>]) -> Vec<Vec<[u8; 32]>> {
    assert!(!leaves.is_empty(), "need at least one leaf");
    let mut level: Vec<[u8; 32]> = leaves.iter().map(|x| leaf_hash(x)).collect();
    let mut levels = vec![level.clone()];
    while level.len() > 1 {
        let mut nxt = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let a = level[i];
            let b = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            nxt.push(node_hash(&a, &b));
            i += 2;
        }
        levels.push(nxt.clone());
        level = nxt;
    }
    levels
}

/// A proof step: sibling on the ('L'eft | 'R'ight) of the path node.
pub type ProofStep = (char, [u8; 32]);

pub fn merkle_proof(levels: &[Vec<[u8; 32]>], index: usize) -> Vec<ProofStep> {
    let mut path = Vec::new();
    let mut idx = index;
    for level in &levels[..levels.len() - 1] {
        if idx % 2 == 0 {
            let sib = if idx + 1 < level.len() { level[idx + 1] } else { level[idx] };
            path.push(('R', sib));
        } else {
            path.push(('L', level[idx - 1]));
        }
        idx /= 2;
    }
    path
}

pub fn merkle_verify(data: &[u8], path: &[ProofStep], root: &[u8; 32]) -> bool {
    let mut h = leaf_hash(data);
    for &(side, sib) in path {
        h = if side == 'L' { node_hash(&sib, &h) } else { node_hash(&h, &sib) };
    }
    &h == root
}

// ---------------------------------------------------------------------------
// DA blob + availability sampling
// ---------------------------------------------------------------------------

pub struct DaBlob {
    pub shards: Vec<Vec<u8>>,
    pub orig_len: usize,
    pub k: usize,
    pub n: usize,
    levels: Vec<Vec<[u8; 32]>>,
}

impl DaBlob {
    pub fn root(&self) -> [u8; 32] {
        self.levels[self.levels.len() - 1][0]
    }
    pub fn proof(&self, i: usize) -> Vec<ProofStep> {
        merkle_proof(&self.levels, i)
    }
}

/// Erasure-code + Merkle-commit a body into a samplable DA blob.
pub fn disperse(body: &[u8], k: usize, n: usize) -> DaBlob {
    let shards = encode(body, k, n);
    let levels = merkle_build(&shards);
    DaBlob { shards, orig_len: body.len(), k, n, levels }
}

/// The DA pointer committed on-chain: `da://` + first 32 hex chars of the root.
pub fn da_pointer(root: &[u8; 32]) -> String {
    format!("da://{}", &hex::encode(root)[..32])
}

/// Deterministic sample indices from a seed — validators must sample the SAME
/// shards to reach the same availability verdict, so this replaces the
/// reference's RNG with a seed-driven (verifiable) selection: a distinct
/// permutation prefix derived by hashing (seed, counter).
pub fn sample_indices(seed: &[u8], n: usize, num_samples: usize) -> Vec<usize> {
    let want = num_samples.min(n);
    let mut chosen: Vec<usize> = Vec::with_capacity(want);
    let mut counter: u64 = 0;
    while chosen.len() < want {
        let mut m = Sha256::new();
        m.update(seed);
        m.update(counter.to_le_bytes());
        let d: [u8; 32] = m.finalize().into();
        let pick = (u64::from_le_bytes(d[..8].try_into().unwrap()) % n as u64) as usize;
        if !chosen.contains(&pick) {
            chosen.push(pick);
        }
        counter += 1;
    }
    chosen
}

/// Availability sampling: the body is deemed available iff every sampled shard
/// is present AND proves against the blob's root. Unrecoverability needs
/// > n-k missing, so a few samples catch withholding with high probability.
pub fn sample_available(
    available: &BTreeMap<usize, Vec<u8>>,
    blob: &DaBlob,
    seed: &[u8],
    num_samples: usize,
) -> bool {
    let root = blob.root();
    for i in sample_indices(seed, blob.n, num_samples) {
        match available.get(&i) {
            None => return false, // a sampled shard is missing
            Some(shard) => {
                if !merkle_verify(shard, &blob.proof(i), &root) {
                    return false; // ...or forged
                }
            }
        }
    }
    true
}
