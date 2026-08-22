# The Genesis Ceremony

How the real Sestrian network is born. Everything here is decided, published,
and verifiable **before** block 1 exists — credibility is set at launch and
cannot be retrofitted (WHITEPAPER §9.8). This document is the checklist and the
script; each item names the mechanism that already implements it.

## 0. Principles (all already protocol invariants)

- **From scratch, on-chain.** Genesis weights are a deterministic random
  initialization from a *public seed* — no pretrained artifact anyone must
  trust. Every parameter of the model is thereafter explainable as a sum of
  signed, attributed, replayable deltas (§3.1).
- **Fair launch.** The genesis ledger has **zero balances**. No premine, no
  pre-sale, no allocation. Every grain is minted by a block reward for
  verifiable work (`rig/token.py`, mirrored bit-exact in `node/core`).
- **Bytes forever; RoPE positions.** Vocabulary = 256; no tokenizer; no learned
  position table — context is a runtime/market choice (§3.1).
- **The founding corpus is a contribution, not a gift.** It enters as registry
  entry zero owned by the founder's wallet at a published weight, earning the
  data share under exactly the rules any later contributor faces — including
  challengeability (§7.2 challenge market).

## 1. Published launch parameters (the genesis file)

A single JSON document, hashed and pinned, containing:

| Parameter | Value at ceremony | Where enforced |
|---|---|---|
| `model_config` | layers / heads / width / training block size | `client/gpt.py` GPTConfig |
| `genesis_seed` | derived from a public randomness beacon (below) | `apply_genesis` (numpy, version-independent) |
| `genesis_state_root` | sha256 of the quantized genesis vector | `rig/chain.state_root` — anyone recomputes |
| `emission` | BASE_REWARD, HALVING_BLOCKS, SUNSET_HEIGHT | `rig/token.py` / `node/core/token.rs` |
| `reward_split` | 7000/1000/2000 bps miners/proposer/data | same |
| `challenge_params` | CHALLENGE_WINDOW, PROPOSER_LOOKBACK | same |
| `data_contributor` | founder wallet address | genesis registry entry zero |
| `genesis_data_weight` | GENESIS_DATA_WEIGHT | same |
| `founding_corpus_hash` | **`85aa06fba4ef397b19bc5bc8e62d394bdb067b5eddde418ef5f4680ce1aae3ae`** (18,087,897,989 bytes · 48,284 documents · built 2026-08-20) | registry entry `data_hash`; corpus pinned on the CAS/DA layer |
| `founding_corpus` | **decided: public-domain only** — ~48k English Project Gutenberg books (~21 GB), built + hashed by `scripts/build_founding_corpus.py` with a per-shard manifest. No web crawl, no share-alike, no gated sources: the founding entry earns the founder's share and is challengeable by design, so its provenance is bulletproof. Code, Wikipedia, and web-scale text enter later through OTHER contributors' staked submissions and the §10.2 campaign track — the data economy working as intended. | `founding_manifest.json` |
| `block_interval` | seconds per round | node config |
| `bootstrap_peers` | seed-node multiaddrs (first public seed: `/ip4/169.58.211.248/udp/9800/quic-v1`) | node config |

**Rule: any change to this table after ceremony is a hard fork by definition.**

## 2. Seed derivation — nobody chooses the genesis weights

`genesis_seed = sha256("sestrian-genesis" || drand_round_R_signature)` where
`R` is a **pre-announced future round** of the drand public randomness beacon
(the League of Entropy). Because `R` is announced before its value exists,
neither the team nor anyone else can grind the initialization. Anyone can
verify: fetch round `R`, hash, compare. (`apply_genesis` then expands the seed
with numpy's byte-stable RNG, so the same seed yields bit-identical weights on
every platform — already verified cross-machine MPS/CUDA/CPU.)

## 3. The founding wallet

- Generated **fresh, offline**, on a machine the founder trusts
  (`python -m client.wallet new` on an air-gapped box; the dev wallet used
  during testnet is retired). Mnemonic backup written down; encrypted wallet
  file backed up separately (see wallet hardening).
- Only the **address** enters the genesis file. The key never touches a server.

## 4. The founding data transaction

The corpus enters through the standard admission path, visible in block 1's
lineage: registry entry zero (`seed_genesis_data`) carries the founder's
address, the corpus content hash, and the published weight. The corpus bytes
are pinned content-addressed (CAS/Bitswap — `client/cas.py`; the DA layer at
scale) so any node can fetch and hash-check exactly what the model eats.
It is **challengeable like any entry** (validity or ownership, §7.2) — the
founder holds no special immunity.

## 5. Ceremony procedure (the runbook)

1. **T−7 days**: publish the genesis file *minus* `genesis_seed` /
   `genesis_state_root`, naming drand round `R`. Publish repo tag, binary
   checksums, seed-node addresses.
2. **T−0**: drand round `R` lands. Anyone (including us) computes
   `genesis_seed`, runs `apply_genesis`, publishes `genesis_state_root`.
   Independent parties confirm the root.
3. **Launch**: seed nodes + founder nodes start with the complete genesis file.
   Block 1 is mined by whoever gets there first — emissions begin, the founding
   registry entry starts earning its weighted data share.
4. **T+window**: the founding corpus sits in its public challenge window like
   any submission.

## 6. Preconditions before the ceremony can run

- [x] Token legal posture — **founder's decision: no counsel engaged at launch.**
      The network fair-launches with no sale, no premine, and no profit
      promises; the founder's data-contributor share is publicly disclosed in
      the genesis parameters. Revisit before any exchange listing or any
      conversion of founder holdings — those are the events that change the
      legal character of the token, not its existence.
- [ ] External audit of consensus + ledger (rig is the spec; `node/core` golden-vector-pinned)
- [x] NAT traversal live — AutoNAT/DCUtR/relay-v2 shipped in the Rust node; the
      first PUBLIC seed+relay is up (`/ip4/169.58.211.248/udp/9800/quic-v1`),
      and a fresh node dialing only that multiaddr connects and agrees on genesis.
- [x] Wallet hardening shipped (encrypted files, BIP39 mnemonics, checksummed pal1… addresses)
- [x] Real corpus decision + license posture — public-domain-only Gutenberg
      (composition table above); pipeline + manifest in
      `scripts/build_founding_corpus.py`; hash lands in this file when the
      ceremony build runs.
- [ ] Repo public; binaries reproducibly built and checksummed

## 7. What the testnet already rehearsed

Every mechanism above is running today on the internal testnet: from-scratch
genesis with a published seed (1337), the founder wallet earning the data share
per block, transfers and data-lane txs settling through `ledger_root`, the
seed node on the cluster, and the Rust devnet converging byte-identically.
The ceremony is those same steps with a beacon-derived seed, a fresh wallet,
the real corpus, and the world watching.
