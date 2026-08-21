# Palimpsest — Production Readiness

The go/no-go tracker for the phased launch. Phase 0 = the rig (done). Phase 1 =
invite-only devnet with known participants. Phase 2 = testnet. Phase 3 = open
mainnet. This maps every hardening task to its state and gates each phase.

## Legend
- ✅ implemented + tested in the shipping node
- 🧪 core/primitive implemented + golden-tested; live integration/validation
  pending the testnet
- 📐 designed (rig / whitepaper); not yet in the node
- ☐ operational item; manifest/script written, applied per-environment

## Consensus safety — ✅ COMPLETE (blocks Phase 1)
- ✅ Block height linkage (90) · delta-length guard (91) · n_txs/height-0 (95)
- ✅ Challenge quorum + disinterested jurors (93)
- ✅ Snapshot ledger validation (94)
- ✅ Length-prefixed signing preimages (96)
- ✅ Overflow-safe arithmetic, numpy parity, dust documented (97)
- ✅ **Non-forgeable work**: header.work = vrf_work(VRF proof), verified in
  validate_block; VRF proposer sortition wired in, fixed-rotation SPOF removed (92, 113)
- ✅ **Byzantine-robust aggregation** at low miner counts (always trim ≥1 at k≥3) (110)
- Golden vectors: 17 families incl. negative, overflow, VRF-chain, and
  low-count-robustness cases; Rust == Python. 35 Rust tests; devnet + soak
  (kill/restart) converge.

## Runtime & DoS hardening — ✅ COMPLETE (blocks Phase 1)
- ✅ Bounded mempools/caches + admission gating (98/99)
- ✅ Admin-token-gated mutating API; balance-before-write upload (100)
- ✅ Byte-budgeted, continuous sync (101)
- ✅ Durable fsync persistence; fatal-on-write-fail; torn-line self-heal (102/103)
- ✅ Single-writer data-dir lock (104)
- ✅ Gossip peer scoring + tight sync limits (105)
- ✅ Keys off argv/git, zeroized (106)
- ✅ Trainer watchdog + clock guard (107)
- ✅ SIGTERM graceful shutdown → final snapshot (131)

## Trust model — 🧪/📐 (blocks Phase 3; invite-only mitigates for Phase 1/2)
- 🧪 DA layer primitive: erasure coding + Merkle sampling (`core::da`) (111)
  - ☐ node routing: disperse on submit, sample on validate, reconstruct on
    replay (112) — integration + testnet validation
- 🧪 Proposer sortition primitive: verifiable stake-weighted VRF (`core::lottery`) (113)
  - ☐ wire eligibility into validate_block + produce; VRF-derived work (92)
  - 📐 threshold-BLS beacon for unbiasability (`rig/beacon.py`)
- 🧪 Capacity retarget controller (`core::capacity`) (117)
  - ☐ enforce the work quota in validate_block
- 📐 Delta scoring: held-out-shard loss, commit-reveal committee, audit (108)
- ✅ Delta stake bond (admission cost) — lock/return done + golden-tested (109);
  slashing on proven fraud couples to scoring (testnet)
- 📐 Byzantine-robust aggregation at low miner counts (110)
- 📐 Dtx cross-inclusion (anti-censorship) (114)
- 📐 Verified fee-bearing inference + receipts (116)

**Phase-1 mitigation:** run invite-only with known miners so the missing delta
scoring can't be exploited; keep mutating API endpoints token-gated/disabled.

## Operations — ☐ manifests/scripts ready; apply per environment
- ☐ Persistent-volume StatefulSet (118) · prebuilt image + CI push (120)
- ✅ Prometheus /metrics endpoint + alert rules (121)
- ☐ Backup/restore script (122)
- ☐ TLS termination for non-loopback API (123) — see below
- ☐ Second bootstrap/DA anchor + failover (119)

## Process
- ✅ CI: warning-clean build + tests + golden parity; image build (124)
- 🧪 node/net tests: store lock/torn-line, mempool window, API auth (125) —
  expand alongside integrations
- 📐 adversarial/chaos suite (126) · cross-machine e2e + soak (128)
- ☐ Python reference suite pinned + green in CI (127)
- ✅ Threat model (132) · this readiness doc (133)

## Remaining (all Phase-2/3 by nature — need live infra + off-chain compute)
These 7 cannot be completed-and-validated in a single-machine coding
environment; the readiness gate for each is the testnet, not a golden vector.
- 108 delta scoring (held-out-shard loss) — needs off-chain model execution; a
  self-reported score without the commit-reveal committee would be gameable, so
  it genuinely gates on the testnet's off-chain verification
- ✅ 109 delta stake bond DONE (see above); slashing gates on 108 (testnet)
- ✅ 111/112 DA routing DONE at the storage layer: bodies are erasure-coded into
  Merkle-committed shards on write, and replay/sync reconstruct from any K shards
  instead of hard-stopping on a missing body (tested: recover from K, fail below
  K; devnet converges with live dispersal). The multi-node piece — distributing
  shards across peers + availability-sampling over gossip — is the testnet extension.
- 114 Dtx cross-inclusion — needs the gossip/scoring layer
- 115 chunked sparse aggregation — a perf optimization validated at real scale
- 116 verified fee-bearing inference — needs off-chain serving + attestation

## Phase gates
- **Phase 1 (invite devnet): ✅ READY** — consensus safety complete (incl.
  non-forgeable work + robust aggregation), runtime hardening complete, ops
  manifests written. Soak (kill/restart) converges. Apply the ops manifests to
  the cluster to launch.
- **Phase 2 (testnet):** DA routing (111/112) live and converging under churn;
  a second anchor (✅ documented, provision it); load/soak on real hosts.
- **Phase 3 (open mainnet):** delta scoring + stake/slash (108/109) enforced;
  fee-bearing inference (116); threshold-BLS beacon; **external audit sign-off**;
  repo public + reproducible checksummed builds.
