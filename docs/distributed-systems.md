# The distributed-systems layer (Bitcoin-inspired)

The rig started with a single trusted coordinator dictating an append-only list.
This layer replaces the faked parts with real mechanisms, leaning on Bitcoin.
What each part maps to:

| Bitcoin | Palimpsest | Module |
|---|---|---|
| Signed transactions | Signed `BackpropTx` delta commitments (Ed25519) | `rig/crypto.py`, `rig/ed25519.py` |
| Block headers, prev-hash linking | `Header` committing prev_hash + state root + txset root + work | `rig/blockchain.py` |
| Independent full validation | `validate_block` from first principles vs parent state | `rig/blockchain.py` |
| Longest/heaviest chain | `BlockTree` heaviest-valid-chain fork choice | `rig/blockchain.py` |
| P2P gossip, mempool | `GossipNode` + `Network` (flood, mempool, orphan buffer, sync-on-heal) | `rig/p2p.py` |
| Difficulty retarget | `WritePriceController` holding delta-admission rate at target | `rig/economics.py` |
| — (novel) | Stake ledger + slashing on provable faults | `rig/economics.py` |

## What is now real

- **No coordinator.** Six nodes reach consensus by gossip + fork choice alone;
  `scripts/run p2p` shows them converge, fork under a network partition, and heal
  to one agreed history when reconnected.
- **Authenticated writes.** A delta counts only with a valid signature from the
  key that submitted it; forged signatures and mismatched/withheld DA bodies are
  rejected at validation and are slashable.
- **Trustless history.** Any node validates any block from first principles and
  replays the winning chain to reproduce its committed head state bit-exact.
- **Self-regulating admission.** The write-price homeostat drives the admitted
  rate to target and prices spam out, using Bitcoin's retarget math (damped,
  because delta submission is a sharper plant than hash power).
- **Economic backing.** Stake is slashable on proven faults; slashed stake pays
  the challenger a bounty and burns the rest.
- **Runs across machines.** `rig/lan.py` (coordinator form) already trains across
  the Mac + chris-server with byte-identical consensus; the gossip form is the
  coordinator-free successor.

## What is still faked or simplified (honest scope)

- **Gossip now runs over real async sockets** (`rig/gossip_net.py`) — verified
  coordinator-free across machines (2 nodes on the Mac + 1 on chris-server over
  Tailscale all converged to the identical head at height 32). The in-process
  `rig/p2p.py` remains as the deterministic, fully-testable model of the same
  logic. Remaining productionization: NAT traversal, peer scoring/eviction, and
  DoS resistance beyond the write-price homeostat.
- **The DA layer is modelled as gossip** — bodies travel with their tx. There is
  no erasure coding or availability sampling yet (§3.3); withholding is caught
  only because bodies are present to hash-check.
- **The randomness beacon is now a real threshold-BLS (drand-style) beacon**
  (`rig/beacon.py`): unbiasable, unpredictable, verifiable (§7.4). Remaining gap:
  it uses a trusted-dealer Shamir setup; production needs distributed key
  generation so no party ever holds the group secret, and it must be wired into
  live block production (currently proven standalone).
- **Block production is round-robin / designated proposers**, not a real
  leader-election or PoW/PoS lottery; fork choice is real but proposer selection
  is a stand-in.
- **Sybil resistance is economic only** (write-price + stake); there is no
  proof-of-work cost and no identity/stake registry on-chain yet.

## Next, in rough priority

1. An **unbiasable randomness beacon** (threshold VRF) — everything downstream
   (shard assignment, committee sampling, eval draws) hangs off it.
2. **Real async socket gossip** across machines (fold `rig/p2p.py` onto
   `rig/protocol.py`), then run a coordinator-free network on Mac + chris-server.
3. A real **data-availability layer** with erasure coding + sampling.
4. **Leader election / proposer lottery** to replace round-robin.
