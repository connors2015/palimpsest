"""The data economy, end to end — contributors earn a bonus, then royalties.

Ties Stage 1 (pay for contribution) and Stage 2 (pay for downstream usage) into
one loop and shows the thing that matters: a data owner's earnings growing as
their data keeps being used in answers people pay for.

  1. contributors each submit a signed data shard to a channel;
  2. admission prices each shard by marginal value × channel rate → a signing
     bonus (Stage 1), vested with clawback;
  3. the model trains on the admitted data, keeping checkpoints;
  4. queries arrive (some domains more popular than others); each paid query is
     attributed across shards by influence, and a royalty slice is paid to the
     owners whose data shaped the answer (Stage 2);
  5. earnings accrue: a niche contributor earns a bonus and a trickle; a
     contributor whose data the world keeps asking about earns a bonus and a
     rising royalty stream.
"""

from dataclasses import dataclass, field

import numpy as np

from . import attribution as attr
from .crypto import Key
from .data import DataLedger, DataTx, channel_rate, content_hash, marginal_value
from .data_model import (DomainModel, N_DOMAINS, TOTAL_CLASSES, domain_batch,
                         make_rules, mixed_batch)

FEE_PER_QUERY = 1.0
ROYALTY_SHARE = 0.30          # fraction of each inference fee that flows to data
NETWORK_RATE = 120.0         # value → tokens rate for the admission bonus (§9.2)
CKPT_STEPS = (15, 40, 80, 140, 220, 320)     # TracIn checkpoints (chain checkpoints)
PROJ_DIM = 512
CONTRIBUTORS = ["alice", "bob", "carol", "dave"]
CHANNELS = ["professional", "professional", "general", "research"]
# how often each domain is queried — "popularity" of each contributor's expertise
POPULARITY = np.array([0.40, 0.30, 0.20, 0.10])


@dataclass
class Result:
    owners: list
    bonus: dict
    royalties: dict          # owner -> list of cumulative royalty over rounds
    total: dict
    fidelity: float
    attribution_acc: float


def run(seed=0, serve_rounds=40, queries_per_round=40, verbose=False):
    rng = np.random.default_rng(seed)
    model = DomainModel(dim=4 + 16, classes=TOTAL_CLASSES, hidden=48)
    rules = make_rules(rng)
    keys = {name: Key.generate(name.encode().ljust(32, b"0")) for name in CONTRIBUTORS}
    proj = attr.Projector(model.param_count, proj_dim=PROJ_DIM)
    ledger = DataLedger(vest_blocks=15)

    # each contributor owns one domain's data shard
    shards = {d: domain_batch(np.random.default_rng(100 + d), 256, d, rules)
              for d in range(N_DOMAINS)}
    owner_of = {d: CONTRIBUTORS[d] for d in range(N_DOMAINS)}
    channel_of = {d: CHANNELS[d] for d in range(N_DOMAINS)}

    # 1–2. admit + price (Stage 1) against a FRESH model, so a shard's bonus
    # reflects how much the knowledge it brings is worth × its channel rate.
    vec = model.init(rng)
    probes = [mixed_batch(np.random.default_rng(900 + i), 200, rules) for i in range(6)]
    for d in range(N_DOMAINS):
        mv = marginal_value(model, vec, shards[d], probes)
        tx = DataTx(owner=keys[owner_of[d]].pub, channel=channel_of[d],
                    content_hash=content_hash(shards[d]), n_examples=256,
                    da_pointer=f"da://{d}", shard_id=d).signed(keys[owner_of[d]])
        assert tx.verify()
        ledger.admit(tx, mv * NETWORK_RATE, block=0)

    # 3. train the model, keeping checkpoints for TracIn attribution
    ckpts = []
    for step in range(1, max(CKPT_STEPS) + 1):
        vec = model.train_step(vec, mixed_batch(rng, 128, rules), lr=0.5, steps=1)
        if step in CKPT_STEPS:
            ckpts.append(vec.copy())
    shard_sk = {d: attr.shard_sketch(model, ckpts, shards[d], proj) for d in range(N_DOMAINS)}

    # attribution quality (for the record)
    fid_qs = [mixed_batch(np.random.default_rng(700 + i), 1, rules) for i in range(24)]
    fid_qs = [(x[0], int(y[0])) for x, y in fid_qs]
    fidelity = attr.sketch_fidelity(model, ckpts[-1], list(shards.values()), fid_qs, proj)

    # 4–5. serve queries; attribute; pay royalties; accrue
    royalty_paid = {}
    hist = {name: [] for name in CONTRIBUTORS}
    correct = tot = 0
    for r in range(serve_rounds):
        doms = rng.choice(N_DOMAINS, size=queries_per_round, p=POPULARITY)
        for qd in doms:
            x, _ = domain_batch(rng, 1, qd, rules)
            ans = int(model.predict(vec, x)[0])
            qsk = attr.answer_sketch(model, ckpts, x[0], ans, proj)
            w = attr.attribute(qsk, shard_sk)
            attr.route_royalty(FEE_PER_QUERY, ROYALTY_SHARE, w,
                               {d: owner_of[d] for d in range(N_DOMAINS)}, royalty_paid)
            correct += (max(w, key=w.get) == qd); tot += 1
        ledger.tick(block=r + 1)
        for name in CONTRIBUTORS:
            hist[name].append(royalty_paid.get(name, 0.0))
        if verbose and (r % 8 == 0 or r == serve_rounds - 1):
            snap = ", ".join(f"{n}:{royalty_paid.get(n,0):.0f}" for n in CONTRIBUTORS)
            print(f"  round {r:>2}  royalties: {snap}", flush=True)

    bonus = {name: 0.0 for name in CONTRIBUTORS}
    for acc in ledger.accounts.values():
        bonus[owner_of[acc.shard_id]] += acc.granted
    total = {n: bonus[n] + royalty_paid.get(n, 0.0) for n in CONTRIBUTORS}
    return Result(owners=CONTRIBUTORS, bonus=bonus,
                  royalties={n: hist[n] for n in CONTRIBUTORS}, total=total,
                  fidelity=fidelity, attribution_acc=correct / max(1, tot))


def main():
    print("=" * 70)
    print("  PALIMPSEST — the data economy (bonus + downstream royalties)")
    print("=" * 70)
    r = run(verbose=True)
    print(f"\nattribution: {r.attribution_acc*100:.0f}% of queries credit the right "
          f"contributor; sketch fidelity {r.fidelity:.2f}")
    print(f"\n{'contributor':<10}{'channel':<14}{'popularity':>11}{'bonus':>8}"
          f"{'royalties':>11}{'total':>9}")
    for i, name in enumerate(CONTRIBUTORS):
        print(f"{name:<10}{CHANNELS[i]:<14}{POPULARITY[i]:>11.0%}"
              f"{r.bonus[name]:>8.1f}{r.royalties[name][-1]:>11.1f}{r.total[name]:>9.1f}")
    print("\n  the more a contributor's data is used in answers, the more it earns —")
    print("  royalties track popularity, and keep paying as long as data stays useful.")
    print("=" * 70)
    # popular contributor out-earns the niche one on royalties
    ok = (r.royalties["alice"][-1] > r.royalties["dave"][-1]
          and r.attribution_acc > 0.6)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
