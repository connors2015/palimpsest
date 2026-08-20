"""The data economy — Stage 1 pricing, Stage 2 royalties, and the hard parts.

Covers: signed submissions, marginal-value pricing (gap-filling > junk, duplicate
~0 = Sybil-resistant), channel differentiation, vesting + clawback, sketch
fidelity (JL), attribution ground truth, royalties tracking usage, and the
poisoning-pays-then-clawed-back economics (§7.2, §9.2, §10.2, §10.4).
"""

import numpy as np

from rig.attribution import (Projector, attribute, answer_sketch, shard_sketch,
                             sketch_fidelity)
from rig.crypto import Key
from rig.data import (DataAccount, DataLedger, DataTx, channel_rate,
                      content_hash, marginal_value)
from rig.data_model import (DomainModel, N_DOMAINS, TOTAL_CLASSES, domain_batch,
                            junk_batch, make_rules, mixed_batch)


def _trained_on_012(rng):
    """Base model that knows domains 0-2 but is MISSING domain 3 (a gap)."""
    m = DomainModel(dim=4 + 16, classes=TOTAL_CLASSES, hidden=48)
    rules = make_rules(rng)
    vec = m.init(rng)

    def b012(r, n):
        dom = r.integers(0, 3, n)
        from rig.data_model import _rows
        return _rows(r, n, dom, rules, N_DOMAINS, 16, 3)
    for _ in range(150):
        vec = m.train_step(vec, b012(rng, 128), lr=0.5, steps=1)
    return m, vec, rules


# -- Stage 1: authentication + pricing ---------------------------------------
def test_datatx_signed_and_forgery_rejected():
    k = Key.generate(b"alice".ljust(32, b"0"))
    shard = domain_batch(np.random.default_rng(1), 8, 0, make_rules(np.random.default_rng(0)))
    tx = DataTx(owner=k.pub, channel="professional", content_hash=content_hash(shard),
                n_examples=8, da_pointer="da://x", shard_id=3).signed(k)
    assert tx.verify()
    attacker = Key.generate(b"m".ljust(32, b"0"))
    tx.sig = attacker.sign(tx.signing_bytes())
    assert not tx.verify()


def test_gapfilling_data_priced_above_junk_and_duplicates():
    rng = np.random.default_rng(0)
    m, vec, rules = _trained_on_012(rng)
    probes = [mixed_batch(np.random.default_rng(900 + i), 200, rules) for i in range(6)]
    gap = [marginal_value(m, vec, domain_batch(np.random.default_rng(i), 256, 3, rules), probes)
           for i in range(6)]
    junk = [marginal_value(m, vec, junk_batch(np.random.default_rng(500 + i), 256), probes)
            for i in range(6)]
    covered = [marginal_value(m, vec, domain_batch(np.random.default_rng(50 + i), 256, i % 3, rules), probes)
               for i in range(6)]
    assert min(gap) > max(junk)            # gap-filling data is the most valuable
    assert min(gap) > max(covered)         # …above already-covered (duplicate) data
    assert abs(np.mean(covered)) < np.mean(gap)   # covered ≈ 0 (Sybil-resistant)


def test_channel_rate_differentiates_bonus():
    led = DataLedger()
    k = Key.generate(b"a".ljust(32, b"0"))
    for ch in ("research", "professional"):
        tx = DataTx(owner=k.pub, channel=ch, content_hash="h", n_examples=1,
                    da_pointer="d", shard_id=hash(ch) % 1000).signed(k)
        led.admit(tx, marginal_value=1.0, block=0)
    accts = {a.channel: a.granted for a in led.accounts.values()}
    assert accts["professional"] > accts["research"]     # same value, higher-rate channel pays more


# -- vesting + clawback ------------------------------------------------------
def test_bonus_vests_over_time():
    led = DataLedger(vest_blocks=10)
    led.accounts[0] = DataAccount(owner="a", shard_id=0, channel="general", granted=100.0)
    led.tick(5); assert 40 <= led.owner_balance("a") <= 60      # ~half vested
    led.tick(10); assert abs(led.owner_balance("a") - 100.0) < 1e-6


def test_clawback_revokes_unvested_bonus():
    led = DataLedger(vest_blocks=10)
    led.accounts[0] = DataAccount(owner="p", shard_id=0, channel="general", granted=100.0)
    led.tick(5)                                       # ~50 vested
    forfeited = led.clawback(0)
    assert 40 <= forfeited <= 60
    led.tick(20)                                      # no further vesting after revoke
    assert led.owner_balance("p") < 60


# -- Stage 2: attribution ----------------------------------------------------
def test_sketch_preserves_influence_signal():
    rng = np.random.default_rng(0)
    m = DomainModel(dim=4 + 16, classes=TOTAL_CLASSES, hidden=48)
    rules = make_rules(rng); vec = m.init(rng)
    for _ in range(200):
        vec = m.train_step(vec, mixed_batch(rng, 128, rules), lr=0.5, steps=1)
    proj = Projector(m.param_count, 512)
    shards = [domain_batch(np.random.default_rng(100 + d), 256, d, rules) for d in range(N_DOMAINS)]
    qs = [mixed_batch(np.random.default_rng(700 + i), 1, rules) for i in range(24)]
    qs = [(x[0], int(y[0])) for x, y in qs]
    assert sketch_fidelity(m, vec, shards, qs, proj) > 0.8      # JL fidelity


def test_attribution_credits_the_right_domain():
    rng = np.random.default_rng(0)
    m = DomainModel(dim=4 + 16, classes=TOTAL_CLASSES, hidden=48)
    rules = make_rules(rng); vec = m.init(rng)
    ckpts = []
    for s in range(1, 321):
        vec = m.train_step(vec, mixed_batch(rng, 128, rules), lr=0.5, steps=1)
        if s in (15, 40, 80, 140, 220, 320):
            ckpts.append(vec.copy())
    proj = Projector(m.param_count, 512)
    sk = {d: shard_sketch(m, ckpts, domain_batch(np.random.default_rng(100 + d), 256, d, rules), proj)
          for d in range(N_DOMAINS)}
    correct = tot = 0
    for qd in range(N_DOMAINS):
        for t in range(8):
            x, _ = domain_batch(np.random.default_rng(3000 + qd * 10 + t), 1, qd, rules)
            ans = int(m.predict(vec, x)[0])
            w = attribute(answer_sketch(m, ckpts, x[0], ans, proj), sk)
            correct += (max(w, key=w.get) == qd); tot += 1
    assert correct / tot > 0.7          # domain-d queries credit domain-d data


# -- end to end + hard parts -------------------------------------------------
def test_flywheel_royalties_track_usage():
    from rig.data_flywheel import run
    r = run(seed=0, serve_rounds=30, queries_per_round=40)
    assert r.attribution_acc > 0.7
    # the most-queried contributor out-earns the least-queried on royalties
    assert r.royalties["alice"][-1] > r.royalties["dave"][-1]


def test_poisoning_is_net_negative_after_clawback_and_slash():
    from rig.economics import StakeLedger
    led = DataLedger(vest_blocks=15)
    stake = StakeLedger()
    led.accounts[1] = DataAccount(owner="poison", shard_id=1, channel="professional",
                                  granted=300.0)
    stake.stake("poison", 200.0)
    royalties = 0.0
    for block in range(1, 16):
        led.tick(block)
        if block < 5:
            royalties += 12.0
        if block == 5:
            led.clawback(1)
            stake.slash("poison", "proven poisoning", "watcher")
    net = led.owner_balance("poison") + royalties - 200.0   # vested + royalties − slashed bond
    assert net < 0                       # poisoning does not pay
