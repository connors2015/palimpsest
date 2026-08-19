"""Run the §12.3 red-team and print an honest verdict. `python3 -m rig.redteam_report`"""

from .redteam import (TH_CLEAN, TH_DELTA_Z, TH_ORACLE, TH_RANDOM, detected_by,
                      run_redteam)


def main():
    r = run_redteam(seed=0)
    A, B = r["A"], r["B"]

    print("=" * 72)
    print("  PALIMPSEST §12.3 red-team — can a stealthy backdoor beat detection?")
    print("=" * 72)
    print(f"\nbase clean accuracy {A['base_clean']:.3f}; honest delta norm "
          f"~{A['honest_norm']:.2f}. A probe 'fires' above: clean>{TH_CLEAN}, "
          f"random>{TH_RANDOM}, |delta_z|>{TH_DELTA_Z}, oracle>{TH_ORACLE}.")

    print("\n--- A. single-delta attacks vs the probe battery ---")
    print(f"{'strategy':<10}{'clean_kept':>11}{'backdoor':>10}   "
          f"{'clean':>6}{'random':>7}{'delta':>7}{'oracle':>7}   caught_by")
    for a in A["results"]:
        d = detected_by(a)
        flags = "".join(k[0].upper() if v else "·" for k, v in d.items())
        print(f"{a.strategy:<10}{a.clean_acc_kept:>11.3f}{a.backdoor_success:>10.3f}   "
              f"{a.p_clean:>6.3f}{a.p_random:>7.3f}{a.p_delta_z:>7.1f}{a.p_oracle:>7.3f}   {flags}")
    print("  (C=clean-loss  R=random-trigger  D=delta-anomaly  O=oracle/known-trigger)")

    print("\n--- B. slow-drip coalition + excision recovery ---")
    print("  backdoor success by block: " +
          " ".join(f"{v:.2f}" for v in B["curve"][::max(1, len(B['curve']) // 8)]))
    print(f"  final poisoned:  backdoor={B['poisoned_backdoor']:.3f} "
          f"clean={B['poisoned_clean']:.3f}")
    print(f"  after excision:  backdoor={B['excised_backdoor']:.3f} "
          f"clean={B['excised_clean']:.3f}   (§10.4 replay without the coalition)")
    z = B["max_coal_z"]
    hidden = "INVISIBLE (below the honest band)" if z < TH_DELTA_Z else "flagged in this toy"
    print(f"  max coalition per-delta anomaly z = {z:+.1f} -> {hidden}: each drip "
          f"delta is no larger than an honest one, so anomaly detection misses it")

    # ---- honest verdicts ----
    stealthy = next(a for a in A["results"] if a.strategy == "stealthy")
    blind_input_fails = (stealthy.p_clean < TH_CLEAN and stealthy.p_random < TH_RANDOM
                         and stealthy.p_oracle > TH_ORACLE)
    oracle_always = all(a.p_oracle > TH_ORACLE for a in A["results"])
    excision_recovers = (B["excised_backdoor"] < 0.1
                         and B["excised_clean"] > B["poisoned_clean"] - 0.15)
    drip_accumulates = B["poisoned_backdoor"] > 0.5

    print("\n--- verdicts (honest) ---")
    def line(ok, s): print(f"  {'✓' if ok else '✗'} {s}")
    line(blind_input_fails,
         "Blind input-space probing FAILS on a stealthy OOD-triggered backdoor "
         "(clean-loss and in-distribution probes miss it) — scale-independent")
    line(oracle_always,
         "A KNOWN trigger is trivially detectable (oracle) — detection needs the "
         "trigger, via disclosure/bounty/the attacker using it")
    line(drip_accumulates,
         "Slow-drip coalition accumulates a strong backdoor across blocks")
    line(excision_recovers,
         "Replay-excision REMOVES a discovered backdoor at modest clean cost — "
         "the design's durable, detection-independent guarantee")

    print("\n  CONCLUSION: poisoning is NOT provably prevented by scoring+probes.")
    print("  The real defenses are (a) staked data-admission raising the cost of")
    print("  getting poison admitted, and (b) replay-excision to recover once a")
    print("  backdoor is identified. The residual — an undetected, never-triggered")
    print("  stealthy backdoor — is real and must be disclosed, not papered over.")
    print("=" * 72)

    ok = blind_input_fails and oracle_always and excision_recovers and drip_accumulates
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
