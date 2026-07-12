"""Diag E — where is the headroom in the 3.75 longitudinal model, and how does it couple to the
fundus-only levers?

The longitudinal model is: with-prior records → persistence (bounded by the 2.76 dB test-retest floor,
~unimprovable); first-visit records (no prior) → the FUNDUS-ONLY branch. So the only improvable part of
the 3.75 is the first-visit stratum, and it improves exactly when fundus-only improves. This quantifies
the coupling: a fundus-only ΔMAE (e.g. the disc crop's −0.146, or a better encoder) flows into the
longitudinal number scaled by the first-visit fraction. No torch.

  python decoder/diag_longitudinal_headroom.py
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = os.path.join(ROOT, "data", "vf_tests", "grape_longitudinal.json")

# documented strata MAEs (eval_strata_longitudinal.py / CLAUDE.md)
MAE_PRIOR = 3.185      # with-prior stratum (persistence; ~test-retest bounded, floor 2.76)
MAE_FIRST = 4.55       # first-visit stratum (fundus-only branch)
FUNDUS_DISC_GAIN = 0.146   # measured disc-crop fundus-only pooled improvement (p1disc vs m1sev)


def main():
    recs = json.load(open(LONG))
    n = len(recs)
    prior = [r for r in recs if r.get('has_prior')]
    first = [r for r in recs if not r.get('has_prior')]
    fp, ff = len(prior) / n, len(first) / n
    print("=" * 88)
    print(f"Diag E — longitudinal headroom  ({n} records)")
    print("-" * 88)
    print(f"  with-prior : {len(prior):>4}  ({fp:5.1%})  → persistence, MAE≈{MAE_PRIOR:.3f} "
          f"(bounded by 2.76 test-retest floor — ~unimprovable)")
    print(f"  first-visit: {len(first):>4}  ({ff:5.1%})  → FUNDUS-ONLY branch, MAE≈{MAE_FIRST:.3f} "
          f"(this is the improvable part)")
    pooled = fp * MAE_PRIOR + ff * MAE_FIRST
    print(f"  implied pooled ≈ {pooled:.3f}  (matches the reported ~3.75)")
    print("-" * 88)
    # severity split of the first-visit (fundus-only) records
    for lab, lo, hi in [("severe<15", -1, 15), ("moderate", 15, 22), ("mild≥22", 22, 99)]:
        k = sum(lo <= r['mean_db'] < hi for r in first)
        print(f"    first-visit {lab:<9}: {k:>3} records")
    print("-" * 88)
    print("COUPLING: a fundus-only improvement ΔMAE flows into the longitudinal number as ΔMAE × "
          f"{ff:.2f} (first-visit share).")
    print(f"  e.g. the disc crop's fundus-only −{FUNDUS_DISC_GAIN:.3f} → longitudinal "
          f"−{FUNDUS_DISC_GAIN * ff:.3f} (→ ~{pooled - FUNDUS_DISC_GAIN * ff:.3f}).")
    print("  So a BETTER ENCODER that lifts fundus-only also lifts the longitudinal SOTA — the two "
          "levers stack. Metadata/prior-VF only help the with-prior stratum, which is already floor-bound.")
    print("=" * 88)
    json.dump({'n': n, 'first_visit_frac': ff, 'with_prior_frac': fp, 'implied_pooled': pooled},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto",
                                "diag_longitudinal_headroom.json"), 'w'), indent=2, default=float)


if __name__ == "__main__":
    main()
