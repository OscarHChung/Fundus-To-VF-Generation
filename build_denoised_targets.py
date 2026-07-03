"""METHOD B — per-eye TRAINING-target denoising (training-only; eval stays RAW).

Single-visit VF labels are corrupted by perimetric test-retest noise (σ≈2.4 dB/point). For each
target visit we fit a per-point ROBUST linear trend (Theil–Sen) of sensitivity vs the eye's
`interval_years` over ALL that eye's visits, then EVALUATE the trend at the TARGET visit's date.
Because the target date is one of the eye's own visit dates this is interpolation (not
extrapolation), so it denoises the target's test-retest error while preserving real progression —
strictly better than naive averaging (which biases toward the mean date).

Fundus-only inference is unaffected: this only refines the TRAINING targets. The eval always
scores against the RAW observed VF (see improvement_playbook §5 / §8). Per-patient folds are
eye-disjoint, so a val eye's timeline never touches a train target → no leakage.

  python build_denoised_targets.py    # -> data/vf_tests/grape_longitudinal_denoised.json

Output = a lookup keyed "PatientID_Laterality_VisitNumber" -> denoised 8x9 hvf (masked cells kept
masked). The training Dataset loads it under --denoised-targets and swaps TRAIN targets only.
"""
import os, json
from collections import defaultdict
import numpy as np

import build_longitudinal_grape as BL

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "vf_tests", "grape_longitudinal_denoised.json")
MASK = 99.0
MIN_VISITS = 3          # eyes with fewer visits fall back to the raw field
MIN_POINT_OBS = 3       # per-point: need ≥3 valid observations to fit a trend, else raw


def theil_sen(t, v):
    """Robust line fit: slope = median pairwise slope, intercept = median(v - slope·t)."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    slopes = []
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            if abs(dt) > 1e-9:
                slopes.append((v[j] - v[i]) / dt)
    if not slopes:
        return 0.0, float(np.median(v))
    slope = float(np.median(slopes))
    intercept = float(np.median(v - slope * t))
    return slope, intercept


def denoise_field(intervals, fields_72, target_idx):
    """intervals: list[n] years; fields_72: list[n] of flat-72 dB (MASK≥99 = masked);
    target_idx: index of the visit being denoised. Returns a denoised flat-72 for that visit.

    Per point: if ≥MIN_POINT_OBS valid obs across visits → Theil–Sen trend evaluated at the
    target interval, clipped to the point's own observed range ±3 dB (interpolation guard).
    Otherwise (or if the target point is masked, or <MIN_VISITS visits) → keep the raw value."""
    intervals = np.asarray(intervals, float)
    F = np.asarray(fields_72, float)                     # (n, 72)
    target = F[target_idx].copy()
    if len(intervals) < MIN_VISITS:
        return target                                    # whole-eye fallback: raw
    out = target.copy()
    t_tgt = intervals[target_idx]
    for p in range(F.shape[1]):
        if target[p] >= MASK:                            # masked at target → stay masked
            continue
        col = F[:, p]
        valid = col < MASK
        if valid.sum() < MIN_POINT_OBS:
            continue                                     # per-point fallback: raw
        tv, vv = intervals[valid], col[valid]
        slope, intercept = theil_sen(tv, vv)
        pred = slope * t_tgt + intercept
        lo, hi = vv.min() - 3.0, vv.max() + 3.0          # interpolation guard (no wild values)
        out[p] = float(np.clip(pred, lo, hi))
    return out


def build(xlsx_path=BL.XLSX, fundus_dir=BL.FUNDUS_DIR, out_path=OUT):
    have = set(os.listdir(fundus_dir))
    rows = BL.read_followup(xlsx_path)
    timeline = defaultdict(list)
    for r in rows:
        timeline[(r["subject"], r["laterality"])].append(r)
    for k in timeline:
        timeline[k].sort(key=lambda x: x["visit"])

    lookup = {}
    diffs, n_denoised, n_fallback = [], 0, 0
    for r in rows:
        if r["cfp"] not in have:
            continue
        eye = timeline[(r["subject"], r["laterality"])]
        intervals = [v["interval"] for v in eye]
        fields = [np.array(v["hvf"], float).flatten() for v in eye]
        tgt_idx = next(i for i, v in enumerate(eye) if v["visit"] == r["visit"])
        raw = fields[tgt_idx]
        den = denoise_field(intervals, fields, tgt_idx)
        vmask = raw < MASK
        if len(intervals) >= MIN_VISITS:
            diffs.append(float(np.abs(den[vmask] - raw[vmask]).mean())); n_denoised += 1
        else:
            n_fallback += 1
        key = f"{int(r['subject'])}_{r['laterality']}_{int(r['visit'])}"
        lookup[key] = den.reshape(8, 9).tolist()

    json.dump(lookup, open(out_path, "w"))
    md = float(np.mean(diffs)) if diffs else 0.0
    print(f"denoised {n_denoised} targets (mean |denoised−raw| = {md:.2f} dB); "
          f"{n_fallback} whole-eye fallbacks (<{MIN_VISITS} visits)")
    print(f"-> {out_path}  ({len(lookup)} keys)")
    return lookup


if __name__ == "__main__":
    build()
