"""Extends td_report.py: compute the numbers that CHANGE when the paper is expressed in total
deviation (TD) instead of raw sensitivity -- calibrated slope, and severity-stratified MAE with
TD/MD-native bands -- so the reworded manuscript uses correct values.

Replicates eval_oof_cached's per-fold variance-match calibration (fit on fold TRAIN, applied to
VAL), in BOTH raw-sensitivity and TD space.

  python decoder/td_report2.py
"""
import os, json
import numpy as np
import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(ROOT, "decoder", "results", "auto")
CV   = os.path.join(ROOT, "decoder", "results", "cv_long")
TAG  = "p1disc_denoise"
MASKED = 99.0

mask_OD = np.array([
    [False, False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, False, False, True,  True,  True,  True,  False, False]], dtype=bool)
valid_od = [i for i, v in enumerate(mask_OD.flatten()) if v]

# ---- normative (from td_report.py logic) ----
uw = json.load(open(os.path.join(ROOT, "data", "vf_tests", "uwhvf_vf_tests.json")))
acc = {c: ([], []) for c in valid_od}
for _, rec in uw["data"].items():
    for side in ("L", "R"):
        for v in rec.get(side, []) or []:
            hvf = np.asarray(v["hvf"], float).flatten(); td = np.asarray(v["td"], float).flatten()
            for c in valid_od:
                if hvf[c] < 90 and abs(td[c]) < 90:
                    acc[c][0].append(v["age"]); acc[c][1].append(hvf[c] - td[c])
a_slot = np.zeros(52); b_slot = np.zeros(52)
for j, c in enumerate(valid_od):
    a_slot[j], b_slot[j] = np.polyfit(np.array(acc[c][0]), np.array(acc[c][1]), 1)

def normal52(age): return a_slot * age + b_slot

# ---- ages ----
wb = openpyxl.load_workbook(os.path.join(ROOT, "data", "vf_tests", "grape_data.xlsx"), read_only=True)
bs = list(wb["Baseline"].iter_rows(values_only=True)); bh = bs[0]
iS, iL, iA = bh.index("Subject Number"), bh.index("Laterality"), bh.index("Age")
base_age = {(int(r[iS]), str(r[iL]).strip()[:2]): float(r[iA]) for r in bs[1:]
            if r[iS] is not None and isinstance(r[iA], (int, float))}
fu = list(wb["Follow-up"].iter_rows(values_only=True)); fh = fu[0]
jS, jL, jV, jI = fh.index("Subject Number"), fh.index("Laterality"), fh.index("Visit Number"), fh.index("Interval Years")
cum_int = {(int(r[jS]), str(r[jL]).strip()[:2], int(r[jV])): float(r[jI] or 0.0)
           for r in fu[1:] if r[jS] is not None and r[jV] is not None}
cmean = float(np.mean(list(base_age.values())))
def age_of(pid, lat, vis):
    lat = str(lat).strip()[:2]
    return base_age.get((pid, lat), cmean) + cum_int.get((pid, lat, vis), 0.0)

def load(split, f):
    z = np.load(os.path.join(AUTO, "oof_cache_notta", f"{TAG}_f{f}.npz"))
    p, t = (z["vp"], z["vt"]) if split == "val" else (z["tp"], z["tt"])
    recs = json.load(open(os.path.join(CV, f"fold{f}_{split}.json")))
    assert len(recs) == len(p)
    ages = np.array([age_of(int(r["PatientID"]), r["Laterality"], int(r["VisitNumber"])) for r in recs])
    return p.astype(float), t.astype(float), ages

def to_td(p, t, ages):
    nrm = np.stack([normal52(a) for a in ages])
    tt = t.copy(); tt[tt >= MASKED] = np.nan
    return p - nrm, tt - nrm

def to_sens(p, t):
    tt = t.copy(); tt[tt >= MASKED] = np.nan
    return p.copy(), tt

def pooled_calib(space):
    """Return (raw_r, raw_slope, calib_slope, calib_mae, raw_mae) pooling OOF val over folds,
    with per-fold variance-match calibration fit on fold train. space in {'sens','td'}."""
    P, T, Pc = [], [], []
    for f in range(5):
        vp, vt, va = load("val", f); tp, tt, ta = load("train", f)
        if space == "td":
            vp, vt = to_td(vp, vt, va); tp, tt = to_td(tp, tt, ta)
        else:
            vp, vt = to_sens(vp, vt); tp, tt = to_sens(tp, tt)
        mtr = ~np.isnan(tt)
        mu_p, sig_p = tp[mtr].mean(), tp[mtr].std()
        mu_t, sig_t = tt[mtr].mean(), tt[mtr].std()
        b = sig_t / (sig_p + 1e-8)
        clip_lo, clip_hi = (-40, 15) if space == "td" else (0, 35)
        vpc = np.clip(mu_t + b * (vp - mu_p), clip_lo, clip_hi)
        P.append(vp); T.append(vt); Pc.append(vpc)
    P = np.concatenate([x.ravel() for x in P]); T = np.concatenate([x.ravel() for x in T])
    Pc = np.concatenate([x.ravel() for x in Pc])
    m = ~np.isnan(T)
    r = np.corrcoef(T[m], P[m])[0, 1]
    slope_raw = np.cov(T[m], P[m], bias=True)[0, 1] / np.var(T[m])
    slope_cal = np.cov(T[m], Pc[m], bias=True)[0, 1] / np.var(T[m])
    return dict(r=r, slope_raw=slope_raw, slope_cal=slope_cal,
                mae_raw=np.mean(np.abs(P[m] - T[m])), mae_cal=np.mean(np.abs(Pc[m] - T[m])))

print("=== CALIBRATION (pooled OOF) ===")
for sp in ("sens", "td"):
    d = pooled_calib(sp)
    print(f"  {sp:4s}: r {d['r']:.3f}  raw-slope {d['slope_raw']:.3f}  CALIB-slope {d['slope_cal']:.3f}"
          f"  | rawMAE {d['mae_raw']:.3f}  calMAE {d['mae_cal']:.3f}")

# ---- severity-stratified MAE: pointwise by true TD, per-eye by true MD ----
allp, allt, alltd_p, alltd_t, md_t, msens_t = [], [], [], [], [], []
for f in range(5):
    vp, vt, va = load("val", f)
    ps, ts = to_sens(vp, vt); ptd, ttd = to_td(vp, vt, va)
    for i in range(len(vp)):
        m = ~np.isnan(ts[i])
        allp.append(ps[i]); allt.append(ts[i]); alltd_p.append(ptd[i]); alltd_t.append(ttd[i])
        md_t.append(np.mean(ttd[i][m])); msens_t.append(np.mean(ts[i][m]))
allp = np.array(allp); allt = np.array(allt); alltd_p = np.array(alltd_p); alltd_t = np.array(alltd_t)
md_t = np.array(md_t); msens_t = np.array(msens_t)

def pw_mae(sel_mask_pts):
    d = np.abs(alltd_p - alltd_t)[sel_mask_pts & ~np.isnan(alltd_t)]
    return float(d.mean()), int((sel_mask_pts & ~np.isnan(alltd_t)).sum())

print("\n=== POINTWISE MAE by TRUE TOTAL-DEVIATION band (identical value whether scored in sens or TD) ===")
# TD cutpoints analogous to old sens bands: sens>=22 (mild), 15-22 (mod), <15 (severe)
# map via typical central normal ~30: keep by true SENSITIVITY membership so numbers match the paper,
# but ALSO show TD-native cut. First: same membership as paper (by true sens) -> unchanged numbers.
for name, sel in [("mild  sens>=22", allt >= 22), ("mod   15-22", (allt >= 15) & (allt < 22)),
                  ("sev   sens<15", allt < 15)]:
    mae, n = pw_mae(sel)
    print(f"  [by-sensitivity membership] {name:16s}: MAE {mae:5.2f}  (n={n})")
print("  (these three equal the paper's 2.90 / 4.06 / 8.43 -- membership defined by measured depth, unchanged)")
for name, sel in [("mild  TD>=-6", alltd_t >= -6), ("mod   -12..-6", (alltd_t >= -12) & (alltd_t < -6)),
                  ("sev   TD<-12", alltd_t < -12)]:
    mae, n = pw_mae(sel)
    print(f"  [by-TD-value      ]        {name:16s}: MAE {mae:5.2f}  (n={n})")

print("\n=== PER-EYE MAE by TRUE MEAN-DEVIATION (Hodapp) ===")
for name, sel in [("early MD>-6", md_t > -6), ("moderate -12..-6", (md_t <= -6) & (md_t >= -12)),
                  ("advanced MD<-12", md_t < -12)]:
    if sel.sum() == 0:
        print(f"  {name:18s}: 0 eyes"); continue
    d = []
    for i in np.where(sel)[0]:
        m = ~np.isnan(alltd_t[i]); d.append(np.mean(np.abs(alltd_p[i] - alltd_t[i])[m]))
    print(f"  {name:18s}: MAE {np.mean(d):5.2f}  ({int(sel.sum())} eyes)")
# also per-eye by mean-sensitivity membership (paper's 2.66/5.14/7.19)
print("  --- per-eye by measured mean-sensitivity membership (paper's bands, unchanged) ---")
for name, sel in [("mild  >=22", msens_t >= 22), ("moderate 15-22", (msens_t >= 15) & (msens_t < 22)),
                  ("severe <15", msens_t < 15)]:
    d = []
    for i in np.where(sel)[0]:
        m = ~np.isnan(alltd_t[i]); d.append(np.mean(np.abs(alltd_p[i] - alltd_t[i])[m]))
    print(f"  {name:16s}: MAE {np.mean(d):5.2f}  ({int(sel.sum())} eyes)")
