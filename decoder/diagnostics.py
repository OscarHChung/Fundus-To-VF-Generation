"""Session-3 diagnostic harness — establish the HONEST baseline and locate the signal.

The Session-2 "ceilings" (overall MAE ~4.12, per-point corr ~0.58) were measured on a
per-RECORD split with 62% fellow-eye leakage and only 5 severe test eyes (see
results/auto/iterations.md). Before trusting any ceiling we (1) re-split per PATIENT with
severity stratification so no eye's fellow eye crosses the train/val line and all 47 severe
eyes are evaluated out-of-fold, and (2) measure how much per-point signal is actually
extractable, via cheap baselines + linear probes on the frozen RETFound features.

Commands (run from repo root):
  python decoder/diagnostics.py split        # build per-patient stratified 5-fold CV jsons
  python decoder/diagnostics.py baseline      # naive predictors + target-variance decomposition
  python decoder/diagnostics.py probe         # linear probes on frozen encoder features (CV)
  python decoder/diagnostics.py eval-champ     # re-eval current champion: per-patient CV vs leaky

`split`/`baseline` are instant (no encoder). `probe`/`eval-champ` lazily load RETFound.
"""
import os, sys, json
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.join(CURRENT_DIR, "..")
VF_DIR      = os.path.join(BASE_DIR, "data", "vf_tests")
FULL_JSON   = os.path.join(VF_DIR, "grape_new_vf_tests.json")
CV_DIR      = os.path.join(CURRENT_DIR, "results", "cv")
MASKED_VALUE_THRESHOLD = 99.0

# ── VF grid constants (replicated from training.py so split/baseline need no encoder) ──
mask_OD = np.array([
    [False, False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, False, False, True,  True,  True,  True,  False, False]
], dtype=bool)
valid_indices_od = [i for i, v in enumerate(mask_OD.flatten()) if v]
mask_OS          = np.fliplr(mask_OD)
valid_indices_os = [i for i, v in enumerate(mask_OS.flatten()) if v]
NUM_VALID_POINTS = len(valid_indices_od)   # 52


# ==============================================================
# Data helpers
# ==============================================================
def load_records(path=FULL_JSON):
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else list(data.values())


def vec52(record):
    """Return the 52 valid sensitivities in QUERY order for this eye (OD/OS aware),
    with masked points as np.nan."""
    hvf = np.array(record['hvf'], dtype=np.float64).flatten()
    lat = str(record.get('Laterality', 'OD')).strip().upper()
    vi  = valid_indices_od if lat.startswith('OD') else valid_indices_os
    v   = hvf[vi]
    v[v >= MASKED_VALUE_THRESHOLD] = np.nan
    return v


def eye_severity(record):
    v = vec52(record)
    return float(np.nanmean(v))


# ==============================================================
# Per-patient, severity-stratified K-fold
# ==============================================================
def build_patient_folds(records, k=5, seed=42):
    """Group eyes by PatientID (keeps both eyes of a patient on the SAME side of every
    split → no fellow-eye leakage). Order patient-groups by mean severity and assign
    them round-robin to k folds so every fold has a proportional share of severe eyes.
    Returns list[k] of record-index lists (the VAL indices for each fold)."""
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(r.get('PatientID', i), []).append(i)
    # group severity = mean over its eyes
    gids = list(groups.keys())
    gsev = {g: np.mean([eye_severity(records[i]) for i in groups[g]]) for g in gids}
    gids.sort(key=lambda g: gsev[g])           # ascending severity (severe first)
    rng = np.random.RandomState(seed)
    # round-robin with a per-block shuffle to decorrelate from severity ties
    folds = [[] for _ in range(k)]
    for blk_start in range(0, len(gids), k):
        block = gids[blk_start:blk_start + k]
        order = list(range(len(block)))
        rng.shuffle(order)
        for slot, oi in enumerate(order):
            folds[slot].extend(groups[block[oi]])
    return folds


def cmd_split(k=5, path=FULL_JSON, cv_dir=CV_DIR, dev_prefix="grape_ppat"):
    os.makedirs(cv_dir, exist_ok=True)
    records = load_records(path)
    folds = build_patient_folds(records, k=k)
    # leakage assertion: no PatientID appears in two folds
    fold_pat = [set(records[i]['PatientID'] for i in f) for f in folds]
    for a in range(k):
        for b in range(a + 1, k):
            assert not (fold_pat[a] & fold_pat[b]), f"patient leak across folds {a},{b}"
    print(f"Per-patient stratified {k}-fold over {len(records)} records / "
          f"{len(set(r['PatientID'] for r in records))} patients:")
    for j, f in enumerate(folds):
        sev = [eye_severity(records[i]) for i in f]
        n_sev = sum(s < 15 for s in sev)
        print(f"  fold {j}: {len(f):3d} recs | {len(fold_pat[j]):3d} patients | "
              f"severe(<15)={n_sev:2d} | meandB {np.mean(sev):.1f}")
        val   = [records[i] for i in f]
        train = [records[i] for jj, ff in enumerate(folds) if jj != j for i in ff]
        json.dump(train, open(os.path.join(cv_dir, f"fold{j}_train.json"), 'w'))
        json.dump(val,   open(os.path.join(cv_dir, f"fold{j}_val.json"),   'w'))
    # also a single clean 80/20 dev split (fold 0 = val) for fast iteration
    val0   = [records[i] for i in folds[0]]
    train0 = [records[i] for jj in range(1, k) for i in folds[jj]]
    json.dump(train0, open(os.path.join(VF_DIR, f"{dev_prefix}_train.json"), 'w'))
    json.dump(val0,   open(os.path.join(VF_DIR, f"{dev_prefix}_val.json"),   'w'))
    print(f"\nWrote {k} folds → {cv_dir}")
    print(f"Wrote clean dev split → {dev_prefix}_train.json ({len(train0)}) / "
          f"{dev_prefix}_val.json ({len(val0)})")


# ==============================================================
# Pooled metrics (truth/pred in dB; lists of per-eye 52-vectors with nan masks)
# ==============================================================
def pooled_metrics(preds, trues):
    """preds/trues: list of np arrays (52,) with nan at masked points. Returns dict."""
    P, T, eyecorrs = [], [], []
    for p, t in zip(preds, trues):
        m = ~np.isnan(t)
        pp, tt = p[m], t[m]
        P.append(pp); T.append(tt)
        if m.sum() >= 5 and np.std(pp) > 1e-6 and np.std(tt) > 1e-6:
            eyecorrs.append(np.corrcoef(pp, tt)[0, 1])
    P = np.concatenate(P); T = np.concatenate(T)
    err = P - T; ae = np.abs(err)
    def band(lo, hi):
        m = (T >= lo) & (T < hi)
        return (float(ae[m].mean()) if m.any() else float('nan')), int(m.sum())
    floor, fn = band(0, 10); deep, dn = band(0, 16)
    slope, intc = np.polyfit(T, P, 1)
    corr = float(np.corrcoef(P, T)[0, 1])
    return {'mae': float(ae.mean()), 'floor': floor, 'floor_n': fn, 'deep': deep, 'deep_n': dn,
            'slope': float(slope), 'corr': corr, 'bias': float(err.mean()),
            'eyecorr': float(np.mean(eyecorrs)) if eyecorrs else float('nan'),
            'sig_ratio': float(np.std(P) / (np.std(T) + 1e-8))}


def fmt(m):
    return (f"MAE {m['mae']:.3f} | floor {m['floor']:.2f}(n{m['floor_n']}) | "
            f"deep {m['deep']:.2f} | slope {m['slope']:.3f} | corr {m['corr']:.3f} | "
            f"eyeCorr {m['eyecorr']:.3f} | bias {m['bias']:+.2f} | σp/σt {m['sig_ratio']:.2f}")


def stratified_report(preds, trues, verbose=True):
    """Bucket per-eye records by TRUE mean dB and report pooled metrics per severity band.
    Required alongside any pooled number so a milder per-visit distribution can't mislead:
    a low pooled MAE driven by easy mild fields is exposed by the severe-band MAE."""
    bands = {"severe": (-1, 15), "moderate": (15, 22), "mild": (22, 99)}
    out = {}
    for name, (lo, hi) in bands.items():
        idx = [i for i, t in enumerate(trues) if lo <= np.nanmean(t) < hi]
        if not idx:
            out[name] = {"n_eyes": 0, "mae": float("nan")}
            continue
        m = pooled_metrics([preds[i] for i in idx], [trues[i] for i in idx])
        m["n_eyes"] = len(idx)
        out[name] = m
    if verbose:
        print("  severity-stratified (per-eye true mean dB):")
        for name in ("severe", "moderate", "mild"):
            r = out[name]
            if r["n_eyes"]:
                print(f"    {name:8s} n={r['n_eyes']:3d} | MAE {r['mae']:.3f} | "
                      f"floor {r.get('floor', float('nan')):.2f} | "
                      f"slope {r.get('slope', float('nan')):.3f}")
    return out


# ==============================================================
# Baselines + target-variance decomposition
# ==============================================================
def cmd_baseline():
    records = load_records()
    folds = build_patient_folds(records, k=5)
    # ----- target variance decomposition (how much signal is spatial vs severity) -----
    all_within, all_between, eye_means, npts = [], [], [], []
    grand_vals = []
    for r in records:
        v = vec52(r); v = v[~np.isnan(v)]
        eye_means.append(v.mean()); npts.append(len(v)); grand_vals.extend(v.tolist())
    grand_vals = np.array(grand_vals); mu = grand_vals.mean()
    eye_means = np.array(eye_means); npts = np.array(npts)
    between = float((npts * (eye_means - mu) ** 2).sum())
    within  = 0.0
    for r in records:
        v = vec52(r); v = v[~np.isnan(v)]
        within += float(((v - v.mean()) ** 2).sum())
    total = between + within
    print("── Target variance decomposition (all 263 eyes) ──")
    print(f"  grand mean {mu:.2f} dB | total SS {total:.0f}")
    print(f"  BETWEEN-eye (severity) {between/total*100:.1f}%  |  "
          f"WITHIN-eye (spatial)    {within/total*100:.1f}%")
    print(f"  → spatial fraction {within/total*100:.1f}% is the headroom EyeCorr/slope must capture\n")

    # ----- naive predictors on per-patient 5-fold CV -----
    print("── Naive baselines (per-patient 5-fold CV, out-of-fold pooled) ──")
    for name in ('global_mean', 'perpoint_mean', 'eye_oracle_mean'):
        preds, trues = [], []
        for j, f in enumerate(folds):
            tr = [records[i] for jj, ff in enumerate(folds) if jj != j for i in ff]
            # build per-point train mean in OD/OS query order
            od = np.vstack([vec52(r) for r in tr if str(r.get('Laterality','OD')).upper().startswith('OD')])
            os_= np.vstack([vec52(r) for r in tr if not str(r.get('Laterality','OD')).upper().startswith('OD')])
            pp_od = np.nanmean(od, axis=0); pp_os = np.nanmean(os_, axis=0)
            gmean = np.nanmean(np.concatenate([od.flatten(), os_.flatten()]))
            for i in f:
                r = records[i]; t = vec52(r)
                is_od = str(r.get('Laterality','OD')).upper().startswith('OD')
                if name == 'global_mean':
                    p = np.full(NUM_VALID_POINTS, gmean)
                elif name == 'perpoint_mean':
                    p = (pp_od if is_od else pp_os).copy()
                else:  # eye_oracle_mean — predict this eye's OWN true mean everywhere (severity oracle)
                    p = np.full(NUM_VALID_POINTS, np.nanmean(t))
                preds.append(p); trues.append(t)
        print(f"  {name:16s}: {fmt(pooled_metrics(preds, trues))}")
    print("\n  (eye_oracle_mean = best possible if you knew each eye's mean but NOTHING spatial.")
    print("   The gap from perpoint_mean→a real model's MAE is the spatial signal a model adds.)")


# ==============================================================
# main
# ==============================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if cmd == "split":
        cmd_split()
    elif cmd == "split-long":
        sys.path.insert(0, os.path.abspath(BASE_DIR))
        import build_longitudinal_grape as B
        cmd_split(path=B.OUT,
                  cv_dir=os.path.join(CURRENT_DIR, "results", "cv_long"),
                  dev_prefix="grape_long")
    elif cmd == "baseline":
        cmd_baseline()
    else:
        print(f"unknown command {cmd!r}")
