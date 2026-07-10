"""Diag B — does GRAPE's clinical metadata add VF signal BEYOND the fundus photo?

The metadata lever only helps if age/gender/IOP/CCT/OCT-RNFL predict the VF that the photo MISSES.
We join baseline metadata (per eye) to the pooled fundus OOF preds and measure, leak-free (5-fold OOF
ridge, fit on train only):
  1. univariate corr of each field with true per-eye MD (which fields carry severity signal);
  2. metadata-alone OOF prediction of true MD;
  3. INCREMENTAL value over the fundus: does [fundus_pred_MD + metadata] beat [fundus_pred_MD] at
     predicting true MD?  (partial-corr + ΔMAE) — the money metric;
  4. crude translation to pooled VF MAE (pooled is ~fundus→MD dominated, so an MD-error cut ≈ a
     pooled-MAE cut of similar size on the severity channel).
No torch (uses the cached p1disc OOF npz + the parsed Baseline sheet).

  python decoder/diag_metadata_value.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_metadata_inventory import load_sheets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")
XLSX = os.path.join(ROOT, "data", "vf_tests", "grape_data.xlsx")

# Baseline sheet columns (1-indexed) → feature name (from diag_metadata_inventory output)
COLS = {'age': 3, 'gender': 4, 'iop': 5, 'cct': 6,
        'rnfl_g': 12, 'rnfl_1': 13, 'rnfl_2': 14, 'rnfl_3': 15, 'rnfl_4': 16}


def baseline_meta():
    """(subject, laterality) → {feature: float}. gender F/M → 0/1."""
    rows = load_sheets(XLSX)['Baseline']
    rnos = sorted(rows)
    meta = {}
    for r in rnos[1:]:
        d = rows[r]
        if 1 not in d or 2 not in d:
            continue
        subj = int(float(d[1])); lat = 'OD' if str(d[2]).startswith('OD') else 'OS'
        feat = {}
        for name, c in COLS.items():
            v = d.get(c)
            if name == 'gender':
                feat[name] = 1.0 if str(v).strip().upper().startswith('M') else 0.0
            else:
                try:
                    feat[name] = float(v)
                except (TypeError, ValueError):
                    feat[name] = np.nan
        meta[(subj, lat)] = feat
    return meta


def oof_ridge(X, y, folds, lam=1.0):
    """5-fold OOF ridge prediction of y from standardized X (fit per train fold)."""
    pred = np.full(len(y), np.nan)
    for f in range(5):
        tr = folds != f; va = folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd; Xva = (X[va] - mu) / sd
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        b = np.linalg.solve(A, Xtr.T @ (y[tr] - y[tr].mean()))
        pred[va] = Xva @ b + y[tr].mean()
    return pred


def spatial_probe(boot=5000, seed=42, lam=10.0):
    """Task C1 — does OCT RNFL-sector predict the within-eye VF *pattern* beyond the fundus?

    Diag B (severity `main()`) killed metadata for the eye-MEAN. Structure-function is fundamentally
    SECTORAL, so this asks the one untested question: after removing the fundus model's predicted
    within-eye residual pattern, does a leak-free linear map from the 4 RNFL sectors still track the
    true within-eye residual pattern? Gate (pre-registered): partial-corr >= 0.15 -> worth a small
    metadata-fusion head; < 0.15 -> RNFL-spatial is redundant with the disc-crop fundus -> metadata
    fully closed (severity AND spatial). Leak-free: RNFL->point map fit on each fold's TRAIN only.
    """
    import d2_template_partial as D2
    meta = baseline_meta()
    RN = ('rnfl_1', 'rnfl_2', 'rnfl_3', 'rnfl_4')
    recs = []   # (fold, pid, vp52, vt52, rnfl4)
    miss = 0
    for f in range(5):
        d = np.load(os.path.join(AUTO, 'oof_cache_notta', f'p1disc_f{f}.npz'))
        items = json.load(open(os.path.join(CV_DIR, f'fold{f}_val.json')))
        for i, it in enumerate(items):
            key = (int(it['PatientID']), 'OD' if str(it['Laterality']).startswith('OD') else 'OS')
            if key not in meta:
                miss += 1; continue
            recs.append((f, int(it['PatientID']), d['vp'][i], d['vt'][i],
                         [meta[key][n] for n in RN]))
    folds = np.array([r[0] for r in recs])
    RNFL = np.array([r[4] for r in recs], dtype=np.float64)
    for j in range(RNFL.shape[1]):                       # mean-impute missing RNFL sectors
        col = RNFL[:, j]; col[np.isnan(col)] = np.nanmean(col)
    rnfl_cov = float(np.mean([np.isfinite(r[4]).all() for r in recs]))

    def resid(v):
        m = ~np.isnan(v); r = np.full(v.shape, np.nan); r[m] = v[m] - v[m].mean(); return r

    eyes = []   # (pid, RR_valid, TR_valid, PR_valid)
    idx = np.arange(len(recs))
    for f in range(5):
        trm = folds != f
        mu, sd = RNFL[trm].mean(0), RNFL[trm].std(0) + 1e-8
        Xtr = (RNFL[trm] - mu) / sd                       # imputed RNFL, standardized on train
        # multi-output ridge: RNFL(4) -> true within-eye residual (52), nan->0 for the fit only
        Ytr = np.vstack([np.nan_to_num(resid(recs[i][3]), nan=0.0) for i in idx[trm]])
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        B = np.linalg.solve(A, Xtr.T @ Ytr)
        for i in idx[folds == f]:
            r = recs[i]
            RR52 = ((RNFL[i] - mu) / sd) @ B              # predict from IMPUTED RNFL (never raw r[4])
            vt = r[3]; m = ~np.isnan(vt)
            if not m.any():
                continue
            TR = resid(vt)[m]
            PR = (r[2] - r[2][m].mean())[m]              # fundus predicted residual on valid pts
            RRv = RR52[m] - RR52[m].mean()               # RNFL-predicted residual, re-centered
            eyes.append((int(r[1]), RRv, TR, PR))

    RRc = np.concatenate([e[1] for e in eyes])
    TRc = np.concatenate([e[2] for e in eyes])
    PRc = np.concatenate([e[3] for e in eyes])
    pcorr = D2.partial_corr(RRc, TRc, PRc)               # RNFL vs true | fundus residual
    rnfl_raw = float(np.corrcoef(RRc, TRc)[0, 1])
    fund_raw = float(np.corrcoef(PRc, TRc)[0, 1])
    dr2 = D2.incr_r2(RRc, TRc, PRc)                       # incremental R^2 of RNFL over fundus residual
    pids = np.array([e[0] for e in eyes]); uniq = np.unique(pids)
    by = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed); bs = np.empty(boot)
    for b in range(boot):
        rows = np.concatenate([by[p] for p in rng.choice(uniq, uniq.size, replace=True)])
        bs[b] = D2.partial_corr(np.concatenate([eyes[i][1] for i in rows]),
                                np.concatenate([eyes[i][2] for i in rows]),
                                np.concatenate([eyes[i][3] for i in rows]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    print("=" * 96)
    print(f"Task C1 — RNFL-sector SPATIAL value over fundus  ({len(eyes)} eyes, {RRc.size} pts, "
          f"{miss} unmatched, RNFL coverage {rnfl_cov:.0%})")
    print("-" * 96)
    print(f"  raw corr(fundus_resid, true_resid)            = {fund_raw:+.3f}")
    print(f"  raw corr(RNFL_resid,   true_resid)            = {rnfl_raw:+.3f}")
    print(f"  PARTIAL corr(RNFL_resid, true | fundus_resid) = {pcorr:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  incremental R^2 of RNFL over fundus residual  = {dr2:+.4f}")
    print("-" * 96)
    gate = ("BUILD metadata-fusion head — RNFL adds within-eye spatial signal (>=0.15)" if pcorr >= 0.15
            else "METADATA-SPATIAL DEAD — RNFL redundant with disc-crop fundus (<0.15); metadata fully closed")
    print(f"  GATE (>=0.15): {gate}")
    print("=" * 96)
    json.dump({'n_eyes': len(eyes), 'partial_corr': pcorr, 'ci': [float(lo), float(hi)],
               'rnfl_raw': rnfl_raw, 'fundus_raw': fund_raw, 'incr_r2': float(dr2),
               'rnfl_coverage': rnfl_cov},
              open(os.path.join(AUTO, 'diag_metadata_spatial.json'), 'w'), indent=2, default=float)


def main():
    if '--spatial' in sys.argv:
        return spatial_probe()
    meta = baseline_meta()
    # join metadata to the fundus OOF preds (p1disc), aligned via fold val jsons
    true_md, fund_md, folds, feats = [], [], [], []
    miss = 0
    for f in range(5):
        d = np.load(os.path.join(AUTO, 'oof_cache_notta', f'p1disc_f{f}.npz'))
        items = json.load(open(os.path.join(CV_DIR, f'fold{f}_val.json')))
        for i, it in enumerate(items):
            key = (int(it['PatientID']), 'OD' if str(it['Laterality']).startswith('OD') else 'OS')
            if key not in meta:
                miss += 1; continue
            vp, vt = d['vp'][i], d['vt'][i]
            m = ~np.isnan(vt)
            true_md.append(vt[m].mean()); fund_md.append(vp[m].mean()); folds.append(f)
            feats.append([meta[key][n] for n in COLS])
    true_md = np.array(true_md); fund_md = np.array(fund_md); folds = np.array(folds)
    F = np.array(feats)
    # mean-impute any nan features (column mean)
    for j in range(F.shape[1]):
        col = F[:, j]; col[np.isnan(col)] = np.nanmean(col)
    names = list(COLS)
    print("=" * 96)
    print(f"Diag B — metadata incremental value over fundus  ({len(true_md)} eyes joined, {miss} unmatched)")
    print("-" * 96)

    # 1. univariate corr with true MD
    print("univariate corr(field, true per-eye MD):")
    for j, n in enumerate(names):
        print(f"    {n:<8} r = {np.corrcoef(F[:, j], true_md)[0, 1]:+.3f}")

    def mae(p): return float(np.abs(p - true_md).mean())
    def corr(p): return float(np.corrcoef(p, true_md)[0, 1])

    # 2. metadata-alone OOF prediction of true MD
    p_meta = oof_ridge(F, true_md, folds)
    # 3. fundus alone, and fundus + metadata
    p_fund = oof_ridge(fund_md.reshape(-1, 1), true_md, folds)
    p_both = oof_ridge(np.column_stack([fund_md, F]), true_md, folds)
    # RNFL-only (the structural star)
    rnfl_idx = [names.index(k) for k in ('rnfl_g', 'rnfl_1', 'rnfl_2', 'rnfl_3', 'rnfl_4')]
    p_rnfl = oof_ridge(F[:, rnfl_idx], true_md, folds)
    p_fund_rnfl = oof_ridge(np.column_stack([fund_md, F[:, rnfl_idx]]), true_md, folds)

    print("\nOOF prediction of true per-eye MD (5-fold, ridge):")
    print(f"    {'predictor':<26}{'MAE':>8}{'corr':>8}")
    for lab, p in [('fundus_pred (raw)', fund_md), ('fundus_pred (recal)', p_fund),
                   ('metadata alone', p_meta), ('RNFL(5) alone', p_rnfl),
                   ('fundus + RNFL(5)', p_fund_rnfl), ('fundus + all metadata', p_both)]:
        print(f"    {lab:<26}{mae(p):>8.3f}{corr(p):>8.3f}")

    # 4. incremental partial-corr of metadata over fundus
    def partial(z_extra):
        Z = np.column_stack([np.ones_like(fund_md), fund_md])
        ry = true_md - Z @ np.linalg.lstsq(Z, true_md, rcond=None)[0]
        rx = z_extra - Z @ np.linalg.lstsq(Z, z_extra, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])
    print(f"\npartial-corr(RNFL_global, true MD | fundus_pred) = {partial(F[:, names.index('rnfl_g')]):+.3f}")
    print(f"partial-corr(metadata_pred, true MD | fundus_pred) = {partial(p_meta):+.3f}")
    d_mae = mae(p_fund) - mae(p_fund_rnfl)
    print("-" * 96)
    print(f"MD-error cut from adding OCT RNFL to the fundus: {mae(p_fund):.3f} → {mae(p_fund_rnfl):.3f} "
          f"= {d_mae:+.3f} dB on the SEVERITY channel")
    print("Interpretation: pooled VF MAE is ~severity-dominated, so an MD-error cut of this size is an "
          "upper-ish estimate of the pooled-MAE gain a metadata model could capture. Compare to the "
          "0.12 dB method-level MDE and the 4.113 → <4.0 gap (need ~−0.15+).")
    print("=" * 96)
    json.dump({'n': len(true_md), 'md_mae_fundus': mae(p_fund), 'md_mae_fundus_rnfl': mae(p_fund_rnfl),
               'md_mae_meta': mae(p_meta), 'partial_rnfl': partial(F[:, names.index('rnfl_g')]),
               'partial_meta': partial(p_meta)},
              open(os.path.join(AUTO, 'diag_metadata_value.json'), 'w'), indent=2, default=float)


if __name__ == "__main__":
    main()
