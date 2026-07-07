"""Pooled 5-fold OOF scorer for cached-trainer checkpoints (the honest gold-standard eval).

run_cv.py trains+scores in one process via the LIVE-encoder path (OOMs for LoRA on this box).
This script instead scores PRE-TRAINED per-fold checkpoints (produced separately, one process at
a time, by train_lora_cached.py). For each fold it loads {tag}_f{fold}_best.pth, predicts the fold's
held-out val + its train (for the per-fold variance-match calibration), then pools the out-of-fold
predictions over all folds → pooled RAW + CALIB metrics, the severity-stratified table, AND the
severity/spatial decomposition (sev_corr = the M1 lever). Writes {tag}_cv.json.

Resumable: each fold's (val/train) predictions are cached to {cache_dir}/{tag}_f{fold}.npz, so a
re-run (or a crash mid-sweep) skips the ~1.2 GB encoder reload + forward for folds already scored.

  python decoder/eval_oof_cached.py --tag m1m3            # score m1m3_f0..f4, write m1m3_cv.json
  python decoder/eval_oof_cached.py --tag m1m3 --folds 0  # just fold 0 (fills its npz)
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")


def _stack(lst):
    """List of (52,) arrays (nan at masked) → (n,52) float64 array."""
    return np.vstack([np.asarray(a, dtype=np.float64) for a in lst])


def fold_preds(tag, fold, cv_dir, cache_dir, no_tta):
    """Return (vp, vt, tp, tt) for a fold, from the npz cache or a fresh encoder eval."""
    npz = os.path.join(cache_dir, f"{tag}_f{fold}.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        vp = list(d['vp']); vt = list(d['vt']); tp = list(d['tp']); tt = list(d['tt'])
        print(f"  fold {fold}: loaded cached preds ({len(vp)} val / {len(tp)} train) ← {os.path.basename(npz)}",
              flush=True)
        return vp, vt, tp, tt
    import eval_ckpt as E  # lazy: loads RETFound
    ckpt = os.path.join(AUTO, f"{tag}_f{fold}_best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    model = E.load_model(ckpt)
    vp, vt = E.per_eye_preds(model, os.path.join(cv_dir, f"fold{fold}_val.json"),   use_tta=not no_tta)
    tp, tt = E.per_eye_preds(model, os.path.join(cv_dir, f"fold{fold}_train.json"), use_tta=not no_tta)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(npz, vp=_stack(vp), vt=_stack(vt), tp=_stack(tp), tt=_stack(tt))
    print(f"  fold {fold}: scored + cached → {os.path.basename(npz)}", flush=True)
    return list(_stack(vp)), list(_stack(vt)), list(_stack(tp)), list(_stack(tt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--folds', default="0,1,2,3,4")
    ap.add_argument('--cv-dir', default=CV_DIR)
    ap.add_argument('--cache-dir', default=os.path.join(AUTO, "oof_cache"))
    ap.add_argument('--no-tta', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(',')]

    import eval_ckpt as E
    import diagnostics as D
    import decompose as DEC

    vp_all, vt_all, vp_cal = [], [], []
    per_fold = {}
    for f in folds:
        vp, vt, tp, tt = fold_preds(a.tag, f, a.cv_dir, a.cache_dir, a.no_tta)
        mu_p, sig_p, mu_t, sig_t = E.pooled_stats(tp, tt)     # calibration fit on fold TRAIN only
        b = sig_t / (sig_p + 1e-8)
        vp_all += vp; vt_all += vt
        vp_cal += E.apply_calib(vp, mu_p, mu_t, b)
        fm = D.pooled_metrics(vp, vt)
        per_fold[f] = {'mae': fm['mae'], 'slope': fm['slope'], 'corr': fm['corr'],
                       'eyecorr': fm['eyecorr'], 'b': float(b)}
        print(f"  fold {f}: {D.fmt(fm)}", flush=True)

    if len(folds) < 5:
        print(f"\n(scored {len(folds)} of 5 folds — pooled numbers below are PARTIAL)", flush=True)

    raw = D.pooled_metrics(vp_all, vt_all)
    cal = D.pooled_metrics(vp_cal, vt_all)
    sev = DEC.decompose(vp_all, vt_all)   # sev_corr = the M1 between-eye lever; res_corr = spatial
    print("\n" + "=" * 92)
    print(f"{a.tag}  5-FOLD OOF ({len(vt_all)} recs)  RAW    {D.fmt(raw)}")
    D.stratified_report(vp_all, vt_all)
    print(f"{a.tag}  5-FOLD OOF ({len(vt_all)} recs)  CALIB  {D.fmt(cal)}")
    D.stratified_report(vp_cal, vt_all)
    print(f"\n  SEVERITY decomp (pooled OOF): sev_corr {sev['sev_corr']:.3f}  sev_shrink "
          f"{sev['sev_shrink']:.2f}  sev_mae {sev['sev_mae']:.2f} | res_corr {sev['res_corr']:.3f}  "
          f"res_shrink {sev['res_shrink']:.2f}")
    print("=" * 92)
    print("refs: long_global 4.290/0.473/r0.657/severe~7.49 ; loraC 4.220/0.485/r0.657/severe7.694")
    print("TARGET: MAE<4.00 AND slope≥0.60 (raw), severe not worse, r≥0.72 ideally.")

    out = a.out or os.path.join(AUTO, f"{a.tag}_cv.json")
    json.dump({'tag': a.tag, 'raw': raw, 'calib': cal, 'severity': sev,
               'per_fold': per_fold, 'n_eyes': len(vt_all),
               'partial': len(folds) < 5}, open(out, 'w'), indent=2, default=float)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
