"""5-fold per-patient CV confirmation harness — the honest gold-standard scorer.

Trains a recipe on each of the 5 per-patient folds (subprocess → clean MPS/module state per
fold), then evaluates each fold's best checkpoint on its OWN held-out val. Pools the
out-of-fold predictions over ALL 263 eyes (and all 47 severe eyes) for a leak-free, low-variance
estimate of overall MAE + the severe metrics, RAW and variance-match CALIBRATED (calibration fit
per-fold on that fold's train only). This is what decides a Session-3 champion.

  python decoder/run_cv.py --tag lever1a --epochs 60 -- --weighting garway_heath \
      --sector-combine sector_only --reweight value --lr 8e-4 --dropout 0.2 \
      --weight-decay 0.005 --dispersion-weight 0.1 --entropy-weight 0 --label-noise 0.1
(everything after the bare `--` is passed verbatim to training.py)
"""
import os, sys, json, argparse, subprocess
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CV_DIR  = os.path.join(CURRENT_DIR, "results", "cv")
AUTO    = os.path.join(CURRENT_DIR, "results", "auto")
PY = sys.executable


def train_fold(tag, fold, epochs, recipe, cv_dir=CV_DIR):
    out_tag = f"{tag}_f{fold}"
    ckpt = os.path.join(AUTO, f"{out_tag}_best.pth")
    log  = os.path.join(AUTO, f"{out_tag}.log")
    cmd = [PY, os.path.join(CURRENT_DIR, "training.py"),
           "--train-json", os.path.join(cv_dir, f"fold{fold}_train.json"),
           "--val-json",   os.path.join(cv_dir, f"fold{fold}_val.json"),
           "--out-tag", out_tag, "--no-champion", "--epochs", str(epochs)] + recipe
    print(f"\n=== fold {fold}: training → {out_tag} (log {os.path.basename(log)}) ===")
    with open(log, 'w') as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--folds', default="0,1,2,3,4")
    ap.add_argument('--cv-dir', default=CV_DIR)
    ap.add_argument('rest', nargs=argparse.REMAINDER)  # after `--`
    a = ap.parse_args()
    recipe = a.rest[1:] if a.rest and a.rest[0] == '--' else a.rest
    folds = [int(x) for x in a.folds.split(',')]
    cv_dir = a.cv_dir

    import eval_ckpt as E   # lazy (loads RETFound)
    vp_all, vt_all, vp_cal = [], [], []
    for f in folds:
        ckpt = train_fold(a.tag, f, a.epochs, recipe, cv_dir)
        model = E.load_model(ckpt)
        vp, vt = E.per_eye_preds(model, os.path.join(cv_dir, f"fold{f}_val.json"))
        tp, tt = E.per_eye_preds(model, os.path.join(cv_dir, f"fold{f}_train.json"))
        mu_p, sig_p, mu_t, sig_t = E.pooled_stats(tp, tt)
        b = sig_t / (sig_p + 1e-8)                       # full variance match, fit on fold-train
        vp_all += vp; vt_all += vt
        vp_cal += E.apply_calib(vp, mu_p, mu_t, b)
        import diagnostics as D
        print(f"  fold {f}: {D.fmt(D.pooled_metrics(vp, vt))}")

    import diagnostics as D
    raw = D.pooled_metrics(vp_all, vt_all)
    cal = D.pooled_metrics(vp_cal, vt_all)
    print("\n" + "=" * 90)
    print(f"5-FOLD OOF ({len(vt_all)} recs)  RAW    {D.fmt(raw)}")
    D.stratified_report(vp_all, vt_all)
    print(f"5-FOLD OOF ({len(vt_all)} recs)  CALIB  {D.fmt(cal)}")
    D.stratified_report(vp_cal, vt_all)
    print("=" * 90)
    print("refs: oracle MAE 4.09 ; honest baseline ~4.46 ; sub-3.9 firm / sub-3.74 stretch.")
    out = os.path.join(AUTO, f"{a.tag}_cv.json")
    json.dump({'tag': a.tag, 'recipe': recipe, 'raw': raw, 'calib': cal,
               'n_eyes': len(vt_all)}, open(out, 'w'), indent=2, default=float)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
