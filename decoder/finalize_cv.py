"""Finalize a 5-fold CV run: pool out-of-fold predictions, apply RAW / single-b / two-level
calibration (each fit per-fold on TRAIN only — no val tuning), report the honest champion metrics
across all 263 eyes (and all 47 severe eyes), save a champion record + a scatterplot.

Two-level calibration (the Session-3 idea): de-shrink the eye-mean (severity) and the within-eye
residual by SEPARATE factors fit on train. 'sev-only' (b_sev=full, b_res=1) targets MIN MAE
(fixes severity shrinkage without residual overshoot); 'full' (both full) targets the steepest
faithful scatterplot line.

  python decoder/finalize_cv.py --tag s3_ghfinal [--recipe "<exact training.py args>"]
"""
import os, sys, json, argparse, shutil, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ckpt as E
import diagnostics as D
import twolevel_calib as TL

CUR = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.join(CUR, "results", "cv")
AUTO = os.path.join(CUR, "results", "auto")
CHAMP = os.path.join(CUR, "results", "champion_ppat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='s3_ghfinal')
    ap.add_argument('--folds', default="0,1,2,3,4")
    ap.add_argument('--recipe', default='--global-head (GH sector_only, value reweight, lr8e-4, do0.2, wd5e-3)')
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(',')]

    raw, oneb, sevonly, full = [], [], [], []
    trues = []
    for f in folds:
        ck = os.path.join(AUTO, f"{a.tag}_f{f}_best.pth")
        model = E.load_model(ck)
        vp, vt = E.per_eye_preds(model, os.path.join(CV_DIR, f"fold{f}_val.json"))
        tp, tt = E.per_eye_preds(model, os.path.join(CV_DIR, f"fold{f}_train.json"))
        mu_p, sig_p, mu_t, sig_t = E.pooled_stats(tp, tt)
        st = TL.fit_stats(tp, tt)
        bsev, bres = st['sig_tm'] / (st['sig_pm'] + 1e-8), st['sig_tr'] / (st['sig_pr'] + 1e-8)
        raw += vp
        oneb += E.apply_calib(vp, mu_p, mu_t, sig_t / (sig_p + 1e-8))
        sevonly += TL.apply_two_level(vp, st, bsev, 1.0)     # de-shrink severity only → min MAE
        full += TL.apply_two_level(vp, st, bsev, bres)        # full two-level → steep line
        trues += vt
        print(f"  fold {f}: raw {D.fmt(D.pooled_metrics(vp, vt))}")

    res = {}
    print("\n" + "=" * 96)
    for name, preds in [('RAW', raw), ('1-level calib', oneb),
                        ('2-level sev-only (min-MAE)', sevonly), ('2-level full (best-line)', full)]:
        m = D.pooled_metrics(preds, trues); res[name] = m
        print(f"OOF {name:<26} {D.fmt(m)}")
    print("=" * 96)
    print(f"({len(trues)} eyes OOF) refs: oracle 4.09 ; leaky champion 4.116 ; 3.74 (leaky/shrunk).")

    os.makedirs(CHAMP, exist_ok=True)
    # representative deployable model = fold 0; record recipe + all metrics
    rep = os.path.join(AUTO, f"{a.tag}_f0_best.pth")
    if os.path.exists(rep):
        shutil.copy(rep, os.path.join(CHAMP, "model_fold0.pth"))
    rec = {'protocol': 'per-patient 5-fold CV (no fellow-eye leakage), OOF pooled',
           'recipe': a.recipe, 'n_eyes': len(trues),
           'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
           'fold_checkpoints': [f"results/auto/{a.tag}_f{f}_best.pth" for f in folds],
           'timestamp': datetime.datetime.now().isoformat(timespec='seconds')}
    json.dump(rec, open(os.path.join(CHAMP, "champion_ppat.json"), 'w'), indent=2)
    print(f"\n🏆 Saved honest champion → {CHAMP}/champion_ppat.json (+ model_fold0.pth)")


if __name__ == "__main__":
    main()
