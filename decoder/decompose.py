"""Decompose a model's val error into SEVERITY (per-eye mean) vs WITHIN-EYE (spatial residual).

Oracle (perfect severity, 0 spatial) = MAE 4.09 on clean data; models sit ~0.4 above it. This
tells us whether the gap is severity-prediction error (→ better mean head / ensemble) or missing
spatial signal (→ encoder/features). Prints per-eye-mean MAE+corr and within-eye residual MAE+corr.

  python decoder/decompose.py <ckpt...> --val-json <fold_val>
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ckpt as E


def decompose(vp, vt):
    sp, st, rp, rt = [], [], [], []
    for p, t in zip(vp, vt):
        m = ~np.isnan(t); pp, tt = p[m], t[m]
        pm, tm = pp.mean(), tt.mean()
        sp.append(pm); st.append(tm)
        rp.extend((pp - pm).tolist()); rt.extend((tt - tm).tolist())
    sp, st, rp, rt = map(np.array, (sp, st, rp, rt))
    sev_mae = np.abs(sp - st).mean()
    sev_corr = np.corrcoef(sp, st)[0, 1]
    sev_shrink = sp.std() / (st.std() + 1e-8)
    res_mae = np.abs(rp - rt).mean()
    res_corr = np.corrcoef(rp, rt)[0, 1]
    res_shrink = rp.std() / (rt.std() + 1e-8)
    return dict(sev_mae=sev_mae, sev_corr=sev_corr, sev_shrink=sev_shrink,
                res_mae=res_mae, res_corr=res_corr, res_shrink=res_shrink)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--val-json', required=True)
    a = ap.parse_args()
    print(f"val={os.path.basename(a.val_json)}  (oracle: sev_mae=0 → overall 4.09)")
    print(f"{'ckpt':<26}{'sevMAE':>7}{'sevCorr':>8}{'sevShr':>7} | "
          f"{'resMAE':>7}{'resCorr':>8}{'resShr':>7}")
    for c in a.ckpts:
        model = E.load_model(c)
        vp, vt = E.per_eye_preds(model, a.val_json)
        d = decompose(vp, vt)
        print(f"{os.path.basename(c):<26}{d['sev_mae']:>7.2f}{d['sev_corr']:>8.3f}"
              f"{d['sev_shrink']:>7.2f} | {d['res_mae']:>7.2f}{d['res_corr']:>8.3f}{d['res_shrink']:>7.2f}")
    print("\nsev = per-eye MEAN (severity) ; res = WITHIN-eye residual (spatial). "
          "Big sevMAE ⇒ severity is the bottleneck (mean head/ensemble); low resCorr ⇒ spatial.")


if __name__ == "__main__":
    main()
