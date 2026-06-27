"""Two-level (severity / residual) post-hoc calibration — decouples the eye-mean stretch from the
within-eye residual stretch. The decomposition showed sevShr≈0.87 (eye-means mildly shrunk) but
resShr≈0.36 (residual heavily shrunk): one global `b` (the old calibration) can't fix both. Here we
fit b_sev on train eye-means and sweep b_res on the residual, so we can (a) MINIMIZE MAE by optimally
de-shrinking severity, and (b) independently pick the residual spread for a steep scatterplot slope.

  python decoder/twolevel_calib.py <ckpt...> --train-json <fold_train> --val-json <fold_val>
Works on a single model or an ensemble (averages per-eye preds across the given checkpoints).
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ckpt as E
import diagnostics as D


def split_mean_resid(preds, trues):
    """Per eye: (pred_mean, valid pred residual, valid true residual, valid true)."""
    out = []
    for p, t in zip(preds, trues):
        m = ~np.isnan(t)
        pm = p.mean()
        out.append((pm, p, t))   # keep full p (52) + t (52, nan-masked); center at apply time
    return out


def ensemble_preds(ckpts, json_path):
    sets = []
    vt = None
    for c in ckpts:
        model = E.load_model(c)
        vp, vt = E.per_eye_preds(model, json_path)
        sets.append(vp)
    ens = [np.mean([sets[m][i] for m in range(len(ckpts))], axis=0) for i in range(len(vt))]
    return ens, vt


def fit_stats(preds, trues):
    pmeans = np.array([p.mean() for p in preds])
    tmeans = np.array([np.nanmean(t) for t in trues])
    pres, tres = [], []
    for p, t in zip(preds, trues):
        m = ~np.isnan(t)
        pres.extend((p - p.mean())[m].tolist())
        tres.extend((t - np.nanmean(t))[m].tolist())
    pres, tres = np.array(pres), np.array(tres)
    return dict(mu_pm=pmeans.mean(), sig_pm=pmeans.std(), mu_tm=tmeans.mean(), sig_tm=tmeans.std(),
                sig_pr=pres.std(), sig_tr=tres.std())


def apply_two_level(preds, st, b_sev, b_res):
    out = []
    for p in preds:
        pm = p.mean()
        m2 = st['mu_tm'] + b_sev * (pm - st['mu_pm'])
        r2 = b_res * (p - pm)
        out.append(np.clip(m2 + r2, 0, 35))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True)
    a = ap.parse_args()

    vp, vt = ensemble_preds(a.ckpts, a.val_json)
    tp, tt = ensemble_preds(a.ckpts, a.train_json)
    st = fit_stats(tp, tt)
    b_sev_full = st['sig_tm'] / (st['sig_pm'] + 1e-8)
    b_res_full = st['sig_tr'] / (st['sig_pr'] + 1e-8)
    print(f"{len(a.ckpts)} model(s) | RAW {D.fmt(D.pooled_metrics(vp, vt))}")
    print(f"full b_sev={b_sev_full:.2f} (severity), b_res={b_res_full:.2f} (residual)\n")
    print(f"{'b_sev':>6}{'b_res':>6} | metrics")
    rows = []
    for bs in sorted({1.0, round(0.5*(1+b_sev_full),2), round(b_sev_full,2)}):
        for br in [1.0, 1.5, 2.0, round(b_res_full, 2)]:
            m = D.pooled_metrics(apply_two_level(vp, st, bs, br), vt)
            m['b_sev'], m['b_res'] = bs, br
            rows.append(m)
            print(f"{bs:>6.2f}{br:>6.2f} | {D.fmt(m)}")
    best_mae = min(rows, key=lambda r: r['mae'])
    # best scatterplot: steepest slope with MAE within 0.3 of the best-MAE point
    feas = [r for r in rows if r['mae'] < best_mae['mae'] + 0.3]
    best_slope = max(feas, key=lambda r: r['slope'])
    print(f"\n▸ MIN-MAE:  b_sev={best_mae['b_sev']:.2f} b_res={best_mae['b_res']:.2f} → "
          f"MAE {best_mae['mae']:.3f}, slope {best_mae['slope']:.3f}, floor {best_mae['floor']:.2f}")
    print(f"▸ BEST-LINE: b_sev={best_slope['b_sev']:.2f} b_res={best_slope['b_res']:.2f} → "
          f"MAE {best_slope['mae']:.3f}, slope {best_slope['slope']:.3f}, floor {best_slope['floor']:.2f}")


if __name__ == "__main__":
    main()
