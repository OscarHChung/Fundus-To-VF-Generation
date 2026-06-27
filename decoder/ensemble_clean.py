"""Seed/checkpoint ensemble on the clean per-patient val — averages per-eye predictions across
models, then reports raw + variance-match calibrated metrics. Averaging decorrelated models cuts
the per-eye SEVERITY (mean) estimate variance — the dominant error per the decomposition.

  python decoder/ensemble_clean.py <ckpt1> <ckpt2> ... --train-json <fold_train> --val-json <fold_val>
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ckpt as E
import diagnostics as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True)
    a = ap.parse_args()

    val_sets, train_sets, vt_ref = [], [], None
    for c in a.ckpts:
        model = E.load_model(c)
        vp, vt = E.per_eye_preds(model, a.val_json)
        tp, tt = E.per_eye_preds(model, a.train_json)
        val_sets.append(vp); train_sets.append((tp, tt))
        vt_ref = vt
        print(f"  {os.path.basename(c):<26} {D.fmt(D.pooled_metrics(vp, vt))}")

    # ensemble = mean of per-eye predictions across models (aligned by eye order)
    ens_v = [np.mean([val_sets[m][i] for m in range(len(a.ckpts))], axis=0)
             for i in range(len(vt_ref))]
    raw = D.pooled_metrics(ens_v, vt_ref)
    print(f"\nENSEMBLE raw ({len(a.ckpts)} models) {D.fmt(raw)}")

    # calibrate: fit on ensemble-of-train predictions
    tt_ref = train_sets[0][1]
    ens_t = [np.mean([train_sets[m][0][i] for m in range(len(a.ckpts))], axis=0)
             for i in range(len(tt_ref))]
    mu_p, sig_p, mu_t, sig_t = E.pooled_stats(ens_t, tt_ref)
    b = sig_t / (sig_p + 1e-8)
    for s in [0.0, 0.5, 0.75, 1.0]:
        bb = 1.0 + s * (b - 1.0)
        m = D.pooled_metrics(E.apply_calib(ens_v, mu_p, mu_t, bb), vt_ref)
        print(f"  calib s={s:.2f} (b={bb:.2f}) {D.fmt(m)}")
    print(f"\nrefs: oracle 4.09 ; control 4.55 ; single global head 4.46 ; sub-3.74 target.")


if __name__ == "__main__":
    main()
