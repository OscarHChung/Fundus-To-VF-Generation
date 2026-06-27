"""Per-point truth-vs-prediction scatterplot with line of best fit, for any checkpoint on the
clean per-patient val (raw or variance-match calibrated). The deliverable for "a strong line of
best fit". Reuses eval_ckpt for predictions + calibration.

  python decoder/scatter_clean.py <ckpt.pth> --train-json <fold_train> --val-json <fold_val> \
      [--calib]  -o decoder/results/auto/<tag>_scatter.png
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ckpt as E
import diagnostics as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--calib', action='store_true', help="apply full variance-match calibration")
    ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()

    model = E.load_model(a.ckpt)
    vp, vt = E.per_eye_preds(model, a.val_json)
    if a.calib:
        tp, tt = E.per_eye_preds(model, a.train_json)
        mu_p, sig_p, mu_t, sig_t = E.pooled_stats(tp, tt)
        vp = E.apply_calib(vp, mu_p, mu_t, sig_t / (sig_p + 1e-8))
    m = D.pooled_metrics(vp, vt)
    P = np.concatenate([p[~np.isnan(t)] for p, t in zip(vp, vt)])
    T = np.concatenate([t[~np.isnan(t)] for t in vt])

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(T, P, s=6, alpha=0.18, color='#1f4e79', edgecolors='none')
    lo, hi = -2, 37
    ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1, label='y = x (ideal)')
    sl, ic = np.polyfit(T, P, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, sl * xs + ic, '-', color='#c00000', lw=2,
            label=f'fit: y={sl:.2f}x+{ic:.1f}')
    # severe band marker
    ax.axvspan(0, 10, color='orange', alpha=0.06)
    ax.text(5, hi - 2, 'severe\n(0–10 dB)', ha='center', va='top', fontsize=8, color='#a0522d')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel('True sensitivity (dB)'); ax.set_ylabel('Predicted sensitivity (dB)')
    tag = "calibrated" if a.calib else "raw"
    ax.set_title(f'Per-point VF prediction ({tag})\n'
                 f'MAE {m["mae"]:.2f} | slope {m["slope"]:.3f} | corr {m["corr"]:.3f} | '
                 f'eyeCorr {m["eyecorr"]:.3f} | floor {m["floor"]:.2f}', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.15)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)),
                                os.path.basename(a.ckpt).replace('.pth', f'_scatter_{tag}.png'))
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"{tag}: {D.fmt(m)}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
