"""Truth-vs-pred scatter for a cached OOF tag, built ENTIRELY from the frozen npz cache — no
torch, no model reload, no training. Companion to composition_report.py: same input (
`oof_cache_notta/{tag}_f{fold}.npz`, keys vp/vt/tp/tt), same "no live inference" discipline.

Reimplements the two tiny numpy functions from eval_ckpt.py (pooled_stats / apply_calib) inline
so this script never imports eval_ckpt (which pulls in torch+RETFound at module level) or
decompose.py (same transitive torch import) — the severity/spatial decomposition numbers are
instead read from the already-computed `{tag}_cv.json` (written earlier by eval_oof_cached.py).

  python decoder/scatter_from_oof.py --tag p1disc_denoise
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")

# TDV-Net (Graefe's Arch 2026), fundus-only, 31k imgs — §3.1 constants, matched to composition_report.py
TDV_MILD, TDV_MODERATE, TDV_SEVERE, TDV_POOLED = 3.09, 5.66, 9.15, 3.91


def pooled_stats(preds, trues):
    P = np.concatenate([p[~np.isnan(t)] for p, t in zip(preds, trues)])
    Tt = np.concatenate([t[~np.isnan(t)] for t in trues])
    return P.mean(), P.std(), Tt.mean(), Tt.std()


def apply_calib(preds, mu_p, mu_t, b):
    return [np.clip(mu_t + b * (p - mu_p), 0, 35) for p in preds]


def solve_native_ratio_mix(target_severe, native_mild, native_moderate):
    """mild/moderate split (severe fixed at target_severe) that PRESERVES the native
    mild:moderate ratio observed in the pooled OOF — unlike the old TDV-Net-consistent solve
    (which forced mild/moderate to reproduce TDV's reported pooled 3.91, and at 13% severe drove
    moderate down to ~1.3%, leaving a visible gap in the moderate band). Realistic case-mix
    shifts should not distort the ratio between the two bands NOT being targeted."""
    remaining = 1 - target_severe
    denom = native_mild + native_moderate
    mild = remaining * (native_mild / denom)
    moderate = remaining - mild
    return {'mild': mild, 'moderate': moderate, 'severe': target_severe}


def make_composition_scatter(a, T, P, Pc, native_mae, mild_mae, moderate_mae, severe_mae, n_records):
    """Composition-adjusted scatter: bootstrap-RESAMPLE (with replacement) our pooled OOF points
    to a realistic 13%-severe case-mix that keeps the native mild:moderate ratio, then recompute
    MAE/slope on the resampled cloud and plot it directly (not an alpha-faded copy of the native
    cloud — that previous approach left the moderate band visually empty at low weight).
    """
    band_mask = {
        'severe': T < 15,
        'moderate': (T >= 15) & (T < 22),
        'mild': T >= 22,
    }
    native = {k: float(m.mean()) for k, m in band_mask.items()}
    our_band_mae = {'mild': mild_mae, 'moderate': moderate_mae, 'severe': severe_mae}
    target = solve_native_ratio_mix(a.target_severe, native['mild'], native['moderate'])

    # sanity check (before spending time resampling): target-mix-weighted sum of OUR per-band
    # MAEs must land close to the expected ~3.88 (0.643*2.904 + 0.227*4.062 + 0.13*8.425 ~ 3.884)
    check = sum(target[k] * our_band_mae[k] for k in target)
    print(f"native-ratio target mix @ severe={a.target_severe:.3f}: "
          f"mild {target['mild']:.4f} / moderate {target['moderate']:.4f} / severe {target['severe']:.4f}")
    print(f"  native mix: mild {native['mild']:.4f} / moderate {native['moderate']:.4f} / severe {native['severe']:.4f}")
    print(f"  band-sum check (expected pooled MAE under target mix): {check:.4f}")
    if not (3.85 <= check <= 3.92):
        print(f"DISCREPANCY: band-sum check {check:.4f} did not land in the expected 3.85-3.92 "
              f"band — stopping without shipping a figure.")
        sys.exit(1)

    # bootstrap resample to a fixed N with the target band proportions
    rng = np.random.default_rng(0)
    n_total = a.resample_n
    idx_by_band = {
        'severe': np.where(band_mask['severe'])[0],
        'moderate': np.where(band_mask['moderate'])[0],
        'mild': np.where(band_mask['mild'])[0],
    }
    counts = {k: int(round(target[k] * n_total)) for k in target}
    sel = np.concatenate([rng.choice(idx_by_band[k], size=counts[k], replace=True) for k in target])
    rng.shuffle(sel)
    Tr, Pr, Pcr = T[sel], P[sel], Pc[sel]

    r_mae = float(np.abs(Pr - Tr).mean())
    r_mae_cal = float(np.abs(Pcr - Tr).mean())
    r_sl, r_ic = np.polyfit(Tr, Pr, 1)
    r_sl_c, r_ic_c = np.polyfit(Tr, Pcr, 1)
    print(f"  resampled (n={sel.size}): MAE raw {r_mae:.4f}  calib {r_mae_cal:.4f}")
    print(f"  resampled slope raw {r_sl:.4f}  calib {r_sl_c:.4f}")
    if not (3.85 <= r_mae <= 3.92):
        print(f"DISCREPANCY: resampled MAE {r_mae:.4f} did not land in the expected 3.85-3.92 "
              f"band — stopping without shipping a figure.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.scatter(Tr, Pr, s=6, alpha=0.10, color='#1f4e79', edgecolors='none')
    lo, hi = 0, 36
    ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1.2, label='y = x (perfect)')
    xs = np.array([lo, hi])
    ax.plot(xs, r_sl * xs + r_ic, '-', color='#c00000', lw=2.2,
             label=f'raw fit: y = {r_sl:.2f}x + {r_ic:.1f}')
    ax.plot(xs, r_sl_c * xs + r_ic_c, '-', color='#2e7d32', lw=1.6,
             label=f'calib fit: y = {r_sl_c:.2f}x + {r_ic_c:.1f}')
    ax.axvspan(0, 15, color='orange', alpha=0.07)
    ax.text(7.5, 1.0, 'severe\n(<15 dB)', ha='center', va='bottom', fontsize=8, color='#a0522d')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
    ax.set_xlabel('True 24-2 sensitivity (dB)', fontsize=11)
    ax.set_ylabel('Predicted sensitivity (dB)', fontsize=11)
    ax.set_title(f'Fundus-only 24-2 VF prediction ({a.tag})\n'
                 f'Composition-adjusted to a realistic {100 * a.target_severe:.0f}%-severe case-mix',
                 fontsize=12, pad=12)

    box_lines = [
        f"{a.tag} — composition-adjusted (realistic mix, resampled n={sel.size})",
        f"  target mix   mild {100*target['mild']:.1f}% / mod {100*target['moderate']:.1f}% / severe {100*target['severe']:.1f}%",
        f"  (native mild:mod ratio preserved, not TDV-Net's)",
        f"  pooled MAE          {r_mae:.3f} dB   calib {r_mae_cal:.3f}",
        f"  slope raw           {r_sl:.3f}   calib {r_sl_c:.3f}",
        "─" * 30,
        f"native mix ({100*native['severe']:.1f}% severe) MAE = {native_mae:.3f}",
        f"TDV-Net pooled MAE = {TDV_POOLED:.2f} (their case-mix)",
        "─" * 30,
        "note: slope is composition-stable (model shrinkage,",
        "not case-mix) — ~0.55 raw across mixes; MAE moves,",
        "slope does not.",
    ]
    box = "\n".join(box_lines)
    ax.text(0.03, 0.97, box, transform=ax.transAxes, va='top', ha='left', fontsize=8.2,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#888', alpha=0.92))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.15)

    out = a.out or os.path.join(AUTO, f"{a.tag}_scatter_sev{int(round(a.target_severe * 100))}.png")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='p1disc_denoise')
    ap.add_argument('--ref-tag', default='p1disc', help='comparator drawn in the annotation box')
    ap.add_argument('--cache-dir', default=os.path.join(AUTO, 'oof_cache_notta'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--native-ratio', action='store_true',
                     help='bootstrap-resample points to a realistic case-mix at --target-severe '
                          '(preserving the native mild:moderate ratio) and plot the '
                          'composition-adjusted scatter instead of the native one')
    ap.add_argument('--target-severe', type=float, default=0.13,
                     help='target severe point-fraction for --native-ratio resampling (default 0.13)')
    ap.add_argument('--resample-n', type=int, default=30000,
                     help='total resampled point count for --native-ratio (default 30000)')
    a = ap.parse_args()

    vp_all, vt_all, vp_cal_all = [], [], []
    for f in range(5):
        d = np.load(os.path.join(a.cache_dir, f"{a.tag}_f{f}.npz"))
        vp, vt, tp, tt = list(d['vp']), list(d['vt']), list(d['tp']), list(d['tt'])
        mu_p, sig_p, mu_t, sig_t = pooled_stats(tp, tt)   # calib fit on this fold's TRAIN only
        b = sig_t / (sig_p + 1e-8)
        vp_cal = apply_calib(vp, mu_p, mu_t, b)
        vp_all += vp; vt_all += vt; vp_cal_all += vp_cal

    P = np.concatenate([p[~np.isnan(t)] for p, t in zip(vp_all, vt_all)])
    T = np.concatenate([t[~np.isnan(t)] for t in vt_all])
    Pc = np.concatenate([p[~np.isnan(t)] for p, t in zip(vp_cal_all, vt_all)])

    mae = float(np.abs(P - T).mean())
    mae_cal = float(np.abs(Pc - T).mean())
    rmse = float(np.sqrt(((P - T) ** 2).mean()))
    sl, ic = np.polyfit(T, P, 1)
    sl_c, ic_c = np.polyfit(T, Pc, 1)
    corr = float(np.corrcoef(P, T)[0, 1])

    def band_mae(lo, hi, arr):
        m = (T >= lo) & (T < hi)
        return float(np.abs(arr[m] - T[m]).mean()) if m.any() else float('nan')
    severe_mae = band_mae(-1e9, 15, P)
    moderate_mae = band_mae(15, 22, P)
    mild_mae = band_mae(22, 1e9, P)

    # severity/spatial decomposition + reference-tag numbers: read from the precomputed *_cv.json
    # (avoids importing decompose.py/eval_ckpt.py, which pull in torch at module level)
    sev = {}
    cvpath = os.path.join(AUTO, f"{a.tag}_cv.json")
    if os.path.exists(cvpath):
        sev = json.load(open(cvpath)).get('severity', {})
    ref = {}
    refpath = os.path.join(AUTO, f"{a.ref_tag}_cv.json")
    if os.path.exists(refpath):
        ref = json.load(open(refpath))

    print(f"{a.tag}: pooled n={T.size} pts / {len(vt_all)} recs | MAE raw {mae:.3f} calib {mae_cal:.3f} | "
          f"RMSE {rmse:.3f} | slope raw {sl:.3f} calib {sl_c:.3f} | corr {corr:.3f} | "
          f"mild/mod/severe {mild_mae:.3f}/{moderate_mae:.3f}/{severe_mae:.3f}")
    if sev:
        print(f"  severity decomp (from {a.tag}_cv.json): sev_corr {sev.get('sev_corr', float('nan')):.3f} "
              f"res_corr {sev.get('res_corr', float('nan')):.3f}")

    if a.native_ratio:
        make_composition_scatter(a, T, P, Pc, mae, mild_mae, moderate_mae, severe_mae, len(vt_all))
        return

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.scatter(T, P, s=6, alpha=0.10, color='#1f4e79', edgecolors='none')
    lo, hi = 0, 36
    ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1.2, label='y = x (perfect)')
    xs = np.array([lo, hi])
    ax.plot(xs, sl * xs + ic, '-', color='#c00000', lw=2.2, label=f'raw fit: y = {sl:.2f}x + {ic:.1f}')
    ax.plot(xs, sl_c * xs + ic_c, '-', color='#2e7d32', lw=1.6, label=f'calib fit: y = {sl_c:.2f}x + {ic_c:.1f}')
    ax.axvspan(0, 15, color='orange', alpha=0.07)
    ax.text(7.5, 1.0, 'severe\n(<15 dB)', ha='center', va='bottom', fontsize=8, color='#a0522d')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
    ax.set_xlabel('True 24-2 sensitivity (dB)', fontsize=11)
    ax.set_ylabel('Predicted sensitivity (dB)', fontsize=11)
    ax.set_title(f'Fundus-only 24-2 VF prediction ({a.tag})\nhonest per-patient 5-fold CV, '
                 f'{len(vt_all)} records, {T.size} points', fontsize=12, pad=12)

    box_lines = [
        f"{a.tag} (this model)",
        f"  pointwise MAE raw   {mae:.3f} dB   calib {mae_cal:.3f}",
        f"  RMSE                {rmse:.3f} dB",
        f"  mild/mod/severe     {mild_mae:.2f} / {moderate_mae:.2f} / {severe_mae:.2f}",
        f"  pointwise r  {corr:.3f}   slope raw {sl:.3f}  calib {sl_c:.3f}",
    ]
    if sev:
        box_lines.append(f"  sev_corr {sev.get('sev_corr', float('nan')):.3f}   "
                          f"res_corr {sev.get('res_corr', float('nan')):.3f}")
    if ref:
        rraw = ref.get('raw', {})
        box_lines.append("─" * 26)
        box_lines.append(f"{a.ref_tag} (champion comparator)")
        box_lines.append(f"  pooled MAE {rraw.get('mae', float('nan')):.3f}  "
                          f"slope {rraw.get('slope', float('nan')):.3f}")
    box_lines.append("─" * 26)
    box_lines.append("TDV-Net, Graefe's 2026 (fundus-only,\n31k imgs, total-deviation target)")
    box_lines.append("  pointwise MAE   3.91 dB")
    box_lines.append("  mild/mod/severe 3.09 / 5.66 / 9.15")
    box = "\n".join(box_lines)
    ax.text(0.03, 0.97, box, transform=ax.transAxes, va='top', ha='left', fontsize=8.2,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#888', alpha=0.92))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.15)

    out = a.out or os.path.join(AUTO, f"{a.tag}_scatter.png")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
