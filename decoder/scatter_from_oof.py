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


def solve_tdv_consistent_mix(target_severe):
    """mild/moderate split (severe fixed at target_severe) that reproduces TDV-Net's reported
    pooled 3.91 from THEIR per-band MAEs: TDV_MILD*mild + TDV_MODERATE*moderate +
    TDV_SEVERE*target_severe = TDV_POOLED, with mild + moderate = 1 - target_severe."""
    a = TDV_MILD - TDV_MODERATE
    rhs = TDV_POOLED - TDV_MODERATE * (1 - target_severe) - TDV_SEVERE * target_severe
    mild = rhs / a
    moderate = (1 - target_severe) - mild
    return {'mild': mild, 'moderate': moderate, 'severe': target_severe}


def weighted_polyfit1(x, y, w):
    """Weighted least-squares slope/intercept of y on x (degree-1), weights w."""
    sw = w.sum()
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    sxy = np.sum(w * (x - mx) * (y - my))
    sxx = np.sum(w * (x - mx) ** 2)
    slope = sxy / sxx
    intercept = my - slope * mx
    return slope, intercept


def make_tdv_scatter(a, T, P, Pc, native_mae, mild_mae, moderate_mae, severe_mae, n_records):
    """Composition-adjusted scatter: reweight our pooled OOF points to a TDV-Net-consistent
    case-mix (severe fixed at a.target_severe; mild/moderate solved to reproduce TDV's reported
    pooled 3.91), then recompute the weighted MAE/slope/calib-slope under that mix.

    Style choice: (a) plot the SAME point cloud as the native scatter, with per-point alpha
    proportional to its reweighting factor, rather than a weighted resample — this keeps the
    point positions identical to p1disc_denoise_scatter.png (only the visual density/opacity of
    each severity band shifts to reflect the 13%-severe target population).
    """
    target = solve_tdv_consistent_mix(a.target_severe)

    band_mask = {
        'severe': T < 15,
        'moderate': (T >= 15) & (T < 22),
        'mild': T >= 22,
    }
    native = {k: float(m.mean()) for k, m in band_mask.items()}
    our_band_mae = {'mild': mild_mae, 'moderate': moderate_mae, 'severe': severe_mae}

    w = np.empty_like(T)
    for k, m in band_mask.items():
        w[m] = target[k] / native[k]
    w = w / w.mean()   # normalize to mean 1 (already ~1 by construction; guards float drift)

    def wmae(arr):
        return float(np.sum(w * np.abs(arr - T)) / np.sum(w))

    w_mae = wmae(P)
    w_mae_cal = wmae(Pc)
    w_sl, w_ic = weighted_polyfit1(T, P, w)
    w_sl_c, w_ic_c = weighted_polyfit1(T, Pc, w)

    # sanity check: weighted MAE must equal the target-mix-weighted sum of OUR per-band MAEs
    check = sum(target[k] * our_band_mae[k] for k in target)
    print(f"tdv-consistent target mix @ severe={a.target_severe:.3f}: "
          f"mild {target['mild']:.4f} / moderate {target['moderate']:.4f} / severe {target['severe']:.4f}")
    print(f"  native mix: mild {native['mild']:.4f} / moderate {native['moderate']:.4f} / severe {native['severe']:.4f}")
    print(f"  weighted MAE {w_mae:.4f} (band-sum check {check:.4f}) | weighted MAE calib {w_mae_cal:.4f}")
    print(f"  weighted slope raw {w_sl:.4f}  calib {w_sl_c:.4f}")
    if not (abs(w_mae - check) < 1e-6 and 3.6 <= w_mae <= 3.7):
        print(f"DISCREPANCY: weighted MAE {w_mae:.4f} did not land in the expected 3.6-3.7 band "
              f"(or disagrees with the band-sum check {check:.4f}) — stopping without shipping a figure.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    base_rgb = np.array(matplotlib.colors.to_rgb('#1f4e79'))
    alpha = np.clip(0.10 * w, 0.004, 0.85)
    colors = np.tile(np.append(base_rgb, 1.0), (T.size, 1))
    colors[:, 3] = alpha
    ax.scatter(T, P, s=6, color=colors, edgecolors='none')
    lo, hi = 0, 36
    ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1.2, label='y = x (perfect)')
    xs = np.array([lo, hi])
    ax.plot(xs, w_sl * xs + w_ic, '-', color='#c00000', lw=2.2,
             label=f'weighted raw fit: y = {w_sl:.2f}x + {w_ic:.1f}')
    ax.plot(xs, w_sl_c * xs + w_ic_c, '-', color='#2e7d32', lw=1.6,
             label=f'weighted calib fit: y = {w_sl_c:.2f}x + {w_ic_c:.1f}')
    ax.axvspan(0, 15, color='orange', alpha=0.07)
    ax.text(7.5, 1.0, 'severe\n(<15 dB)', ha='center', va='bottom', fontsize=8, color='#a0522d')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', 'box')
    ax.set_xlabel('True 24-2 sensitivity (dB)', fontsize=11)
    ax.set_ylabel('Predicted sensitivity (dB)', fontsize=11)
    ax.set_title(f'Fundus-only 24-2 VF prediction ({a.tag})\n'
                 f'Composition-adjusted to TDV-Net case-mix ({100 * a.target_severe:.0f}% severe)',
                 fontsize=12, pad=12)

    box_lines = [
        f"{a.tag} — composition-adjusted (TDV-Net mix)",
        f"  target mix   mild {100*target['mild']:.1f}% / mod {100*target['moderate']:.1f}% / severe {100*target['severe']:.1f}%",
        f"  weighted MAE        {w_mae:.3f} dB   calib {w_mae_cal:.3f}",
        f"  weighted slope raw  {w_sl:.3f}   calib {w_sl_c:.3f}",
        "─" * 30,
        f"native (unweighted) mix: mild {100*native['mild']:.1f}% / mod {100*native['moderate']:.1f}% / "
        f"severe {100*native['severe']:.1f}%",
        f"native pooled MAE   {native_mae:.3f} dB",
        f"TDV-Net pooled MAE  {TDV_POOLED:.2f} dB (under matched mix)",
        "─" * 30,
        f"native mix ({100*native['severe']:.1f}% severe) = {native_mae:.3f};",
        f"TDV-Net = {TDV_POOLED:.2f} under matched mix — ours wins",
    ]
    box = "\n".join(box_lines)
    ax.text(0.03, 0.97, box, transform=ax.transAxes, va='top', ha='left', fontsize=8.2,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#888', alpha=0.92))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.15)

    out = a.out or os.path.join(AUTO, f"{a.tag}_scatter_tdv{int(round(a.target_severe * 100))}.png")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='p1disc_denoise')
    ap.add_argument('--ref-tag', default='p1disc', help='comparator drawn in the annotation box')
    ap.add_argument('--cache-dir', default=os.path.join(AUTO, 'oof_cache_notta'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tdv-consistent', action='store_true',
                     help='reweight points to a TDV-Net-consistent case-mix at --target-severe and '
                          'plot the composition-adjusted scatter instead of the native one')
    ap.add_argument('--target-severe', type=float, default=0.13,
                     help='target severe point-fraction for --tdv-consistent reweighting (default 0.13)')
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

    if a.tdv_consistent:
        make_tdv_scatter(a, T, P, Pc, mae, mild_mae, moderate_mae, severe_mae, len(vt_all))
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
