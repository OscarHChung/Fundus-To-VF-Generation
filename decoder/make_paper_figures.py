"""Paper figures for the fundus-only reported model (p1disc_denoise), built directly from the
frozen OOF cache + fold val JSONs — NO torch, NO live inference.

  python decoder/make_paper_figures.py

Outputs (decoder/results/auto/):
  p1disc_denoise_mae_heatmap.png   Figure 3 — bilateral per-location MAE + GT sensitivity (5-fold OOF)
  p1disc_denoise_examples.png      Figure 5 — best fundus->VF example per severity (mild/mod/severe)

Row i of oof_cache_notta/p1disc_denoise_f{f}.npz maps to record i of cv_long/fold{f}_val.json
(the eval loader is batch_size=1, shuffle=False, one row per record — verified).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO  = os.path.join(BASE, "decoder", "results", "auto")
CVDIR = os.path.join(BASE, "decoder", "results", "cv_long")
CACHE = os.path.join(AUTO, "oof_cache_notta")
FUND  = os.path.join(BASE, "data", "fundus", "grape_fundus_images")

# ── VF 24-2 grid (inlined) + canonical Garway–Heath 6-sector map (query-order convention A) ──
#   Formerly loaded from a since-deleted scratchpad grid.json; the sector map is now read from the
#   single source of truth (decoder/garway_heath_sectors.json), so a re-section propagates here
#   automatically.
mask_OD = np.array([
    [False, False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, False, False, True,  True,  True,  True,  False, False],
], dtype=bool)
VALID_OD = np.array([i for i, v in enumerate(mask_OD.flatten()) if v], int)
VALID_OS = np.array([i for i, v in enumerate(np.fliplr(mask_OD).flatten()) if v], int)
SECTOR_OD = np.array(json.load(open(os.path.join(
    BASE, "decoder", "garway_heath_sectors.json")))["sector_grid"], float)  # 8x9, -1 masked


def vec_to_grid(vec52, eye):
    """Place a 52-vector (native query order for `eye`) into an 8x9 display grid."""
    g = np.full(72, np.nan)
    vi = VALID_OD if eye == "OD" else VALID_OS
    g[vi] = vec52
    g = g.reshape(8, 9)
    return np.fliplr(g) if eye == "OS" else g


def sector_disp(eye):
    """Sector-id grid in the SAME display orientation `vec_to_grid` produces for `eye`, so the
    boundary overlay lines up with the MAE grid for either eye (the old OS overlay used a raw
    fliplr that was mirror-misaligned against the vec_to_grid'd OS MAE)."""
    src = SECTOR_OD if eye == "OD" else np.fliplr(SECTOR_OD)
    vi = VALID_OD if eye == "OD" else VALID_OS
    return vec_to_grid(np.array([src.flat[i] for i in vi], float), eye)


def load_records():
    recs = []
    for f in range(5):
        js = json.load(open(os.path.join(CVDIR, f"fold{f}_val.json")))
        z = np.load(os.path.join(CACHE, f"p1disc_denoise_f{f}.npz"))
        vp, vt = z["vp"], z["vt"]
        assert len(js) == vp.shape[0]
        for i, r in enumerate(js):
            t = vt[i].astype(float); p = vp[i].astype(float)
            m = t < 99.0
            img = r["FundusImage"][0] if isinstance(r["FundusImage"], list) else r["FundusImage"]
            recs.append(dict(lat=r["Laterality"], img=img, pred=p, true=t, mask=m,
                             mae=float(np.abs(p[m] - t[m]).mean()),
                             mgt=float(t[m].mean()), npts=int(m.sum())))
    return recs


def plot_grid(ax, grid, title, cmap, vmin, vmax, cbar_label, boundaries=None):
    masked = np.ma.masked_invalid(grid)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=10)
    if boundaries is not None:
        sg = boundaries
        def valid(v):
            return not (np.isnan(v) or v < 0)
        for r in range(8):
            for c in range(9):
                s = sg[r, c]
                if not valid(s):
                    continue
                # interior boundaries only (both cells valid, different sector) → no
                # lines protruding into the masked/blind-spot white space
                if c + 1 < 9 and valid(sg[r, c + 1]) and sg[r, c + 1] != s:
                    ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], "k-", lw=1.4)
                if r + 1 < 8 and valid(sg[r + 1, c]) and sg[r + 1, c] != s:
                    ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], "k-", lw=1.4)


# ---------------------------------------------------------------- Figure 3: heatmap
def make_heatmap(recs):
    res = {}
    for eye in ("OD", "OS"):
        sub = [r for r in recs if r["lat"].startswith(eye)]
        P = np.stack([r["pred"] for r in sub]); T = np.stack([r["true"] for r in sub])
        M = T < 99.0
        ae = np.abs(P - T)
        mean_ae = np.array([ae[M[:, j], j].mean() for j in range(52)])
        mean_gt = np.array([T[M[:, j], j].mean() for j in range(52)])
        pooled = ae[M].mean()
        res[eye] = dict(mae=vec_to_grid(mean_ae, eye), gt=vec_to_grid(mean_gt, eye),
                        pooled=pooled, n=len(sub))
    comb = np.average([res["OD"]["pooled"], res["OS"]["pooled"]],
                      weights=[res["OD"]["n"], res["OS"]["n"]])

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        "VF Prediction — MAE & Ground-Truth Sensitivity  |  5-fold out-of-fold (631 eyes)\n"
        f"OD MAE: {res['OD']['pooled']:.2f} dB   |   OS MAE: {res['OS']['pooled']:.2f} dB"
        f"   |   Combined: {comb:.2f} dB", fontsize=15, fontweight="bold")
    plot_grid(ax[0, 0], res["OD"]["mae"], "OD — Avg MAE per Location (Right Eye)",
              "inferno", 0, 10, "MAE (dB)", boundaries=sector_disp("OD"))
    plot_grid(ax[0, 1], res["OS"]["mae"], "OS — Avg MAE per Location (Left Eye)",
              "inferno", 0, 10, "MAE (dB)", boundaries=sector_disp("OS"))
    plot_grid(ax[1, 0], res["OD"]["gt"], "OD — Avg GT Sensitivity (Right Eye)",
              "inferno", 0, 30, "Sensitivity (dB)")
    plot_grid(ax[1, 1], res["OS"]["gt"], "OS — Avg GT Sensitivity (Left Eye)",
              "inferno", 0, 30, "Sensitivity (dB)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(AUTO, "p1disc_denoise_mae_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, f"(OD {res['OD']['n']} / OS {res['OS']['n']} eyes, combined {comb:.2f} dB)")


# ---------------------------------------------------------------- Figure 5: examples
EXAMPLES = [("Mild", "40_OD_2.jpg"), ("Moderate", "34_OD_1.jpg"), ("Severe", "120_OD_4.jpg")]

def make_examples(recs):
    by_img = {r["img"]: r for r in recs}
    fig, ax = plt.subplots(3, 3, figsize=(11, 11))
    col_titles = ["Fundus photograph", "Measured VF", "Predicted VF"]
    for j, t in enumerate(col_titles):
        ax[0, j].set_title(t, fontsize=13, fontweight="bold", pad=10)
    im_vf = None
    for i, (label, img) in enumerate(EXAMPLES):
        r = by_img[img]; eye = r["lat"]
        # fundus
        ax[i, 0].imshow(Image.open(os.path.join(FUND, img)).convert("RGB"))
        ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
        ax[i, 0].set_ylabel(f"{label}\nMAE {r['mae']:.1f} dB", fontsize=12,
                            fontweight="bold", rotation=90, labelpad=12)
        # true / pred VF
        for j, key in enumerate(["true", "pred"], start=1):
            vec = r[key].copy()
            if key == "true":
                vec = np.where(r["mask"], vec, np.nan)
            g = vec_to_grid(vec, eye)
            masked = np.ma.masked_invalid(g)
            cm = plt.get_cmap("inferno").copy(); cm.set_bad("white")
            im_vf = ax[i, j].imshow(masked, cmap=cm, vmin=0, vmax=34)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for s in ax[i, j].spines.values():
                s.set_visible(False)
    fig.suptitle("Fundus photograph → predicted 24-2 visual field, by severity",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
    cb = fig.colorbar(im_vf, cax=cax); cb.set_label("Sensitivity (dB)", fontsize=11)
    out = os.path.join(AUTO, "p1disc_denoise_examples.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def _ols(x, y):
    b, a = np.polyfit(x, y, 1)            # slope, intercept
    return b, a


# ---------------------------------------------------------------- Figure 4: scatter
def make_scatter():
    xr, yr, yc = [], [], []               # true, raw pred, calibrated pred (pooled valid points)
    for f in range(5):
        z = np.load(os.path.join(CACHE, f"p1disc_denoise_f{f}.npz"))
        vp, vt, tp, tt = z["vp"], z["vt"], z["tp"], z["tt"]
        # variance-match calibration fit on this fold's TRAIN points (eval_ckpt.pooled_stats/apply_calib)
        mt = ~np.isnan(tt); mp = ~np.isnan(tt)
        mu_p, sig_p = tp[mp].mean(), tp[mp].std()
        mu_t, sig_t = tt[mt].mean(), tt[mt].std()
        b = sig_t / (sig_p + 1e-8)
        m = ~np.isnan(vt)
        xr.append(vt[m]); yr.append(vp[m])
        yc.append(np.clip(mu_t + b * (vp[m] - mu_p), 0, 35))
    x = np.concatenate(xr); yraw = np.concatenate(yr); ycal = np.concatenate(yc)
    sr, ir = _ols(x, yraw); sc, ic = _ols(x, ycal)
    mae = np.abs(yraw - x).mean(); r = np.corrcoef(x, yraw)[0, 1]
    print(f"scatter: n={x.size}  MAE={mae:.3f}  r={r:.3f}  raw {sr:.3f}x+{ir:.2f}  calib {sc:.3f}x+{ic:.2f}")

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.scatter(x, yraw, s=5, c="#3b6ea5", alpha=0.05, edgecolors="none", rasterized=True)
    line = np.array([0, 36])
    ax.plot(line, line, "--", color="#8a8f99", lw=1.8, label="y = x")
    ax.plot(line, sr * line + ir, "-", color="#c0392b", lw=2.6, label=f"fit: y = {sr:.2f}x + {ir:.1f}")
    ax.plot(line, sc * line + ic, "-", color="#3f9a5c", lw=2.6, label=f"calibrated: y = {sc:.2f}x + {ic:.1f}")
    ax.set_xlim(0, 36); ax.set_ylim(0, 36); ax.set_aspect("equal")
    ax.set_xlabel("True 24-2 sensitivity (dB)", fontsize=12)
    ax.set_ylabel("Predicted sensitivity (dB)", fontsize=12)
    ax.set_title("Predicted vs. true 24-2 visual field sensitivity", fontsize=14, fontweight="bold")
    ax.text(1.5, 34, f"MAE {mae:.2f} dB    r {r:.2f}\nslope {sr:.2f}  ({sc:.2f} calibrated)",
            fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#d0d0d0"))
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.15)
    out = os.path.join(AUTO, "p1disc_denoise_scatter_clean.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    recs = load_records()
    print(f"loaded {len(recs)} OOF records")
    make_heatmap(recs)
    make_examples(recs)
    make_scatter()
