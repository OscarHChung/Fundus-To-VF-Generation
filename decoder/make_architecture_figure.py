"""Figure 2 — model pipeline schematic for the reported fundus-only model (p1disc_denoise).

Restored from the original horizontal-pipeline layout, with only two changes: bigger fonts and
tighter (less empty) text boxes, which also narrows the overall figure. Torch-free.

  python decoder/make_architecture_figure.py  ->  decoder/results/auto/fig2_architecture.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(BASE, "decoder", "results", "auto")
FUND = os.path.join(BASE, "data", "fundus", "grape_fundus_images")
_G = json.load(open(os.path.join(
    "/private/tmp/claude-501/-Users-oscarchung-Documents-Python-Projects-Fundus-To-VF-Generation/"
    "f4b3cc59-4014-4e41-8eb4-5db61c1e602b/scratchpad", "grid.json")))
VALID_OD = np.array(_G["valid_indices_od"], int)

DISC_CX_OD, DISC_CY, DISC_HALF = 0.78, 0.49, 0.27
FROZEN = "#d7e3f4"; FROZEN_E = "#4677b8"      # blue  = frozen / pretrained
TRAIN  = "#d8efdd"; TRAIN_E  = "#3f9a5c"      # green = trained on GRAPE
BAND   = "#f2f0ea"; INK = "#20242b"
EX_IMG = "40_OD_2.jpg"

# ---- (change #1) bigger fonts; ---- (change #2) tighter boxes ----
F_TITLE, F_SUB, F_CAP, F_LEG, F_BAND = 14, 12.5, 12.5, 12, 12.5
FIG_W, FIG_H = 9.7, 4.3
BOX_W, BOX_H, GAP = 1.72, 1.55, 0.30                 # both boxes share one width; wide gaps for arrows
IMG = BOX_H                                          # images are square and the same height as the boxes


def pred_grid_for(img):
    for f in range(5):
        js = json.load(open(os.path.join(BASE, "decoder", "results", "cv_long", f"fold{f}_val.json")))
        for i, r in enumerate(js):
            im = r["FundusImage"][0] if isinstance(r["FundusImage"], list) else r["FundusImage"]
            if im == img:
                z = np.load(os.path.join(AUTO, "oof_cache_notta", f"p1disc_denoise_f{f}.npz"))
                g = np.full(72, np.nan); g[VALID_OD] = z["vp"][i]; return g.reshape(8, 9)
    raise ValueError(img)


def fx(inches):  return inches / FIG_W
def fy(inches):  return inches / FIG_H


def box(ax, x_in, w_in, yc, h_in, fc, ec, title, sub):
    x, w, h = fx(x_in), fx(w_in), fy(h_in)
    ax.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc=fc, ec=ec, lw=2, mutation_aspect=FIG_H / FIG_W))
    ax.text(x + w / 2, yc + fy(0.32), title, ha="center", va="center",
            fontsize=F_TITLE, fontweight="bold", color=INK, linespacing=1.05)
    ax.text(x + w / 2, yc - fy(0.36), sub, ha="center", va="center",
            fontsize=F_SUB, color=INK, linespacing=1.12)


def arrow(ax, x0_in, x1_in, yc):
    ax.add_patch(FancyArrowPatch((fx(x0_in), yc), (fx(x1_in), yc),
                                 arrowstyle="-|>", mutation_scale=18, lw=2.3, color="#7c8595"))


def img_axes(fig, x_in, yc, s_in):
    return fig.add_axes([fx(x_in), yc - fy(s_in) / 2, fx(s_in), fy(s_in)])


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

    yc = 0.585                                   # common centerline for every stage
    cap_y = yc - fy(BOX_H / 2 + 0.30)            # one caption baseline under the whole row (spaced down)
    im = Image.open(os.path.join(FUND, EX_IMG)).convert("RGB")
    w, h = im.size; cx, cy, hf = DISC_CX_OD * w, DISC_CY * h, DISC_HALF * w

    total = 3 * IMG + 2 * BOX_W + 4 * GAP
    x0 = (FIG_W - total) / 2
    x = x0
    gaps = []                                    # (start,end) of each inter-stage gap, in inches

    def cap(cx_in, text):
        bg.text(fx(cx_in), cap_y, text, ha="center", fontsize=F_CAP, fontweight="bold", color=INK)

    # 1) fundus
    fa = img_axes(fig, x, yc, IMG); fa.imshow(im, aspect="auto"); fa.axis("off")
    fa.add_patch(Rectangle((cx - hf, cy - hf), 2 * hf, 2 * hf, fill=False, ec="#ffe14d", lw=2, ls="--"))
    cap(x + IMG / 2, "Fundus"); x += IMG
    gaps.append((x, x + GAP)); x += GAP
    # 2) disc crop
    da = img_axes(fig, x, yc, IMG)
    da.imshow(im.crop((int(cx - hf), int(cy - hf), int(cx + hf), int(cy + hf))).resize((224, 224)),
              aspect="auto")
    da.axis("off")
    for s in da.spines.values():
        s.set_visible(True); s.set_edgecolor(FROZEN_E); s.set_linewidth(1.4)
    cap(x + IMG / 2, "Disc crop"); x += IMG
    gaps.append((x, x + GAP)); x += GAP
    # 3) encoder box (frozen) — equal width to the decoder box
    box(bg, x, BOX_W, yc, BOX_H, FROZEN, FROZEN_E, "RETFound\nViT-L", "frozen + LoRA")
    cap(x + BOX_W / 2, "encoder"); x += BOX_W
    gaps.append((x, x + GAP)); x += GAP
    # 4) per-point attention box (trained) — sub wrapped to 3 short lines for interior padding
    box(bg, x, BOX_W, yc, BOX_H, TRAIN, TRAIN_E, "Per-point\nattention",
        "Garway–Heath\nprior + global\n+ severity")
    cap(x + BOX_W / 2, "decoder"); x += BOX_W
    gaps.append((x, x + GAP)); x += GAP
    # 5) predicted VF — aspect='auto' so the grid fills the same square as the photos
    va = img_axes(fig, x, yc, IMG)
    g = np.ma.masked_invalid(pred_grid_for(EX_IMG)); cm = plt.get_cmap("inferno").copy(); cm.set_bad("white")
    va.imshow(g, cmap=cm, vmin=0, vmax=34, aspect="auto"); va.axis("off")
    cap(x + IMG / 2, "Predicted VF")

    # arrows LAST, so a later-drawn box fill can never cover an arrowhead; ends held clear of edges
    for a, b in gaps:
        arrow(bg, a + 0.05, b - 0.05, yc)

    # legend (top), centered
    ly = 0.925
    fc_sw = FIG_W / 2 - 1.95; tr_sw = FIG_W / 2 + 0.55
    bg.add_patch(Rectangle((fx(fc_sw), ly - fy(0.10)), fx(0.24), fy(0.20), fc=FROZEN, ec=FROZEN_E, lw=1.4))
    bg.text(fx(fc_sw + 0.32), ly, "frozen (pretrained)", va="center", fontsize=F_LEG, color=INK)
    bg.add_patch(Rectangle((fx(tr_sw), ly - fy(0.10)), fx(0.24), fy(0.20), fc=TRAIN, ec=TRAIN_E, lw=1.4))
    bg.text(fx(tr_sw + 0.32), ly, "trained on GRAPE", va="center", fontsize=F_LEG, color=INK)

    # bottom band — spans the centered content row
    bl = x0 - 0.02
    bg.add_patch(FancyBboxPatch((fx(bl), fy(0.14)), fx(total + 0.04), fy(0.82),
                                boxstyle="round,pad=0.003,rounding_size=0.02",
                                fc=BAND, ec="#d7d2c6", lw=1.3, mutation_aspect=FIG_H / FIG_W))
    bg.text(fx(bl + 0.18), fy(0.70), "Training", fontsize=F_BAND, fontweight="bold",
            color=TRAIN_E, va="center")
    bg.text(fx(bl + 1.08), fy(0.70),
            "GRAPE pairs · Theil–Sen denoised targets · GH-weighted loss",
            fontsize=F_BAND, color=INK, va="center")
    bg.text(fx(bl + 0.18), fy(0.36), "Evaluation", fontsize=F_BAND, fontweight="bold",
            color=FROZEN_E, va="center")
    bg.text(fx(bl + 1.26), fy(0.36),
            "leak-free 5-fold CV · external validation on PAPILA (r = 0.75)",
            fontsize=F_BAND, color=INK, va="center")

    out = os.path.join(AUTO, "fig2_architecture.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.16); plt.close(fig)
    print("wrote", out, f"({FIG_W}x{FIG_H} in; min font {min(F_TITLE,F_SUB,F_CAP,F_LEG,F_BAND)}pt)")


if __name__ == "__main__":
    main()
