"""Figure 2 — model pipeline schematic for the reported fundus-only model (p1disc_denoise).

Revision 2026-07-27 — addresses the Jul-13 reviewer comments on the pipeline figure:
  * "Make clearer HOW GRAPE is fed into the pipeline"  -> an explicit GRAPE dataset card at the
    left holds the PAIRED record; the fundus half leaves it as the model INPUT, the VF half leaves
    it as the training LABEL, and the loss feeds back into the trainable parts.
  * "and how Garway-Heath goes in...?" + "add a VF cartoon and a Garway-Heath cartoon"
    -> a Garway-Heath panel drawn from the committed sector config: an optic-disc sector wheel
    (published GH angular limits) alongside the 52 VF points painted in the same sector colors,
    feeding the loss.
  * "change any arrows to just simple triangles" -> every connector is a large solid triangle;
    long routes are a plain line terminated by one.

Honest depiction (unchanged from the 2026-07-15 correction):
  * INFERENCE is fundus-only: Fundus -> Disc crop -> RETFound -> per-point attention -> Predicted VF.
    The per-point attention carries a RETINOTOPIC VF-grid->patch prior (build_vf_to_patch_prior in
    training.py) — this is the structural prior, and it is NOT the Garway-Heath map.
  * GARWAY-HEATH enters as the anatomical SECTOR WEIGHTING of the TRAINING LOSS (+ the frame for
    per-sector reporting), using the canonical 6-sector GH map. It is training-only.
  * GRAPE enters as TRAINING SUPERVISION: the measured VF is the loss target, never a runtime input.

Torch-free / self-contained: cartoons come from committed files (garway_heath_sectors.json,
cv_long/fold*_val.json, oof_cache_notta/*.npz, the fundus image).

  python decoder/make_architecture_figure.py  ->  decoder/results/auto/fig2_architecture.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Wedge
from matplotlib.colors import to_rgba, to_rgb
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(BASE, "decoder", "results", "auto")
FUND = os.path.join(BASE, "data", "fundus", "grape_fundus_images")

# ── VF 24-2 grid (inlined from decoder/diagnostics.py so this stays torch-free & self-contained) ──
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
VALID_OD = np.array([i for i, v in enumerate(mask_OD.flatten()) if v], int)   # 52, query order

DISC_CX_OD, DISC_CY, DISC_HALF = 0.78, 0.49, 0.27
FROZEN = "#d7e3f4"; FROZEN_E = "#4677b8"      # blue  = frozen / pretrained
TRAIN  = "#d8efdd"; TRAIN_E  = "#3f9a5c"      # green = trained on GRAPE / training-only
INK, MUTE, FLOW = "#20242b", "#5c6470", "#5f6874"
CARD, CARD_E = "#faf7f1", "#8d8578"           # GRAPE dataset card
INF_BG = "#f4f7fc"; TRAIN_BG = "#f5fbf6"
EX_IMG = "40_OD_2.jpg"

# Canonical Garway-Heath 6 sectors -> colors from the published GH sector figure, except Temporal,
# darkened from its published pale yellow so it cannot be confused with Nasal at print size.
# ids: 0=Temporal 1=Superotemporal 2=Superonasal 3=Nasal 4=Inferonasal 5=Inferotemporal
SECTOR_COLORS = {0: "#e2a33c", 1: "#53823a", 2: "#ec1d23",
                 3: "#f0ea1c", 4: "#97cb49", 5: "#2aade3"}
SECTOR_ABBR = {0: "T", 1: "ST", 2: "SN", 3: "N", 4: "IN", 5: "IT"}
SECTOR_FULL = {0: "Temporal", 1: "Superotemporal", 2: "Superonasal",
               3: "Nasal", 4: "Inferonasal", 5: "Inferotemporal"}
# Published GH disc-sector angular limits (right eye; 0 deg = temporal horizontal meridian,
# increasing superiorly). Converted to matplotlib wedge angles by  math = 180 - gh,
# which puts temporal to the LEFT and nasal to the RIGHT — the orientation of the OD disc crop.
GH_WEDGE = {0: (140, 229), 1: (100, 139), 2: (60, 99),
            3: (-50, 59), 4: (-90, -51), 5: (-130, -91)}

FIG_W, FIG_H = 13.8, 9.5
F_TITLE, F_CAP, F_LEG, F_TXT, F_SM = 13.5, 11.5, 12, 10.5, 10

# ── layout (inches) — three bordered sections: GRAPE card | Inference band | Training band ──
CX0, CX1, CY0, CY1 = 0.30, 2.60, 2.02, 8.45          # GRAPE dataset card
BX0, BX1 = 3.15, 13.50                               # shared band left/right
IY0, IY1 = 5.66, 8.00                                # inference band
TY0, TY1 = 0.45, 4.55                                # training band
yA, IMGH = 7.00, 1.42                                # inference row centreline / element height
CAP1, CAP2 = 6.20, 5.98                              # the two caption lines under a row element
ENC_X = (5.33, 7.48)                                 # RETFound box
ATT_X = (7.96, 10.11)                                # per-point attention box
PVF_CX = 11.34                                       # predicted-VF thumbnail centre
LANE, FB = 3.20, 4.10                                # label lane / gradient feedback lane
LX0, LX1, LY0, LY1 = 11.40, 13.30, 1.30, 3.90        # loss box
DROP = 12.60                                         # prediction drop column
GX0, GY0, GX1, GY1 = 3.42, 0.55, 9.82, 2.77          # Garway-Heath panel
EDGE = 0.10                                          # clearance left between an arrow tip and its target


# ══════════════════════════════════════════════════════════════════════════ data helpers
def _find_record(img):
    for f in range(5):
        js = json.load(open(os.path.join(BASE, "decoder", "results", "cv_long", f"fold{f}_val.json")))
        for i, r in enumerate(js):
            im = r["FundusImage"][0] if isinstance(r["FundusImage"], list) else r["FundusImage"]
            if im == img:
                return f, i, r
    raise ValueError(img)


def pred_grid_for(img):
    f, i, _ = _find_record(img)
    z = np.load(os.path.join(AUTO, "oof_cache_notta", f"p1disc_denoise_f{f}.npz"))
    g = np.full(72, np.nan); g[VALID_OD] = z["vp"][i]
    return g.reshape(8, 9)


def gt_grid_for(img):
    """The eye's real GRAPE ground-truth VF (the training label), masked, in the display grid."""
    _, _, r = _find_record(img)
    g = np.array(r["hvf"], float)
    return np.where(g >= 99.0, np.nan, g)


def sector_grid():
    """8x9 canonical Garway-Heath 6-sector-id grid (-1 = off-field), from the committed GH config."""
    cfg = json.load(open(os.path.join(BASE, "decoder", "garway_heath_sectors.json")))
    return np.array(cfg["sector_grid"], int)


def sector_rgba(grid):
    out = np.ones((*grid.shape, 4))
    for sid, col in SECTOR_COLORS.items():
        out[grid == sid] = to_rgba(col)
    return out


def _on(color):
    """Readable ink for text drawn on `color`."""
    r, g, b = to_rgb(color)
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


# ══════════════════════════════════════════════════════════════════ inches -> figure fraction
def fx(v):  return v / FIG_W
def fy(v):  return v / FIG_H


def panel(ax, x0, y0, x1, y1, fc, ec, lw=1.8, ls="-", r=0.02, z=0):
    ax.add_patch(FancyBboxPatch((fx(x0), fy(y0)), fx(x1 - x0), fy(y1 - y0),
                                boxstyle=f"round,pad=0.002,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z,
                                mutation_aspect=FIG_H / FIG_W))


def txt(ax, x, y, s, size=F_TXT, color=INK, weight="normal", ha="center", va="center",
        style="normal", ls_=1.18, z=7):
    ax.text(fx(x), fy(y), s, ha=ha, va=va, fontsize=size, color=color, fontweight=weight,
            style=style, linespacing=ls_, zorder=z)


def tri(ax, x, y, size=0.36, color=FLOW, d="right", z=6):
    """A plain solid direction triangle (no shaft) centred at (x, y) inches."""
    h, b = size, size * 0.94
    pts = {"right": [(x - h / 2, y + b / 2), (x - h / 2, y - b / 2), (x + h / 2, y)],
           "left":  [(x + h / 2, y + b / 2), (x + h / 2, y - b / 2), (x - h / 2, y)],
           "down":  [(x - b / 2, y + h / 2), (x + b / 2, y + h / 2), (x, y - h / 2)],
           "up":    [(x - b / 2, y - h / 2), (x + b / 2, y - h / 2), (x, y + h / 2)]}[d]
    ax.add_patch(Polygon([(fx(a), fy(c)) for a, c in pts], closed=True, fc=color, ec="none", zorder=z))


def route(ax, pts, color=FLOW, ls="-", lw=2.2, z=5):
    ax.plot([fx(p[0]) for p in pts], [fy(p[1]) for p in pts], color=color, ls=ls, lw=lw,
            solid_capstyle="round", zorder=z, clip_on=False)


def arrow(ax, pts, color=TRAIN_E, ls="-", lw=2.2, head=0.34):
    """Polyline ending in a solid triangular head whose TIP lands exactly on pts[-1].

    Callers place pts[-1] just short of the destination edge, so no head ever bleeds into
    the box it points at.
    """
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / n, dy / n
    route(ax, list(pts[:-1]) + [(x1 - ux * head, y1 - uy * head)], color, ls, lw)
    d = {(1, 0): "right", (-1, 0): "left", (0, 1): "up", (0, -1): "down"}[(round(ux), round(uy))]
    tri(ax, x1 - ux * head / 2, y1 - uy * head / 2, head, color, d)


def box(ax, x0, x1, yc, h, fc, ec, title, sub):
    panel(ax, x0, yc - h / 2, x1, yc + h / 2, fc, ec, lw=2.2, z=2)
    txt(ax, (x0 + x1) / 2, yc + 0.30, title, F_TITLE, INK, "bold")
    txt(ax, (x0 + x1) / 2, yc - 0.34, sub, F_SM, INK)


def img_axes(fig, cx, cy, s):
    return fig.add_axes([fx(cx - s / 2), fy(cy - s / 2), fx(s), fy(s)])


def vf_axes(fig, cx, cy, s, grid, cmap, edge=TRAIN_E, ls="-"):
    a = img_axes(fig, cx, cy, s)
    a.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=34, aspect="auto"); a.axis("off")
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(1.4); sp.set_linestyle(ls)
    return a


def disc_wheel(fig, cx, cy, box_in=1.40, lim=1.11):
    """Optic-disc sector wheel (right eye): the 6 Garway-Heath wedges at their published angles."""
    a = fig.add_axes([fx(cx - box_in / 2), fy(cy - box_in / 2), fx(box_in), fy(box_in)])
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.axis("off")
    for sid, (t1, t2) in GH_WEDGE.items():
        a.add_patch(Wedge((0, 0), 1.0, t1, t2, fc=SECTOR_COLORS[sid], ec="white", lw=1.5))
        m = np.deg2rad((t1 + t2) / 2)
        a.text(0.66 * np.cos(m), 0.66 * np.sin(m), SECTOR_ABBR[sid], ha="center", va="center",
               fontsize=F_SM, fontweight="bold", color=_on(SECTOR_COLORS[sid]))
    a.add_patch(Circle((0, 0), 1.0, fill=False, ec="#5b6270", lw=1.6))
    a.add_patch(Circle((0, 0), 0.30, fc="#fdfbf6", ec="#8b93a0", lw=1.2))     # cup
    return a


# ══════════════════════════════════════════════════════════════════════════════════ figure
def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
    cm = plt.get_cmap("inferno").copy(); cm.set_bad("white")

    LCX = (LX0 + LX1) / 2
    enc_cx, att_cx = sum(ENC_X) / 2, sum(ATT_X) / 2
    box_bot = yA - IMGH / 2

    # ── the three bordered sections, each with its title above ──────────────────────────
    panel(bg, BX0, IY0, BX1, IY1, INF_BG, FROZEN_E, lw=1.7, z=0)
    txt(bg, BX0, IY1 + 0.18, "Inference", F_CAP, FROZEN_E, "bold", ha="left")
    panel(bg, BX0, TY0, BX1, TY1, TRAIN_BG, TRAIN_E, lw=1.7, ls=(0, (7, 4)), z=0)
    txt(bg, BX0, TY1 + 0.18, "Training only", F_CAP, TRAIN_E, "bold", ha="left")

    # ── GRAPE dataset card (holds the paired record; spans both bands) ───────────────────
    ccx = (CX0 + CX1) / 2
    panel(bg, CX0, CY0, CX1, CY1, CARD, CARD_E, lw=2.0, z=1)
    txt(bg, CX0, CY1 + 0.18, "GRAPE dataset", F_CAP, INK, "bold", ha="left")
    txt(bg, ccx, 8.18, "631 paired fundus–VF records\n263 eyes  ·  144 patients", F_SM, MUTE)

    im = Image.open(os.path.join(FUND, EX_IMG)).convert("RGB")
    w, h = im.size; dcx, dcy, hf = DISC_CX_OD * w, DISC_CY * h, DISC_HALF * w
    fa = img_axes(fig, ccx, yA, IMGH); fa.imshow(im, aspect="auto"); fa.axis("off")
    fa.add_patch(Rectangle((dcx - hf, dcy - hf), 2 * hf, 2 * hf, fill=False,
                           ec="#ffe14d", lw=2.2, ls="--"))
    txt(bg, ccx, CAP1, "Fundus photo", F_CAP, INK, "bold", va="top")
    txt(bg, ccx, CAP2, "Model input", F_SM, FROZEN_E, "bold", va="top")

    route(bg, [(ccx, 5.72), (ccx, 5.44)], CARD_E, ls=(0, (2, 2.4)), lw=1.6)
    panel(bg, CX0 + 0.16, 4.86, CX1 - 0.16, 5.42, "#ffffff", CARD_E, lw=1.3, r=0.015, z=1)
    txt(bg, ccx, 5.14, "One paired record\nSame eye · same visit", F_SM, INK, style="italic")
    route(bg, [(ccx, 4.84), (ccx, 3.86)], CARD_E, ls=(0, (2, 2.4)), lw=1.6)

    vf_axes(fig, ccx, LANE, 1.20, gt_grid_for(EX_IMG), cm, TRAIN_E, ls=(0, (3, 2)))
    txt(bg, ccx, 2.50, "Measured 24-2 VF", F_CAP, INK, "bold", va="top")
    txt(bg, ccx, 2.28, "Training label", F_SM, TRAIN_E, "bold", va="top")

    # ── inference row ───────────────────────────────────────────────────────────────────
    tri(bg, (CX1 + BX0) / 2, yA)
    da = img_axes(fig, 4.14, yA, IMGH)
    da.imshow(im.crop((int(dcx - hf), int(dcy - hf), int(dcx + hf), int(dcy + hf))).resize((224, 224)),
              aspect="auto")
    da.axis("off")
    for sp in da.spines.values():
        sp.set_visible(True); sp.set_edgecolor("#ffc61a"); sp.set_linewidth(1.8)
    txt(bg, 4.14, CAP1, "Disc crop", F_CAP, INK, "bold", va="top")
    txt(bg, 4.14, CAP2, "224 × 224", F_SM, MUTE, va="top")

    tri(bg, 5.09, yA)
    box(bg, *ENC_X, yA, IMGH, FROZEN, FROZEN_E, "RETFound\nViT-L",
        "Encoder · frozen\nOnly LoRA adapters train")
    tri(bg, 7.72, yA)
    box(bg, *ATT_X, yA, IMGH, TRAIN, TRAIN_E, "Per-point\nattention",
        "Decoder · 52 point-queries\nRetinotopic prior + global\n+ severity heads")
    tri(bg, 10.37, yA)
    vf_axes(fig, PVF_CX, yA, IMGH, pred_grid_for(EX_IMG), cm, "#9aa1ac")
    txt(bg, PVF_CX, CAP1, "Predicted 24-2 VF", F_CAP, INK, "bold", va="top")
    txt(bg, PVF_CX, CAP2, "Model output", F_SM, MUTE, va="top")

    # ── Garway-Heath panel ──────────────────────────────────────────────────────────────
    panel(bg, GX0, GY0, GX1, GY1, "#ffffff", "#b9c0ca", lw=1.5, z=1)
    txt(bg, GX0 + 0.17, 2.59, "Garway–Heath sector map", F_CAP, INK, "bold", ha="left")
    txt(bg, GX0 + 0.17, 2.37,
        "Every VF location is tied to the optic-disc sector whose nerve fibers serve it",
        F_SM, MUTE, ha="left")
    disc_wheel(fig, 4.62, 1.60)
    txt(bg, 4.62, 0.93, "Optic disc — right eye\nSuperior up · temporal left", F_SM, INK, va="top")
    tri(bg, 5.69, 1.60, 0.30, "#8a919c")
    sa = img_axes(fig, 6.72, 1.60, 1.18)
    sa.imshow(sector_rgba(sector_grid()), aspect="auto"); sa.axis("off")
    for sp in sa.spines.values():
        sp.set_visible(True); sp.set_edgecolor("#9aa1ac"); sp.set_linewidth(1.2)
    txt(bg, 6.72, 0.93, "The 52 VF points\nColored by disc sector", F_SM, INK, va="top")
    for k, sid in enumerate([1, 2, 3, 4, 5, 0]):
        ky = 2.22 - 0.25 * k
        bg.add_patch(Rectangle((fx(7.97), fy(ky - 0.075)), fx(0.16), fy(0.15),
                               fc=SECTOR_COLORS[sid], ec="#7d848f", lw=0.9, zorder=3))
        txt(bg, 8.21, ky, f"{SECTOR_ABBR[sid]}   {SECTOR_FULL[sid]}", 9.5, INK, ha="left")

    # ── the loss: three inputs in, gradients out ────────────────────────────────────────
    panel(bg, LX0, LY0, LX1, LY1, TRAIN, TRAIN_E, lw=2.2, z=2)
    txt(bg, LCX, 3.22, "Garway–Heath\nsector-weighted\ntraining loss", F_TITLE, TRAIN_E, "bold")
    txt(bg, LCX, 2.30, "Per point:\nprediction error ×\nits disc-sector weight", F_SM, INK)
    txt(bg, LCX, 1.68, "(Also the frame for the\nper-sector error report)", 9.5, MUTE, style="italic")

    arrow(bg, [(CX1 + 0.04, LANE), (LX0 - EDGE, LANE)])                     # GRAPE label -> loss
    txt(bg, (CX1 + LX0) / 2, LANE + 0.21,
        "GRAPE ground-truth VF supervises the prediction", F_TXT, TRAIN_E, "bold")

    arrow(bg, [(GX1 + 0.12, 1.60), (LX0 - EDGE, 1.60)])                     # GH weights -> loss

    arrow(bg, [(PVF_CX + IMGH / 2, yA), (DROP, yA), (DROP, LY1 + EDGE)])    # prediction -> loss
    txt(bg, DROP + 0.14, (TY1 + IY0) / 2, "Prediction", F_TXT, TRAIN_E, "bold", ha="left")

    for cx in (enc_cx, att_cx):                                             # gradients -> trainable
        arrow(bg, [(LX0 + 0.30, LY1), (LX0 + 0.30, FB), (cx, FB), (cx, box_bot - EDGE)],
              ls=(0, (5.5, 3.5)), lw=2.0)

    # ── legend ──────────────────────────────────────────────────────────────────────────
    ly = 9.25
    keys = [(FROZEN, FROZEN_E, "Frozen (pretrained)", None),
            (TRAIN, TRAIN_E, "Trained on GRAPE", None),
            (None, TRAIN_E, "Training only", (0, (5, 3)))]
    xs = [FIG_W / 2 - 3.55, FIG_W / 2 - 1.05, FIG_W / 2 + 1.45]
    for (fc, ec, label, ls), xk in zip(keys, xs):
        if fc is None:
            route(bg, [(xk, ly), (xk + 0.30, ly)], ec, ls=ls, lw=2.2)
        else:
            bg.add_patch(Rectangle((fx(xk), fy(ly - 0.10)), fx(0.28), fy(0.20), fc=fc, ec=ec, lw=1.5))
        txt(bg, xk + 0.40, ly, label, F_LEG, INK, ha="left")

    out = os.path.join(AUTO, "fig2_architecture.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.16); plt.close(fig)
    print("wrote", out, f"({FIG_W}x{FIG_H} in; min font 8.5pt)")


if __name__ == "__main__":
    main()
