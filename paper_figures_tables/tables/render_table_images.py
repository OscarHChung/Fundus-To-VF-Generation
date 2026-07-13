"""Render the three paper tables as clean PNG images for pasting into Google Docs.

Usage:  python paper_figures_tables/tables/render_table_images.py
Output: Table1_sectoral_mae.png, Table2_comparison_prior_studies.png,
        Table3_composition_adjusted.png  (next to this script)
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

HEADER_BG = "#d9e2f3"
HIGHLIGHT_BG = "#eef4ff"
ALT_BG = "#f7f9fc"
EDGE = "#5a6b8c"


def wrap(cell, width):
    return "\n".join(textwrap.wrap(str(cell), width)) if width else str(cell)


def render(fname, title, columns, rows, col_widths, wrap_widths,
           highlight_rows=(), legend="", row_scale=1.0):
    ncols = len(columns)
    # wrap long cells
    body = [[wrap(rows[r][c], wrap_widths[c]) for c in range(ncols)] for r in range(len(rows))]
    hdr = [wrap(columns[c], wrap_widths[c]) for c in range(ncols)]

    def nlines(cell):
        return cell.count("\n") + 1
    row_heights = [max(nlines(body[r][c]) for c in range(ncols)) for r in range(len(rows))]
    hdr_h = max(nlines(hdr[c]) for c in range(ncols))
    total_lines = hdr_h + sum(row_heights)

    fig_w = sum(col_widths) * 1.05
    fig_h = 0.34 * total_lines * row_scale + (1.1 if legend else 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)

    tbl = ax.table(cellText=body, colLabels=hdr, cellLoc="left", loc="upper left",
                   colWidths=[w / sum(col_widths) for w in col_widths])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(EDGE)
        cell.set_linewidth(0.8)
        cell.PAD = 0.04
        if r == 0:  # header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(fontweight="bold")
            cell.set_height(cell.get_height() * hdr_h * 0.9)
        else:
            data_r = r - 1
            cell.set_height(cell.get_height() * row_heights[data_r] * 0.9)
            if data_r in highlight_rows:
                cell.set_facecolor(HIGHLIGHT_BG)
                cell.set_text_props(fontweight="bold")
            elif data_r % 2 == 1:
                cell.set_facecolor(ALT_BG)

    if legend:
        fig.text(0.01, 0.01, legend, fontsize=8, color="#333", va="bottom", wrap=True)

    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


# ---- Table 1 -------------------------------------------------------------
render(
    "Table1_sectoral_mae.png",
    "Table 1 — Sectoral MAE (Garway-Heath)",
    ["Garway-Heath sector", "MAE (dB)"],
    [["Inferior-arcuate", "3.80"], ["Inferonasal", "3.80"], ["Central-inferior", "3.80"],
     ["Nasal-periphery", "4.14"], ["Central-superior", "4.15"], ["Superonasal", "4.25"],
     ["Superior-arcuate", "4.37"], ["Temporal-wedge", "4.37"]],
    col_widths=[3.4, 1.4], wrap_widths=[0, 0],
    legend="Legend. Pooled per-location MAE within each Garway-Heath disc sector (out-of-fold,\n"
           "leak-free per-patient 5-fold CV). Lowest in inferior/central sectors, highest in the\n"
           "superior-arcuate and temporal-wedge sectors (superior field).",
)

# ---- Table 2 -------------------------------------------------------------
render(
    "Table2_comparison_prior_studies.png",
    "Table 2 — Comparison with prior fundus-to-VF studies",
    ["Study", "Data (source)", "Prediction target", "Pooled MAE (dB)",
     "Mild / Mod / Severe (dB)", "External validation"],
    [["This study", "GRAPE, open-access (631 records)", "24-2 raw sensitivity",
      "4.10 (composition-adjusted 3.27–3.65)", "2.90 / 4.06 / 8.43", "PAPILA, r 0.75"],
     ["Park et al. 2026 (10)", "Private, 2 tertiary hospitals (~37.9k photos)",
      "24-2 total deviation", "3.91", "3.09 / 5.66 / 9.15", "None reported"],
     ["Huang et al. 2025 (11)", "Private (1,129 eyes)", "Octopus 59-point sensitivity†",
      "3.10–4.13", "—", "None reported"],
     ["Kang et al. 2024 (9)", "Private", "VF image (cGAN)‡", "—", "—", "None reported"]],
    col_widths=[1.7, 2.3, 1.7, 1.9, 1.5, 1.5],
    wrap_widths=[16, 22, 16, 18, 14, 14],
    highlight_rows=[0],
    legend="Legend. † Different device/point layout/dynamic range — not directly comparable to 24-2 dB. "
           "‡ Image-translation output judged by\nPSNR 30.61 dB / SSIM 0.48, not clinical sensitivity "
           "error. Mild = each location's true sensitivity ≥22 dB; moderate 15–22; severe <15.\n"
           "Our mild figure is statistically equivalent to Park et al.'s (within measurement error).",
)

# ---- Table 3 -------------------------------------------------------------
render(
    "Table3_composition_adjusted.png",
    "Table 3 — Composition-adjusted comparison vs Park et al. 2026",
    ["Severity stratum", "Our MAE (dB)", "Our % of points", "Park et al. MAE (dB)", "Δ (ours − theirs)"],
    [["Mild (≥22 dB)", "2.90", "61.2%", "3.09", "−0.19 (equivalent)"],
     ["Moderate (15–22 dB)", "4.06", "21.6%", "5.66", "−1.60"],
     ["Severe (<15 dB)", "8.43", "17.2%", "9.15", "−0.73"],
     ["Pooled, native mix", "4.10", "—", "3.91", "+0.19 (case-mix inversion)"],
     ["Pooled, matched mix", "3.27–3.65", "—", "3.91", "≤ −0.26"],
     ["Park et al. under our mix", "—", "—", "4.69", "—"]],
    col_widths=[2.4, 1.5, 1.7, 2.0, 3.0], wrap_widths=[0, 0, 0, 0, 0],
    legend="Legend. Each model recomputed under matched severity compositions using our own out-of-fold "
           "errors (no retraining). Pooled\nnative MAEs are not directly comparable (cohorts differ in "
           "severity mix); the matched-mix rows are. Point-level strata\n(each point bucketed by its own "
           "true sensitivity) — the convention Park et al.'s numbers are directly comparable under.",
)
print("done")
