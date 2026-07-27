# Pipeline figure revision — make GRAPE and Garway–Heath legible

> ⚠️ **Layout superseded 2026-07-27** by a second pass over the same reviewer comments (the Jul-15
> version left GRAPE implicit and drew GH as a bare VF sector thumbnail). The figure now carries an
> explicit **GRAPE dataset card** (the paired record: fundus = input, VF = label), a **disc-sector
> wheel ↔ VF sector map** cartoon, a **gradient feedback path** into the trainable parts, and **solid
> triangle connectors** instead of thin arrows. Scientific framing below is unchanged.
>
> ⚠️ **Partly superseded same day** by `2026-07-15-garway-heath-sectoring-fix-design.md`. This
> revision's GH depiction (8-sector draft map, drawn as if it fed the attention *prior*) was wrong on
> two counts and was corrected: the sectoring is now the **canonical 6-sector Garway–Heath map**, and
> it is shown as the **training-loss weighting**, not the attention prior (which is retinotopic). The
> GRAPE-supervision and layout decisions here still stand.

**Date:** 2026-07-15
**File touched:** `decoder/make_architecture_figure.py` → `decoder/results/auto/fig2_architecture.png`
**Origin:** reviewer comments from Sophia Wang (Jul 13) on the model pipeline figure.

## Problem

The current Figure 2 (fundus-only model, `p1disc_denoise`) is a clean left→right pipeline:
`Fundus → Disc crop → RETFound ViT-L [frozen] → Per-point attention [trained] → Predicted VF`,
plus a bottom Training/Evaluation band. Two things a reader cannot see from it:

1. **How GRAPE feeds in.** GRAPE appears only as text ("trained on GRAPE", "GRAPE pairs"). It
   is not visible *how* the dataset participates.
2. **How Garway–Heath goes in.** GH is only the text "Garway–Heath prior" inside the decoder
   box. Its mechanism (an anatomical disc→VF sector map that biases the attention) is invisible.

Reviewer ask (verbatim): *"probably need to add to figure — VF cartoon, and Garway–Heath
cartoon and make it clear how these are being input into the model somehow."*

**Secondary defect (must fix):** the script loads `grid.json` from a hard-coded scratchpad path
belonging to a **dead session** (`.../f4b3cc59-…/scratchpad/grid.json`). That file no longer
exists, so `python decoder/make_architecture_figure.py` currently **fails** — the figure cannot
be regenerated as-is.

## Scientific framing (decided)

This is the **fundus-only** model. At inference the **only** input is the fundus image.

- **GRAPE ground-truth VF = training supervision (the loss target), NOT a runtime input.**
  Depicting it as an input arrow into the model would misrepresent the fundus-only claim.
- **Garway–Heath = a fixed anatomical prior** genuinely wired into the per-point attention
  (each of the 52 VF query points is tied to its anatomically-corresponding disc sector). This
  is a real architectural input and is drawn as one.

## Design (approved)

**Layout: inline attach** — keep the approved horizontal inference row; attach each new element
exactly where it acts.

### Core row (unchanged in meaning)
`Fundus → Disc crop → RETFound ViT-L [frozen] → Per-point attention [trained] → Predicted VF`.

### New element 1 — Garway–Heath prior cartoon (answers "how GH goes in")
- A small inset **above the decoder box**, connected by an arrow pointing **down into** the box,
  labeled **"Garway–Heath anatomical prior."**
- Content, drawn from the real `decoder/garway_heath_sectors.json` `sector_grid`
  (8 sectors): a **disc-sector wheel** (8 colored wedges) `↔` a **VF footprint** (the 52-point
  grid painted into the same 8 sector colors). Conveys: each VF point ← its disc sector.
- Fallback if cramped above one box: VF-sector-map only, captioned "disc → VF sectors".

### New element 2 — GRAPE ground-truth VF supervision (answers "how GRAPE feeds in")
- A **ground-truth VF cartoon** (the real `hvf` of the same example eye `40_OD_2.jpg`, same
  `inferno` colormap / `vmin=0, vmax=34` as Predicted VF for direct comparability) placed
  **below Predicted VF**.
- A **loss node** (`GH-weighted loss`) between Predicted VF and ground-truth VF, drawn as a
  **dashed green double-headed connector** tagged **"training only."**
- A light **dashed link** ties the input **Fundus** to this **GRAPE ground-truth VF**, labeled
  **"GRAPE fundus–VF pair"** — makes the pairing explicit (image drives prediction; VF supervises).
  Kept low-key; dropped if it clutters on render.

### Legend
Add a third key **`--- training only`** (dashed green) beside the existing *frozen (pretrained)*
and *trained on GRAPE*.

### Bottom band
Kept (Training / Evaluation summary).

## Required code fix (bundled)

Remove the dead-session `grid.json` load. Inline `mask_OD` and `valid_indices_od` (copied from
`decoder/diagnostics.py`) so the script is **self-contained, torch-free, and reproducible** from
committed files only (`config.json`, `cv_long/fold*_val.json`, `oof_cache_notta/*.npz`, the fundus
image).

## Non-goals / YAGNI

- No change to the model, metrics, or any reported number.
- Not converting this into the longitudinal-model figure (VF stays supervision, never an input).
- No new dependencies; still Agg / matplotlib only.

## Success criteria

1. `python decoder/make_architecture_figure.py` runs clean and writes the PNG (no dead path).
2. The rendered figure visibly shows (a) a GH disc↔VF sector cartoon feeding the attention box,
   and (b) a GRAPE ground-truth VF compared to the prediction via a training-only loss node.
3. Fundus-only honesty preserved: VF never appears as a model input.
4. Fonts ≥ ~12 pt, no overlaps/clipping, legend has the training-only key.
