# Garway–Heath sectoring correction

**Date:** 2026-07-15
**Trigger:** Sophia Wang flagged that the sectoring in the pipeline figure and the MAE heatmap
"is not correct at all in practice for what an actual Garway–Heath sectioning looks like."

## Root cause (traced)

`SECTOR_GRID` in `decoder/garway_heath_weighting.py` was a hand-authored **8-sector draft** — the
code labeled it *"EDITABLE SOURCE OF TRUTH (8 sectors, draft)"* / *"a Garway–Heath-style variant …
a DRAFT meant to be hand-edited."* It named sectors by **field region** and placed them naively
(superior-named sectors in superior rows), which is **not** the empirical Garway–Heath map.

Where it was used:
- **Model training loss** — `train_lora_cached.py` pulls `sector_weight_tensors()` and applies it
  as the per-point loss weight (`sector_combine='sector_only'`). The reported `p1disc` /
  `p1disc_denoise` checkpoints were trained with it (a mild, mean-1-normalized spatial reweight).
- **Per-sector MAE table** (paper Table 1).
- **Both figures** — architecture cartoon (via `config.json`) and MAE heatmap sector overlay.

**Not** used by: the model's structural **attention prior** (`build_vf_to_patch_prior`,
`training.py`) — that is a **separate retinotopic VF-grid→patch-grid Gaussian**, independent of
`SECTOR_GRID`. So the sectors never structurally wire the predictions.

## Decisions (user-approved)

1. **Adopt the canonical Garway–Heath 6-sector map** (Garway-Heath et al., *Ophthalmology* 2000):
   Temporal (T), Superotemporal (ST), Superonasal (SN), Nasal (N), Inferonasal (IN),
   Inferotemporal (IT).
2. **Figures + recompute the per-sector table on the new sectors; no retrain.** Changing
   `SECTOR_GRID` does not touch the trained checkpoints or their cached OOF predictions.

## The map (how it was obtained — authoritative, not eyeballed)

Downloaded a published color 24-2 GH sector figure (IJO 69:1825, PMC8374815, right-eye field view),
**sampled the pixel color at each of the 72 grid-cell centers**, matched each against the figure's
own sector-wheel reference RGBs, and confirmed: (a) the masked-cell pattern matches `mask_OD`
exactly, (b) all 6 sectors are used, (c) 52 points total. A re-render matched the source
cell-for-cell. Sector sizes: T 6, ST 10, SN 11, N 4, IN 7, IT 14.

Encoded on the OD 8×9 grid (0=T,1=ST,2=SN,3=N,4=IN,5=IT; `_`=masked):

```
_ _ _ 4 4 4 4 _ _      superior field (top)  → inferior-disc sectors IN/IT
_ _ 4 5 5 5 5 4 _
_ 5 5 5 5 5 5 4 3      right col (temporal field) → N (nasal disc = temporal wedge)
5 5 5 5 0 0 0 _ 3      central band → T (temporal disc = papillomacular)
2 1 1 1 0 0 0 _ 3
_ 2 1 1 1 1 1 2 3
_ _ 2 2 1 1 2 2 _      inferior field (bottom) → superior-disc sectors SN/ST
_ _ _ 2 2 2 2 _ _
```

The **structure-function crossing** (superior field ↔ inferior disc; temporal field ↔ nasal disc)
is what the draft got wrong and this corrects.

## Provenance & honesty

- `SECTOR_WEIGHTS` default to **uniform (1.0)** — the corrected map is used for anatomical
  reporting + figures; the loss no longer depends on any hand-drawn sectoring.
- The legacy 8-sector draft + weights are preserved as `LEGACY_SECTOR_GRID_DRAFT` /
  `LEGACY_SECTOR_WEIGHTS` so the reported checkpoints' exact training loss stays reproducible.
- Paper text notes: the reported checkpoint's loss used the legacy draft as a mild spatial
  reweight that does not structurally determine predictions (attention prior is retinotopic).

## Downstream changes

- `garway_heath_weighting.py` — new `SECTOR_GRID`, `N_SECTORS=6`, names, uniform weights, legacy.
- `decoder/garway_heath_sectors.json` — regenerated.
- `make_architecture_figure.py` — 6-sector colors; **fix conceptual error**: depict GH sectors as
  the (training-only) sector-weighted **loss**, and label the attention prior as *retinotopic*
  (it is not the GH map).
- `make_paper_figures.py` — repoint dead `grid.json` dependency; heatmap overlay uses new map;
  regenerate `p1disc_denoise_mae_heatmap.png`.
- Paper Table 1 (per-sector MAE) recomputed on the 6 sectors from cached OOF; docs updated.
