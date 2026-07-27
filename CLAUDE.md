# CLAUDE.md

Guidance for working in this repository.

## Project Overview

Predicts the Humphrey Visual Field (HVF 24-2, 52 points, dB) for a glaucoma eye from a fundus
photo, using a frozen RETFound encoder + a trainable per-point attention decoder. For ARVO 2026.

There are **two tracks**:

1. **Fundus-only (the paper — this is what "the model" means by default).** Reported model is
   `p1disc_denoise`: disc-crop-as-sole-view + LoRA + severity head, trained with Theil-Sen
   per-point trajectory target-denoising (train-only; validation is always scored against the raw
   HVF). Pooled OOF MAE **4.102 dB**, 95% CI [3.789, 4.455], r 0.692, raw slope 0.549 /
   calibrated 0.645. Manuscript + figures in `paper/`.
2. **Longitudinal (reported separately, secondary).** Predicts a visit's VF from
   `fundus + the eye's prior VF + the inter-test interval`. Pooled MAE 3.75, calibrated slope 0.72.
   Code: `decoder/longitudinal_*.py`, `decoder/train_longitudinal.py`, `decoder/eval_*_longitudinal.py`.

### Honest-framing guardrails (do not violate in any output)
1. **Do not claim native sub-4.0 dB** — the 95% CI upper bound is 4.455.
2. **Raw slope (0.549) does not clear 0.60.** Only calibrated (0.645) and disattenuated (0.650) do.
3. **PAPILA dB MAE (~26 dB) is meaningless** (24-2 vs 30-2 scale mismatch) — report correlation
   only (overall r 0.753, glaucoma subgroup r 0.755).
4. **The reliable head-to-head claim is composition-adjusted** (ours 3.27–3.65 vs TDV-Net's 3.91),
   not a native pooled-MAE win — native 4.10 vs 3.91 is a case-mix inversion.

Full derivation and CIs: `paper/headline_results.md`.

## Repo layout

```
data/          GRAPE + UWHVF + PAPILA inputs (images & large JSON not in git)
encoder/       frozen RETFound ViT-L weights + vendored RETFound_MAE
decoder/       all model / training / eval / figure code  (paths resolve relative to this dir)
  results/     cv_long/ (frozen folds), auto/ (per-tag results + OOF caches), auto/logs/
  tests/       pytest suite — bare `pytest` from the repo root runs it
scripts/       data-prep entry points + train_folds.sh
paper/         manuscript, headline results, figures/, tables/
docs/          specs/ (designs), plans/, history/ (iterations + improvement log)
```

**Path convention:** every script under `decoder/` resolves data via
`os.path.dirname(__file__)/results/...`. Keep `decoder/` as the code root and `results/` under it,
or those paths break.

## Replication — the reported fundus-only model (4.102)

Run from the repo root. Training detects `mps` > `cuda` > `cpu`.

```bash
# --- Reproduce every published number & figure from the frozen OOF cache (seconds, no torch) ---
python decoder/composition_report.py --tag p1disc_denoise   # 4.102 + composition-adjusted 3.27-3.65
python decoder/scatter_from_oof.py  --tag p1disc_denoise   # Figure 4 scatter
python decoder/make_paper_figures.py                        # Figure 3 (heatmap) + Figure 5 (examples)
python decoder/make_architecture_figure.py                  # Figure 2 (pipeline schematic)
python decoder/td_report.py && python decoder/td_report_calibrated.py   # total-deviation re-expression

# --- Retrain from scratch ---
python scripts/build_longitudinal_grape.py    # -> data/vf_tests/grape_longitudinal.json (631 records)
python decoder/diagnostics.py split-long      # -> decoder/results/cv_long/ (per-patient 5-fold)
# 1. fundus-only base — also the per-fold warm-start for the reported model
python decoder/run_cv.py --tag long_global --cv-dir decoder/results/cv_long --epochs 60 -- \
  --weighting garway_heath --sector-combine sector_only --reweight value \
  --lr 8e-4 --dropout 0.2 --weight-decay 0.005 --global-head
# 2. the reported model (disc-crop + LoRA + severity head + denoised targets), 5 folds, serial
./scripts/train_folds.sh
# 3. score
python decoder/eval_oof_cached.py --tag p1disc_denoise --no-tta
python decoder/paired_decision.py --tag p1disc_denoise --ref p1disc
```

**The frozen OOF caches (`decoder/results/auto/oof_cache_notta/*.npz`, ~160 KB each) are the
source of truth for every paper number.** They are torch-free and regenerate all figures and
tables. Model checkpoints are ~500 MB–1.2 GB each and are *not* in git; only the reported model's
5 folds (`p1disc_denoise_f{0..4}_best.pth`) are kept on disk.

### Longitudinal track (secondary)
```bash
python decoder/vf_autoencoder.py                              # -> decoder/pretrained_vf_ae.pth
python decoder/train_longitudinal.py --tag long_prior --epochs 16   # needs long_global warm-start
python decoder/eval_strata_longitudinal.py                    # with-prior 3.185 vs visit-1 4.55
python decoder/eval_blend_longitudinal.py                     # model == deterministic blend (3.752)
python decoder/make_scatterplot.py
```

## Architecture

`decoder/training.py` — `PerPointVFModel`. All 52-vectors are in OD/OS query order.

- **Frozen RETFound** (`mae_vit_large_patch16_dec512d8b`, ViT-L) — 224×224 fundus -> 196 patch
  tokens + CLS. `_encode` runs it WITHOUT the MAE random patch shuffle (a fixed correctness bug).
  The reported model adds trainable LoRA adapters (r8) over the last 8 blocks.
- **PerPointAttention** — 52 query vectors attend over patches with a mandatory **retinotopic**
  (VF-grid→patch) Gaussian prior (`build_vf_to_patch_prior`) + a zero-init global-spatial head +
  cross-point refinement. NOTE: this attention prior is retinotopic, NOT the Garway–Heath sector
  map. Garway–Heath enters only as the (training-only) sector-weighted loss + per-sector reporting,
  using the canonical 6-sector GH map in `decoder/garway_heath_sectors.json`
  (`garway_heath_weighting.py`); the reported checkpoints were trained with a preliminary 8-sector
  draft, preserved as `LEGACY_SECTOR_GRID_DRAFT`. See
  `docs/specs/2026-07-15-garway-heath-sectoring-fix-design.md`.
- **Longitudinal only** — `decoder/longitudinal_model.py` (`LongitudinalVFModel`) adds a frozen
  VF-manifold autoencoder over the eye's prior VF and predicts
  `pred = prior_field + gate(interval)·delta` with the delta head zero-initialised (starts exactly
  at persistence). Key honest finding: the learned delta adds ~nothing — the follow-up stratum is
  bounded by the 2.76 dB test-retest noise floor. See `docs/history/iterations.md`.

### VF grid conventions
- 24-2 = 52 valid points of an 8×9 grid (72 cells); value ≥ 99.0 = masked. `mask_OD` is the right
  eye; `mask_OS = fliplr(mask_OD)`. `valid_indices_od/os` index the flattened 72-grid. The model
  predicts 52 values in OD canonical (query) order; laterality flip is inside `PerPointAttention`.
- VF stored as 8×9 lists under `"hvf"`. Record schema:
  `{"PatientID":int, "Laterality":"OD"|"OS", "VisitNumber":int, "FundusImage":[file], "hvf":[8×9],
    "interval_years":float, "has_prior":bool, "prior_hvf":[8×9]|null, "delta_t":float,
    "hvf_denoised":[8×9]}`.

## Key Implementation Details
- **MPS:** scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1`. This box OOM-kills concurrent torch
  processes — train folds serially (`scripts/train_folds.sh` does).
- **Eval:** honest = per-patient 5-fold CV (`decoder/diagnostics.py` `build_patient_folds` +
  `pooled_metrics` + `stratified_report`), eye-disjoint and frozen in `decoder/results/cv_long`.
  Never reseed it. The longitudinal track additionally enforces causal ordering: a target visit
  uses only that eye's EARLIER visits as the prior input.
- **Promotion gate (§6.5):** a new tag replaces the reported model only on ΔMAE ≤ −0.12 dB with
  CI upper < 0 and ≥4/5 folds better. `p1disc_denoise` was promoted on a severe-band/slope win at
  pooled-neutral MAE, explicitly *against* this gate — see `paper/headline_results.md`.
- **TTA:** rotations `[-5°, 0°, 5°]` at val, averaged. The reported numbers are **no-TTA**
  (`--no-tta` → `oof_cache_notta/`). **Loss:** GH-weighted Huber + per-eye CCC + variance.
- **Checkpoints bundle the frozen RETFound weights** (hence their size); `eval_ckpt.load_model`
  reloads the encoder fresh, so only the small decoder tensors in them matter.
- **Closed levers — do not re-attempt** (all failed pre-registered gates): high-res disc @384/448,
  PAPILA-MD severity co-train, RetiZero/RETFound-Green ensembles, generic DINOv2, disc+full
  fusion, ordinal CORAL, RNFL metadata, disc geometric jitter. Details in
  `docs/specs/fundus_only_ceiling_design.md` and `docs/history/iterations.md`.
