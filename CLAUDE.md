# CLAUDE.md

Guidance for working in this repository.

## Project Overview

Predicts the Humphrey Visual Field (HVF 24-2, 52 points, dB) for a glaucoma eye from a fundus
photo, using a frozen RETFound encoder + a trainable decoder with Garway-Heath anatomical
sectoring. The current best model is **longitudinal**: it predicts a visit's VF from
`fundus + the eye's prior VF + the inter-test interval`. For ARVO 2026.

**Honest headline result (leak-free per-patient 5-fold CV over 631 records):** pooled pointwise
MAE **3.75 dB** (raw), calibrated line-of-best-fit slope **0.72**, severe-band MAE **6.06**.
Fundus-only (no prior VF) baseline on the same eval = **4.29**. Persistence baseline = 3.71.

## Replication pipeline (reproduces the 3.75)

Run from the repo root. All training detects `mps` > `cuda` > `cpu`.

```bash
# 1. Build the longitudinal dataset from the GRAPE Excel "Follow-up" sheet
#    -> data/vf_tests/grape_longitudinal.json (631 fundus->same-visit-VF pairs, each with its
#       most-recent causal prior VF + interval_years)
python build_longitudinal_grape.py

# 2. Per-patient stratified 5-fold split (eye-disjoint, leak-free) -> decoder/results/cv_long/
python decoder/diagnostics.py split-long

# 3. Pretrain the VF-manifold autoencoder on UWHVF (~29k VF-only) -> decoder/pretrained_vf_ae.pth
python decoder/vf_autoencoder.py

# 4. Train the fundus-only base (warm-start for the longitudinal model + the 4.29 baseline)
#    -> decoder/results/auto/long_global_f{0..4}_best.pth
python decoder/run_cv.py --tag long_global --cv-dir decoder/results/cv_long --epochs 60 -- \
  --weighting garway_heath --sector-combine sector_only --reweight value \
  --lr 8e-4 --dropout 0.2 --weight-decay 0.005 --global-head

# 5. Train the longitudinal model (warm-starts the frozen fundus branch from step 4)
#    -> decoder/results/auto/long_prior_f{0..4}_best.pth + long_prior_cv.{log,json}
python decoder/train_longitudinal.py --tag long_prior --epochs 16

# 6. Deliverables / proof
python decoder/eval_strata_longitudinal.py     # per-stratum: with-prior 3.185 vs visit-1 4.55
python decoder/eval_blend_longitudinal.py       # shows model == deterministic blend (3.752)
python decoder/make_scatterplot.py              # decoder/results/auto/longitudinal_scatter.png

# Data utilities (only needed to regenerate inputs from raw)
python vf_test_standardizer.py   # raw UWHVF JSON -> standardized (autoencoder training data)
python vf_test_converter.py      # GRAPE Excel Baseline sheet -> grape_new_vf_tests.json (mapping ref)
```

## Architecture (longitudinal model)

`decoder/longitudinal_model.py` — `LongitudinalVFModel`, a subclass of `PerPointVFModel`
(`decoder/training.py`). All 52-vectors are in OD/OS query order.

- **Frozen RETFound** (`mae_vit_large_patch16_dec512d8b`, ViT-L) — 224×224 fundus -> 196 patch
  tokens + CLS. `_encode` runs it WITHOUT the MAE random patch shuffle (a fixed correctness bug).
- **PerPointAttention** — 52 query vectors attend over patches with a **Garway-Heath** anatomical
  Gaussian prior (mandatory). + a zero-init global-spatial head + cross-point refinement.
- **VF-manifold autoencoder** (`decoder/vf_autoencoder.py`, frozen) — encodes the eye's prior VF
  (mean-imputed + mask channel, non-zeroed) into a 64-d latent; infills missing prior points.
- **Prediction:** for prior-bearing records, `pred = prior_field + gate(interval)·delta`, with the
  delta head **zero-initialised** so it starts exactly at persistence (3.185) and only earns
  corrections. For first-visit records (no prior), `pred =` the warm-started frozen fundus branch.
- **Key honest finding:** the learned delta adds ~nothing (follow-up stratum is bounded by the
  2.76 dB test-retest noise floor); the model is equivalent to "persistence on follow-ups +
  fundus on first-visits." See `decoder/results/auto/iterations.md`.

### VF grid conventions
- 24-2 = 52 valid points of an 8×9 grid (72 cells); value ≥ 99.0 = masked. `mask_OD` is the right
  eye; `mask_OS = fliplr(mask_OD)`. `valid_indices_od/os` index the flattened 72-grid. The model
  predicts 52 values in OD canonical (query) order; laterality flip is inside `PerPointAttention`.
- VF stored as 8×9 lists under `"hvf"`. Record schema:
  `{"PatientID":int, "Laterality":"OD"|"OS", "VisitNumber":int, "FundusImage":[file], "hvf":[8×9],
    "interval_years":float, "has_prior":bool, "prior_hvf":[8×9]|null, "delta_t":float}`.

## Data
```
data/
  fundus/grape_fundus_images/        # GRAPE fundus images (not in git, large)
  vf_tests/
    grape_data.xlsx                  # GRAPE source (Baseline + Follow-up sheets)
    grape_longitudinal.json          # 631 per-visit pairs + causal prior VF  (built, step 1)
    grape_long_train/val.json        # fast dev split (built, step 2)
    grape_new_vf_tests.json          # 263 baseline eyes (mapping reference / test)
    uwhvf_vf_tests.json              # raw UWHVF (~29k VF-only)
    uwhvf_vf_tests_standardized.json # standardized UWHVF (autoencoder training data)
decoder/results/
  cv_long/                           # per-patient 5-fold splits (the eval folds)
  auto/long_prior_f*_best.pth        # the final longitudinal model (5 folds)
  auto/long_global_f*_best.pth       # fundus-only base / warm-start (5 folds)
  auto/long_prior_cv.{log,json}      # the 3.75 result (proof)
  auto/longitudinal_scatter.png      # deliverable scatterplot
  auto/iterations.md                 # full Sessions 1-4 narrative (what we tried + why)
  champion/longitudinal_champion.json
  garway_heath/config.json           # GH sector map (read by garway_heath_weighting.py)
```

## Key Implementation Details
- **MPS:** scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Eval:** honest = per-patient 5-fold CV (`decoder/diagnostics.py` `build_patient_folds` +
  `pooled_metrics` + `stratified_report`). Causal ordering: a target visit uses only that eye's
  EARLIER visits as the prior input (leak-free; standard longitudinal forecasting).
- **TTA:** rotations `[-5°, 0°, 5°]` at val, averaged. **Loss:** GH-weighted Huber + per-eye CCC
  + variance + a small delta regularizer.
- **Checkpoints are ~1.2 GB** (they bundle the frozen RETFound weights); `eval_ckpt.load_model`
  reloads the encoder fresh, so only the small decoder/longitudinal tensors matter.
```
