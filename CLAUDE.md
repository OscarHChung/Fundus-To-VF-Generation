# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project predicting Humphrey Visual Field (HVF 24-2) tests from fundus (retinal) images using a frozen RETFound encoder + trainable decoder. Presented at ARVO 2026.

## Common Commands

```bash
# Stage 1: Pre-train VF auto-decoder on UWHVF data (~29k VF tests, no images)
python decoder/pretraining.py

# Stage 2: Split GRAPE dataset into train/test (run once, seeds fixed at 42)
python decoder/separate_datasets.py

# Stage 3: Train full model (RETFound encoder frozen, decoder trained on GRAPE)
python decoder/training.py

# Run inference and save best predictions by severity category
python decoder/predict_vf_from_fundus.py \
  --fundus-file 1_OD_1.jpg \
  --output-dir decoder/best_prediction_out

# Generate per-point scatterplot for the current inference model
python decoder/generate_scatterplot.py

# Data conversion utilities
python vf_test_standardizer.py   # Convert UWHVF JSON to standardized format
python vf_test_converter.py      # Convert GRAPE Excel to JSON
python decoder/expand_GRAPE.py   # Expand GRAPE with additional image paths
```

## Architecture

### Two-stage pipeline

**Stage 1 — VF Auto-decoder Pre-training** (`decoder/pretraining.py`):
Trains a VF auto-encoder (encoder-decoder) using UWHVF's ~29k VF-only records. The decoder learns to reconstruct 52-point VF sensitivity vectors from corrupted/masked inputs. Saves `decoder/pretrained_vf_decoder.pth`.

**Stage 2 — Main Model Training** (`decoder/training.py`):
Uses the GRAPE dataset (fundus images + paired VF tests).

The current model (`PerPointVFModel`, v10.2) architecture:
1. **RETFound encoder** (frozen `mae_vit_large_patch16_dec512d8b`, ViT-Large) — encodes 224×224 fundus image into 196 patch tokens (1024-dim) + CLS token
2. **PerPointAttention** — 52 learned query vectors attend over patch tokens with anatomical priors (Gaussian distance bias toward the anatomically corresponding patch region) and learnable temperature. Returns per-point attended features.
3. **PointHead** — shared MLP per point that takes `[attended_feature ‖ cls_token]` (2048-dim) and predicts scalar sensitivity in dB
4. **CrossPointRefinement** — small MLP over all 52 predictions jointly, zero-initialized with learned gate, adds spatial coherence

An older simpler model (`MultiImageModel`) in `decoder/predict_vf_from_fundus.py` uses CLS-token → MLP projection instead of per-point attention; used for backward-compatible inference.

### VF grid conventions
- HVF 24-2 has 52 valid test points out of a 8×9 grid (72 cells). Positions with value ≥ 99.0 are masked/invalid.
- `mask_OD` defines which cells are valid for the right eye; `mask_OS = np.fliplr(mask_OD)` for left eye.
- `valid_indices_od` / `valid_indices_os` are flat indices into the 72-element flattened grid.
- The model always predicts 52 values in OD canonical order; laterality flipping is handled inside `PerPointAttention` by flipping the anatomical prior horizontally.
- VF data stored as 8×9 Python lists in JSON under key `"hvf"`.

## Data

```
data/
  fundus/
    grape_fundus_images/   # GRAPE fundus images (not in git, large)
  vf_tests/
    uwhvf_vf_tests.json             # Raw UWHVF data
    uwhvf_vf_tests_standardized.json # Standardized for pretraining
    grape_data.xlsx                  # Raw GRAPE spreadsheet
    grape_new_vf_tests.json          # Processed GRAPE (all eyes)
    grape_train.json                 # 80% split (seed=42)
    grape_test.json                  # 20% split (seed=42)
```

JSON record format (GRAPE):
```json
{
  "PatientID": 1,
  "FundusImage": ["1_OD_1.jpg", "1_OD_2.jpg"],  // list for multi-image
  "Laterality": "OD",
  "hvf": [[...8 rows of 9 values...]]            // 100.0 = masked point
}
```

## Key Implementation Details

- **MPS (Apple Silicon)**: All training scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1` and detect `mps` > `cuda` > `cpu`. Training won't work on MPS without this env var because some ops fall back to CPU.
- **Severity weighting**: Training uses `WeightedRandomSampler` to oversample eyes with more severe VF loss (lower mean dB).
- **TTA**: Test-time augmentation uses rotations `[-5°, 0°, 5°]` during validation; predictions are averaged.
- **Loss**: Weighted Huber loss + per-eye CCC loss (starts epoch 5) + variance penalty (starts epoch 8) + attention entropy bonus.
- **Model checkpoints**: `decoder/best_multi_image_model.pth` (training state), `decoder/inference_model.pth` (minimal inference state). `predict_vf_from_fundus.py` tries multiple candidates in priority order.
- **Metrics target**: Baseline ~3.74 dB MAE. v10.2 trains toward sub-4 dB.
