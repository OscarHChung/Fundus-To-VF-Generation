# Garway–Heath sector-weighted loss — design / spec

Status: **draft for review** (the sector grid and weights are deliberately
editable constants — see §2/§4). Opt-in; the baseline is untouched.

Companion code:
- `decoder/garway_heath_weighting.py` — sector map, weights, deep-floor loss
  shaping config, helpers, sanity checks, and the evaluation/figure pipeline.
- `decoder/training.py` — `--weighting garway_heath` hook (minimal, reversible),
  deep-floor loss shaping inside `compute_loss`, plus live training diagnostics
  and self-stopping.
- `decoder/generate_scatterplot.py` — refactored to expose `plot_vf_scatter(...)`
  so the new path renders an identical-style Figure 4 (behavior of its own
  `main()` is unchanged).

---

## 1. Motivation

The fundus→VF model predicts 52 Humphrey 24-2 threshold-sensitivity values (dB)
per eye. Like other fundus→VF models, it **underestimates loss in the
peripheral / arcuate field** and regresses toward the mean (low slope): central
points are easy, but the superonasal / inferonasal / nasal-step and arcuate
regions — exactly where glaucomatous damage concentrates — are predicted worst.

This is visible in the held-out set (current `inference_model.pth`, 47 eyes,
3-rotation TTA):

| sector            | MAE (dB) |
|-------------------|---------:|
| Inferior-arcuate  | 3.72 |
| Temporal-wedge    | 3.93 |
| Central-Superior  | 4.02 |
| Superior-arcuate  | 4.09 |
| Central-Inferior  | 4.22 |
| Superonasal       | 4.56 |
| Nasal-periphery   | 4.62 |
| Inferonasal       | 4.65 |

(overall plain MAE 4.21 dB, slope 0.35). The nasal/peripheral sectors are
~0.7–0.9 dB worse than the best sectors.

**Idea:** add a *spatial* (sector) weight to the per-point loss that penalises
errors in the worse, clinically-critical sectors more, pushing the model to
stop sacrificing the periphery. This is **distinct from** the existing
value-based (severity) weight in `compute_loss`
(`1 + WEIGHT_SCALE·(MAX_DB−t)/MAX_DB`) and from the `WeightedRandomSampler`;
it is layered in as an extra multiplicative factor and is fully configurable.

### 1.1 What the first GH run actually did (and what it broke)

The first sector-weighted training run (`--weighting garway_heath`, 47 held-out
GRAPE eyes, 3-rotation TTA) vs. the value-only baseline:

| metric          | baseline | GH run-1 | Δ |
|-----------------|---------:|---------:|---:|
| plain MAE (dB)  | 4.228 | 4.040 | **−0.188** |
| slope           | 0.346 | 0.410 | +0.064 |
| Pearson r       | 0.563 | 0.598 | +0.035 |
| R²              | 0.317 | 0.357 | +0.040 |

So the headline metrics all improved: lower MAE, higher slope (less
regression-to-mean), better correlation. **7 of 8 sectors improved**;
Temporal-wedge improved most (−0.426); Superior-arcuate was essentially flat
(+0.025) — the laggard.

But two regressions showed up that motivated this round of changes:

1. **The very deepest *points* got worse.** Stratified by GT sensitivity,
   GT 0–10 dB MAE rose +0.293 dB (n=160) and GT 10–20 dB rose +0.142. Sector
   weights are *spatial*: they cannot target a value band like 0–10 dB, so they
   can lift the periphery on average while still letting the deepest scotoma
   points drift.
2. **Global bias drifted positive,** +0.190 → +0.730 dB. A positive bias means
   the model **over-predicts sensitivity** = it **under-deepens** = it
   underestimates scotoma depth — the exact failure mode this project is trying
   to fix.

These two are the same root problem on the value axis, and they motivate the
**deep-floor loss shaping** (§4.1) and the **per-sector re-tune** (§4): bump the
laggard Superior-arcuate hardest, and add a value-axis rescue for deep points.

### 1.2 Run 2 — Failure analysis and corrected design

The second GH training run was launched with `--floor-boost 2.0 --overpred-penalty 0.6` to
address run-1's positive bias (+0.73 dB). It auto-stopped at epoch 3 with a
"SLOPE COLLAPSE" diagnostic after approximately 45 gradient steps. Post-mortem
identified **three bugs** in the diagnostic / hyperparameter design — not a real
slope collapse.

**Bug 1 — Slope collapse check had no minimum-epoch guard.**
`diagnose_training()` checked `len(slope) >= 3 and all(s < 0.08 for s in slope[-3:])`. With
`epoch <= 3` forcing validation at epochs 1, 2, 3, the val slope is naturally
0.001–0.013 for an uninitialized model. The check tripped after exactly 3
epochs of data. This is initialization, not collapse.

**Bug 2 — All diagnostic warnings had no minimum-check guard.**
The bias soft-warning fired at epoch 1 with bias = −3.92 dB and attributed it
to `--overpred-penalty`/`--floor-boost`. But at epoch 1 the model predicts
approximately `PROJ_INIT_BIAS ≈ 18 dB` for all points while mean true
sensitivity is approximately 22 dB, giving a natural initialization bias of
approximately −4 dB. The floor_boost had had no time to cause anything.

**Bug 3 — floor_boost=2.0 and overpred_penalty=0.6 are too aggressive for their intent.**
Run-1 GH had +0.73 dB bias at convergence — a real but modest over-prediction.
Values of 2.0 and 0.6 were chosen to push hard in the other direction, but they
are large enough to reverse the direction entirely once the model warms up.

**Corrected design values** (updated in §4.1 and in `garway_heath_weighting.py`
module defaults):

| constant | run-2 (failed) | corrected |
|---|---:|---:|
| FLOOR_BOOST | 2.0 | **0.5** |
| OVERPRED_PENALTY | 0.6 | **0.15** |

Conservative enough to nudge the positive bias without overcorrecting.

**Corrected diagnostic guard thresholds** (updated in §8 and in
`training.py diagnose_training()`):

| guard | before | after | rationale |
|---|---|---|---|
| Slope collapse minimum | `len(slope) >= 3` | `len(slope) >= 8` | ≥16 epochs after the val gate opens; model has time to actually learn |
| Bias warning minimum | none | `len(bias) >= 5` | skip initialization transients |
| Slope warning minimum | none | `len(slope) >= 5` | skip initialization transients |
| Rising-floor warning minimum | `len(floor) >= 3` | `len(floor) >= 5` | reduce false positives early in training |

---

## 2. The 52-point → sector mapping

52 valid points live in an 8×9 = 72 grid masked by `mask_OD`
(`valid_indices_od = [i for i,v in enumerate(mask_OD.flatten()) if v]`, length 52).
The blind spot is temporal (masked at col 7, rows 3 & 4 for OD).

`SECTOR_GRID` is the **single source of truth**, defined once on the OD
(right-eye) grid as an editable 8×9 array of sector ids (`-1` = masked). It is a
Garway–Heath-*style* **8-section** variant (canonical Garway-Heath, Ophthalmology
2000, uses 6 disc sectors; here we split central into superior/inferior — the
macular vulnerability zones — for finer control). **This grid is a draft and is
meant to be hand-edited (§4).** Editing the ids is all that's required to
re-sector; weights, OS mirroring, metrics and figures follow automatically, and
`selftest()` re-verifies coverage/laterality.

```
OD (right-eye) view, superior field on top, low col = nasal:
 .  .  .  2  2  3  3  .  .
 .  .  2  2  3  3  3  7  .
 .  6  2  2  0  3  3  7  7
 6  6  2  0  0  0  3  .  7
 6  6  4  1  1  1  5  .  7
 .  6  4  4  1  5  5  7  7
 .  .  4  4  5  5  5  7  .
 .  .  .  4  4  5  5  .  .
```

| id | name              | size | rationale |
|---:|-------------------|-----:|-----------|
| 0  | Central-Superior  | 4 | paracentral, superior of fixation |
| 1  | Central-Inferior  | 4 | paracentral, inferior of fixation |
| 2  | Superonasal       | 7 | superior nasal field |
| 3  | Superior-arcuate  | 8 | superotemporal Bjerrum/arcuate zone |
| 4  | Inferonasal       | 7 | inferior nasal field |
| 5  | Inferior-arcuate  | 8 | inferotemporal Bjerrum/arcuate zone |
| 6  | Nasal-periphery   | 6 | nasal step zone |
| 7  | Temporal-wedge    | 8 | far temporal periphery |

All 52 points are covered exactly once; the 20 masked cells are excluded
(asserted at import and in `selftest`). `SECTOR_MAP` (query-index → sector id,
OD reference order) is derived for convenience.

---

## 3. OS mirroring rule (the laterality subtlety)

The model outputs `pred_52` in a fixed **query order**; for OD the target is
gathered with `valid_indices_od`, for OS with `valid_indices_os`. **There are two
incompatible OS conventions in this repo:**

- `training.py` & `generate_scatterplot.py`: `valid_indices_os` from
  `fliplr(mask_OD)` — **this is the one `compute_loss` uses to index
  `pred_52`/`target_52`.**
- `predict_vf_from_fundus.py` & `visualize_MAE_heatmap.py`:
  `reversed(valid_indices_od)` — a different ordering (180° vs. mirror).

These genuinely differ (the blind-spot column is asymmetric). The sector
weighting **follows `training.py`'s `fliplr` convention** so the per-query weight
lines up with how the loss indexes points. Concretely:

```
SECTOR_ID_OD[k] = SECTOR_GRID.flat[valid_indices_od[k]]
SECTOR_ID_OS[k] = fliplr(SECTOR_GRID).flat[valid_indices_os[k]]   # fliplr both
```

Subtle but important: because of the blind-spot asymmetry, the per-query sector
**vectors differ between eyes** (`SECTOR_ID_OD != SECTOR_ID_OS`), yet the
**displayed** sector maps mirror exactly. The laterality round-trip test (§5.4)
asserts the display consistency; `selftest` also prints that the per-query
vectors differ, so it isn't mistaken for a bug.

`vector_to_grid_query(vec_52, eye)` in the module places a query-order vector
into an 8×9 display grid using this same `fliplr` convention — **fixing** the
inconsistency in `visualize_MAE_heatmap.py`/`predict_vf_from_fundus.py`.

**Target:** raw threshold sensitivity in dB (clamped 0–35), consistent with the
rest of the codebase — **not** total deviation.

---

## 4. Weighting scheme & config

Per-sector **raw** weights (editable, in `SECTOR_WEIGHTS`). Re-tuned after
run-1 (§1.1): the laggard Superior-arcuate gets the biggest bump, the arcuate /
nasal / temporal zones are nudged up, and the spared centre is nudged toward 1.

```
Central-Superior 0.75   Superonasal      1.10   Inferonasal      1.15   Nasal-periphery 1.35
Central-Inferior 0.75   Superior-arcuate 1.55   Inferior-arcuate 1.50   Temporal-wedge  1.25
```

| id | sector            | run-1 | now | rationale |
|---:|-------------------|------:|----:|-----------|
| 0  | Central-Superior  | 0.70 | 0.75 | spared centre, slight de-emphasis only |
| 1  | Central-Inferior  | 0.70 | 0.75 | "" |
| 2  | Superonasal       | 1.10 | 1.10 | unchanged |
| 3  | Superior-arcuate  | 1.40 | **1.55** | lagged most in run-1 (+0.025) → biggest bump |
| 4  | Inferonasal       | 1.10 | 1.15 | nasal field |
| 5  | Inferior-arcuate  | 1.40 | 1.50 | Bjerrum zone |
| 6  | Nasal-periphery   | 1.30 | 1.35 | nasal step |
| 7  | Temporal-wedge    | 1.20 | 1.25 | far periphery (already best mover; small nudge) |

(Still normalized to mean 1.0 over the 52 valid points, so weighted-MAE stays on
the dB scale — see below.)

- **Normalization.** `sector_weight_vector(eye)` builds the length-52 query-order
  weight vector and divides by its mean, so weights **average to 1.0 over the 52
  valid points**. This keeps both the training loss and the reported weighted-MAE
  on the dB scale. (Both eyes share the same multiset of sector ids → identical
  normaliser.)
- **Combine mode** (`--sector-combine`, default `both`): how the new sector
  weight interacts with the existing value-based weight inside `compute_loss`:
  - `both` → `value_w × sector_w` (keep severity emphasis + add spatial)
  - `sector_only` → replace value weight with sector weight (clean ablation)
  - `value_only` → exactly the baseline (sector weighting off)
- **Where it applies.** Only the per-point Huber term is multiplied by the
  weight. The returned/validation **MAE stays unweighted**, so val MAE remains
  apples-to-apples with the baseline. CCC / variance / attention-entropy terms
  are unchanged.
- **Backward compatible.** `compute_loss(..., sector_weights=None, deep_cfg=None)`
  (the defaults) is byte-for-byte the original computation.

### 4.1 Deep-floor loss shaping (the value-axis rescue)

Sector weights are *spatial* and cannot target a value band, so they can't fix
the run-1 deep-point regression or the positive bias (§1.1). This adds a
**value-axis** rescue, defined as editable constants in
`garway_heath_weighting.py` (`FLOOR_DB=12.0`, `FLOOR_BOOST=0.5`,
`OVERPRED_PENALTY=0.15`) and surfaced via `deep_loss_config(floor_db=None,
floor_boost=None, overpred_penalty=None)`, which returns a plain dict (no
`import training` — preserves the no-circular-import design; `None` args fall back
to the module defaults so CLI flags can override individual fields).

It is active **only** under `--weighting garway_heath`. Setting `floor_boost=0`
and `overpred_penalty=0` recovers the pure sector-weighted run from before. Both
mechanisms live inside `training.compute_loss`, gated on `deep_cfg is not None`,
so the baseline path stays byte-for-byte unchanged:

- **(a) Floor boost** — an **additive** extra weight on top of the existing
  value-based severity weight, equal to
  `floor_boost · clamp(floor_db − gt, min=0) / floor_db`. It peaks at GT=0 and
  tapers linearly to 0 at `FLOOR_DB`. With the defaults the deepest points get
  ~5× weight vs. healthy points (was ~3× from the severity weight alone). It
  folds into `value_w`, so it participates in the value×sector combine.
- **(b) Asymmetric over-prediction penalty** — the per-point Huber is multiplied
  by `(1 + overpred_penalty)` **only** where the model over-predicts a deep point
  (`pred > gt AND gt < floor_db`). Being asymmetric, it pushes deep predictions
  **down**, directly attacking the positive bias / depth underestimation.

The returned/validation **MAE is still unweighted**, so val MAE stays
apples-to-apples with the baseline.

### 4.2 Plumbing & CLI overrides

- `compute_loss` gained a trailing `deep_cfg=None` param (backward compatible).
- `train()` gained a trailing `deep_cfg=None` param; under
  `--weighting garway_heath` it defaults to `deep_loss_config()`, and the
  resolved values are written into `results/garway_heath/config.json` under a
  `"deep_loss"` key. `resolved_config()` in the GH module also includes a
  `"deep_loss"` block.
- New CLI overrides on `decoder/training.py` (GH only): `--floor-boost`,
  `--overpred-penalty`, `--floor-db`. Each overrides the corresponding
  `deep_loss_config` field so the floor shaping is tunable without editing code,
  e.g.
  `python decoder/training.py --weighting garway_heath --floor-boost 3.0 --overpred-penalty 0.9`.

**To confirm / tune (§4 of the task):** the 8-sector `SECTOR_GRID` layout, the
`SECTOR_WEIGHTS` values, and the deep-floor constants are drafts. Edit the
constants in `garway_heath_weighting.py` (or pass the CLI overrides); the
uniform-weight invariant, deep-floor monotonicity check, and parity checks guard
against mistakes.

---

## 5. Sanity checks (`python decoder/garway_heath_weighting.py --selftest`)

1. **Coverage** — all 52 points map to exactly one sector; no unmapped /
   double-mapped; the 20 masked cells excluded.
2. **Uniform-weight invariant** — with equal weights, weighted MAE == plain MAE.
3. **Normalization** — OD & OS normalized weight vectors average to 1.000.
4. **Laterality round-trip** — `sector_weight_vector('OD')` and `('OS')`, placed
   back via `vector_to_grid_query(..., eye)`, land in mirrored positions
   (identical display grid) for both sector ids and weights. Guards the OS bug.
   (4b prints that the per-query vectors differ — expected.)
5. **Tensor/numpy parity** — torch weights match numpy weights.
6. **Deep-floor weight shaping** — the additive floor weight is non-negative,
   monotonically non-increasing in GT, peaks at GT=0 (= `FLOOR_BOOST`), and is
   exactly 0 at/above `FLOOR_DB`. Mirrors the formula used in
   `training.compute_loss`, so the rescue can't silently invert or leak above
   the floor.

Plus **baseline parity** (run the eval path on the baseline checkpoint): the new
pipeline reproduces `generate_scatterplot.py`'s MAE/slope/R² on the same model
(see §6 outputs) and the uniform/`value_only` configs reproduce baseline training.

---

## 6. Model-class fix for the heatmap (Figure 3)

`visualize_MAE_heatmap.py` builds the **old** `MultiImageModel` (projection +
`VFAutoDecoder`) and loads weights with `strict=False`, so it does **not** reflect
the live `PerPointVFModel`. The new heatmap (`_save_heatmap`) drives the **same
`PerPointVFModel`** as training/scatterplot, reuses `visualize_MAE_heatmap.plot_single`
for styling, and uses the corrected `vector_to_grid_query` (fliplr/`training.py`
convention). It optionally overlays sector boundaries. All three figures thus
come from one model and one OS convention. (Note: the new heatmap uses the same
3-rotation TTA as training/scatterplot, vs. the old script's 4-aug TTA.)

---

## 7. Outputs → `decoder/results/garway_heath/`

- `metrics.json` / `metrics.csv` — overall **plain MAE** (apples-to-apples vs.
  baseline) and **sector-weighted MAE**, plus Pearson r, R², slope, signed bias,
  RMSE, per-eye corr (computed with the existing `training.py` helpers), the
  per-sector MAE + point-count table, and the **severity-stratified MAE** (goal 1):
  per-point GT-sensitivity-bin MAE (`severity_pointwise`, incl. a `deep(<16)` vs.
  `preserved(>=16)` split) and per-eye severity-group MAE (`severity_per_eye`:
  severe `<16` / moderate `16–26` / early `>26` dB mean sensitivity).
- `vf_scatterplot_garway_heath.png` — Figure 4, identical style (same `vf_err`
  colormap, identity line, fitted line, error-colored points, info box, limits,
  aspect) via the shared `plot_vf_scatter`.
- `bilateral_MAE_garway_heath.png` — Figure 3, bilateral 4-panel (OD/OS MAE +
  OD/OS GT), `PerPointVFModel`-driven, optional sector-boundary overlay.
- `comparison_baseline_vs_garway_heath.csv` — baseline vs. new side-by-side:
  overall plain/weighted MAE, r, R², slope, bias, RMSE, per-eye corr, per-sector
  MAEs, **and the severity strata** (GT-bin and per-eye-group MAE), each with
  deltas (which sectors/severities improved, and the central cost).
- `config.json` — resolved sector grid + weights + the `deep_loss` block
  (`floor_db` / `floor_boost` / `overpred_penalty`) + hyperparameters + the
  evaluated checkpoint, so a run is reproducible.
- `inference_model_gh.pth` / `best_model_gh.pth` — the sector-weighted checkpoint
  (baseline `inference_model.pth` is left intact).

---

## 8. Live training diagnostics & self-stopping (`training.py`, GH-aware)

To make a GH run *legible while it trains* — and to abort runs that have already
gone wrong — `training.py` gained instrumentation aimed at the run-1 failure
modes (positive bias, deep-point regression, slope collapse):

- **Richer eval.** `evaluate(model, loader, detailed=True)` now also returns a
  dict with `bias`, `rmse`, `floor_mae` (GT 0–10 dB — the **goal-1** metric),
  `floor_n`, `deep_mae` (GT < 16), `deep_n`. Helper `_flat_severity_stats` does
  the flat masked extraction.
- **"Severe watch" line.** Each validation check prints `bias`, `floor(0-10)`,
  `deep(<16)` with `[targets: bias→0, floor↓, deep↓]`, so depth calibration is
  visible every check instead of only at the end.
- **`diagnose_training(history)`** — a pure function returning
  `(should_stop, reason, warnings)`. Hard auto-stops, each with a plain-English
  reason, on: (1) a **non-finite metric** (numerical instability); (2) **val MAE
  diverging** (rose 3 checks straight *and* > 1 dB above best); (3) **slope
  collapse** (val slope < 0.08 for **8 checks** straight — requires `len(slope) >= 8`
  before the check can fire, giving the model at minimum ~16 epochs after the val
  gate first opens to actually learn; see §1.2 for why the former threshold of 3
  caused a false stop in run 2). Soft live **warnings** (no stop), each with a
  **minimum-check guard** to skip initialization transients: low slope (< 0.20,
  requires `len(slope) >= 5`), large |bias| (> 2 dB, requires `len(bias) >= 5`),
  and a rising deep-floor MAE (requires `len(floor) >= 5`; formerly 3). The existing plateau-patience stop now also records a
  reason.
- **End-of-run "DIAGNOSIS" block.** Prints the stop reason and the
  start→best/last trajectory of Val MAE, slope, bias, and Floor 0–10 (the goal-1
  metric), plus an interpretation tied to the deep-floor knobs: if the
  best-checkpoint bias > 1, suggests raising `--overpred-penalty` /
  `--floor-boost`; if < −1, lowering them; otherwise "calibration healthy".

---

## 9. How to run

Always use the project's `python` (pyenv: numpy 1.26 + torch 2.7); `python3` on
this machine is a broken alias.

```bash
# 0. Sanity checks (fast, no model)
python decoder/garway_heath_weighting.py --selftest

# 1a. Baseline (unchanged) — for reference
python decoder/training.py                       # or --weighting baseline

# 1b. Garway–Heath sector-weighted training (opt-in)
#     Deep-floor shaping (§4.1) is ON by default under garway_heath.
python decoder/training.py --weighting garway_heath --sector-combine both
#   → saves results/garway_heath/inference_model_gh.pth (+ best_model_gh.pth, config.json)

# 1c. Tune the deep-floor rescue without editing code (GH only)
python decoder/training.py --weighting garway_heath --floor-boost 3.0 --overpred-penalty 0.9
python decoder/training.py --weighting garway_heath --floor-db 14.0

# 2. Regenerate ALL results (metrics, scatter, heatmap, comparison, config)
python decoder/garway_heath_weighting.py --evaluate
#   compares results/garway_heath/inference_model_gh.pth vs baseline inference_model.pth

# Ablations
python decoder/training.py --weighting garway_heath --sector-combine sector_only
python decoder/training.py --weighting garway_heath --sector-combine value_only   # == baseline
python decoder/training.py --weighting garway_heath --floor-boost 0 --overpred-penalty 0  # pure sector run (run-1)
```

---

## 10. Goals & external comparison (TDV-Net)

Three success criteria drive this work:

1. **Better severe-case prediction.** Measured by the severity-stratified MAE
   (§7): per-point GT-sensitivity bins (esp. `deep(<16 dB)`) and per-eye severity
   groups. A higher regression **slope** (less regression-to-mean) is a secondary
   signature — the sector weighting should pull peripheral/deep predictions away
   from the population mean.
2. **Beat the prior raw-sensitivity MAE.** The historical project baseline was
   quoted at **3.74 dB**, but that was an *older/better checkpoint*; the current
   `inference_model.pth` measures **4.21 dB** (independently reproduced by
   `generate_scatterplot.py` at 4.24). The comparison CSV reports against whatever
   baseline checkpoint is supplied via `--baseline`; if the 3.74 checkpoint is
   located it can be dropped in there.
3. **Compare on TDV-Net's footing** (Park et al., Pusan National University,
   *Graefe's Archive* 2026; EfficientNet-B3; PNUH train / YPNUH test; **pointwise
   MAE on Total Deviation ≈ 3.91 dB**).

   **Total-Deviation MAE ≡ raw-sensitivity MAE.** TD is a per-point shift,
   `TD(x) = sensitivity(x) − normal(x, age)`, so the same normative cancels in the
   error: `|TD_pred − TD_gt| = |pred − gt|`. Hence the **pointwise MAE, RMSE and
   bias are identical** whether expressed in sensitivity or TD — our dB MAE is
   **directly comparable** to TDV-Net's 3.91 dB with no normative required. (Only
   correlation/slope/R² change under the shift, since they depend on the value
   spread.) GRAPE provides `Age` (18–79) in `grape_data.xlsx`, joinable by
   `Subject Number`+`Laterality`, so a **TD-space scatter** (matching TDV-Net's
   slope/R²/axes) can be produced if desired; that figure — and only that figure —
   needs an embedded citable 24-2 age-normative (Heijl et al. 1987, as used in the
   `visualFields` R package; Marín-Franch & Swanson, *J Vis* 2013). The headline
   MAE comparison does not.

Note: GRAPE's `Category of Glaucoma` is glaucoma **type** (OAG 254 / ACG 9), not
severity stage, so severity is derived from the VF itself (GT bins + per-eye mean
sensitivity), not from that column.
