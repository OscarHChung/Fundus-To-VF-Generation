# Breakthrough design doc — reaching MAE < 4.0 AND slope ≥ 0.6 (fundus-only)

**Audience:** a fresh implementing session. **Status:** research/design only — nothing here is built.
**This is v2**, rewritten after a 4-agent research pass (codebase grounding + a quantitative feasibility
analysis + two literature surveys). The v1 hypothesis ("label noise caps r") was **quantitatively
refuted**; the priorities below are re-derived from the math and cross-checked against the literature.

**Goal (hard):** leak-free per-patient 5-fold OOF, scored vs the RAW observed VF — **pooled MAE < 4.00
AND line-of-best-fit slope ≥ 0.60**, without worsening the severe-band MAE, r ideally ≥ 0.72.

**Non-negotiable constraints:** two-stage pretrained-RETFound-encoder + pretrained-VF-decoder (LoRA/
adapter/SSL adaptation still counts as pretrained RETFound); **FUNDUS-ONLY inference** (fundus +
laterality only — no prior VF / OCT / RNFL / text at test time); Garway–Heath sectoring stays;
open data only (UWHVF 28,943 VF-only + GRAPE ~631 fundus↔VF records / 263 eyes / 144 patients);
longitudinal & structural data (OCT/RNFL/IOP/age) may be used **in TRAINING ONLY**; don't change the
frozen folds/seeds/eval; never overwrite `long_global_*` / `long_prior_*`; score only vs raw VF; report
raw AND variance-matched-calibrated with the severity-stratified table every time.

---

## 0. TL;DR — the decisive finding

A faithful feasibility analysis on the actual out-of-fold predictions (631 records, 32,812 points)
says:

- **The target is feasible fundus-only and needs pooled r ≈ 0.72–0.75 (best estimate 0.73), up from
  the current 0.657** — a **+0.07** jump. At r = 0.657, MAE < 4.0 is impossible at any slope (rescale
  floor 4.258); you cannot calibrate your way there — r must rise.
- **Label noise is NOT the binding constraint** (v1 was wrong). Measured per-point test-retest σ ≈
  **3.17 dB** ⇒ a noisy-label correlation ceiling r_max ≈ **0.916**, far above the needed 0.73. The
  current model sits far *below* its own ceiling (signal-corr 0.717 vs 0.916). So denoising helps only
  as *variance reduction on small data*, not by "lifting a noise ceiling."
- **The single largest, most attainable lever is BETWEEN-EYE SEVERITY (eye-mean / MD) estimation.**
  Between-eye variance is 56% of the total; the model's severity corr is already 0.80 while its
  within-eye spatial corr is only 0.41. Decomposition on the real predictions:

  | intervention (keep everything else fixed) | MAE | slope | r |
  |---|---|---|---|
  | current (loraC) | 4.220 | 0.485 | 0.657 |
  | raise eye-mean severity corr 0.80 → **0.875** | **3.986** | **0.600** | 0.707 |
  | raise eye-mean severity corr 0.80 → 0.92 | 3.762 | — | 0.737 |
  | perfect severity + current spatial (ceiling of this lever) | 3.411 | 0.650 | 0.794 |

  **Improving fundus→severity from r 0.80 to ~0.875 hits BOTH targets by itself** — and it *also*
  de-shrinks the pooled slope from 0.49 to 0.60 with no rescale, because the low slope is largely
  between-eye compression (predicted eye-means are shrunk). Published fundus→MD models reach r ≈
  0.8–0.9, so 0.875 is realistic. **v1's claim "beating 4.0 REQUIRES within-eye spatial accuracy" is
  false** — severity is the cheaper, bigger lever, and within-eye spatial (0.41→~0.51) is a valuable
  *second* lever that adds margin and deepens scotomata.
- **Where the error lives:** the entire MAE excess over 4.0 and the whole slope deficit are in the
  **moderate + severe bands** (severe MAE 7.69 / slope 0.20; moderate 5.23 / slope 0.20; mild is
  already 2.68). Every method should be judged on moderate+severe.

**Corrected facts** (v1 had these wrong): eye-mean oracle MAE ≈ **3.85** (not 4.09; 4.09 was the
263-eye baseline split); test-retest σ ≈ **3.17 dB** (not 2.4); per fold ≈ **210 eyes / ~500 records**
(not "500 eyes"); the longitudinal teacher is ≈ **persistence** (its delta head adds ~nothing), so
distilling it transfers a *prior-VF anchor the fundus can't recover* → demote it.

---

## 1. What's already been tried (do NOT repeat — see improvement_log.md)

- **LoRA on the last K ViT blocks** (cached-prefix trainer, `decoder/train_lora_cached.py`): raises r
  on favorable folds (fold 0: r 0.73) but **overfits the ~210-eye folds** → pooled r stayed 0.657 and
  severe got worse. Pooled OOF: 4.220 / 0.485 / r 0.657 (calib 4.480 / 0.628).
- **Balanced-MSE (BMC) loss** alone at r≈0.66: dominated by post-hoc variance-matched calibration.
- **Per-eye Theil–Sen target denoising** and **cached rotation augmentation**: marginal-to-null.
- **Rank 16 > overfit**; **low decoder-LR > worse**; **warm-start decoder + slope-aware selection** =
  the best knobs found. Reuse the cached-prefix trainer — it's the RAM-safe, fast path on the 17 GB box.

---

## 2. Feasibility math (why the priorities are what they are)

- Exact frontier: a predictor of correlation r rescaled to best-fit slope g has
  `RMSE = σ_t·√(1 − 2g + g²/r²)` (min at g = r², `RMSE = σ_t·√(1−r²)`). Solving MAE@slope0.60 = 4.0
  gives **r\* ≈ 0.72–0.75** (Gaussian 0.77 pessimistic; leptokurtic-VF empirical 0.70 optimistic;
  faithful sim ≈ 0.725).
- Variance split (631 records): **between-eye severity 56% / within-eye spatial 44%.**
- Noisy-label ceiling: σ_n 3.17 → r_max 0.916 (σ_n 2.4→0.95, 4.0→0.86 — all ≫ 0.73). **Not binding.**
- Lever isolation (above table): severity is the dominant, most-attainable path; fixing it also fixes
  the slope. Within-eye spatial ceiling on frozen features ≈ 0.51 (probe) — a real but secondary lever.

**Conclusion: the binding constraints are (a) global severity/feature quality (L2) and (b) small-fold
overfitting/variance (L3) — NOT label noise (L1).** Optimize for eye-level severity in moderate+severe
eyes first; add within-eye spatial and variance reduction for margin.

---

## 3. Prioritized methods (re-derived; cross-checked vs literature)

### M1 — Sharpen fundus→SEVERITY (eye-mean / MD), focused on moderate+severe  ★ the decisive lever
The model already has a `mean_residual` decomposition (CLS→eye-mean + zero-mean spatial residual) and a
`global_head`. Make severity a **first-class, well-supervised, well-calibrated output**:
- Explicit **eye-mean / MD regression head** with its own loss term; de-shrink it (its predictions are
  compressed — eye-mean slope 0.71). Consider a **two-stage: predict MD (severity) then the within-eye
  pattern conditioned on it** (coarse→fine).
- **Concentrate capacity/loss on moderate+severe eyes** (where all the error+slope deficit live) via
  eye-level severity reweighting — but guard the pooled bias (don't just shift the level).
- Expected (from §0 table): severity r 0.80→0.875 ⇒ MAE 3.99 / slope 0.60 / r 0.71 — clears both.
- Gate: pooled eye-mean corr ↑ and pooled slope ↑ at MAE ≤ baseline; then full-CV MAE < 4.05.

### M2 — Structural-surrogate multi-task: fundus → RNFL ("Machine-to-Machine")  ★ raises the r ceiling
GRAPE's xlsx **contains OCT RNFL thickness, IOP, CCT, age, gender** (Baseline sheet) that are **NOT
currently extracted** (`build_longitudinal_grape.py` drops them). RNFL is the *actual* structure-
function substrate and — unlike the prior VF — **is learnable from a fundus** (Medeiros "M2M",
Ophthalmology 2019: predicted-RNFL-from-fundus discriminated glaucoma at AUC 0.94). Add an **auxiliary
head that regresses RNFL (and/or MD/age) from the fundus features**, training-only; fundus-only at
inference. This injects the learnable structural signal that raises both severity and spatial r — the
theoretically "right" privileged channel (vs the un-learnable prior VF).
- First step: extend `build_longitudinal_grape.py` to emit RNFL/IOP/CCT/age per record; audit coverage.
- If RNFL coverage is thin, pseudo-label GRAPE fundus with a frozen M2M/RNFL model.
- Gate: aux head trains without hurting VF MAE; ablation shows +r on the main task.

### M3 — Replace LoRA with ULTRA-LOW-DOF adaptation + variance reduction  ★ fixes the fold-overfit
The 5-fold collapse is an L3 (variance/overfit) problem. Both literature agents: on ~500 samples,
**fewer trainable params generalize better.**
- **Swap LoRA → VeRA (≈10× fewer params than LoRA), IA³, BitFit (bias-only), or LayerNorm-affine
  tuning**, later blocks only, + **LoRA-dropout**, high weight-decay, low rank.
- **Variance reduction (near-free, compounding):** per-fold **5-seed or snapshot ensemble**, **SWA/EMA**
  of decoder+adapter, **manifold / C-Mixup** (mix at RETFound-feature level, pairing VF-similar eyes),
  **multi-scale + multi-crop TTA** (generalize the current rotation-only TTA).
- Gate: pooled r ≥ 0.70 AND MAE ≤ 4.10 AND severe ≤ baseline (must beat the LoRA-overfit result).

### M4 — Structured, imbalance-aware output for the moderate+severe band  ★ slope + severe
- **Deep Imbalanced Regression: LDS + FDS** (label- and feature-distribution smoothing; density fit on
  the 28,943 UWHVF fields) + a **few-shot-region loss (Dist-Loss)** — the textbook fix for the deep band
  and regression-to-mean; a *different mechanism* than the calibration-dominated Balanced-MSE.
- **Predict into a structured VF target**, not 52 free scalars: the frozen **VF-AE 64-d latent** (decode
  to a plausible field) and/or **archetype weights** (Elze archetypal analysis on UWHVF — ~15 clinical
  loss patterns). Constrains outputs to real glaucomatous fields → de-noises, infills, protects slope,
  represents deep localized loss.
- **Distributional / ordinal per-point head** (CORAL/CORN, or the existing `--head distributional`) for
  the bounded 0–40 dB range → better deep points, principled slope without post-hoc variance match.
- **Respect the biophysical floor (Hood):** RNFL/structure floors out below ≈ −12 dB; do not chase or
  over-penalize the deepest points — censor/weight them and report severe separately.
- Gate: severe MAE ≤ baseline AND pooled slope ↑ (this is the method that must FIX the severe regression
  the LoRA run caused).

### M5 — Fundus-derivable context at inference (MLEDL-style, inference-safe)
MLEDL (npj Digital Med 2025, **633 patients — our exact regime, fundus-only at inference**) reached
pointwise MAE **3.10–3.90** using **optic-disc/cup segmentation ROI + metadata (age, interval)** — no
prior VF. Add a disc/cup segmentation branch (or disc-centred crop, the `disc_crop` path exists) and any
inference-available metadata. Multi-view (disc + full, multi-scale) late-fusion targets the peripapillary
region where structure-function lives.

### M6 — Target denoising as a VARIANCE-REDUCTION regularizer (not the primary lever)
Repositioned from v1's #1. It helps small-fold generalization, bounded (label noise isn't binding):
- **VF-AE as a masked-autoencoder denoiser** on the training targets (a VF masked-AE beat a VAE by
  26.5%); **3×3 spatial / Garway-Heath-cluster smoothing**; **per-eye Kalman/GP** smoothing across the
  eye's visit series; or **heteroscedastic / batch-inverse-variance** weighting to down-weight noisy
  deep points during training. Use as a modest add-on under M1–M4, with a raw-slope guardrail.

### M7 — Longitudinal distillation — DEMOTED to a slope-guarded regularizer
The teacher's edge is a **prior-VF persistence anchor the fundus cannot recover** (LUPI "worst case";
failure mode = regression-to-mean → *lower* slope). Keep only as a **small soft-target regularizer on
follow-up visits**, instrumented with an **OOF-slope guardrail** — if pooled slope drops, cut it.

---

## 4. Recommended stack & expected trajectory

Build cumulatively; keep only what passes its full-CV gate (self-correcting loop, one method at a time).

1. **M1 (severity) + M3 (low-DOF adapter + ensembling/SWA/EMA)** — the two highest-confidence levers:
   better, well-calibrated global severity from better-generalizing adapted features. Target: pooled
   r → ~0.70–0.72, MAE → ~4.0, slope → ~0.58–0.60 (per §0 table this alone can clear the bar).
2. **+ M2 (fundus→RNFL surrogate)** — raise the feature/severity ceiling with the learnable structural
   signal. Target r → ~0.72–0.74.
3. **+ M4 (structured/imbalance-aware output)** — fix the severe band and lock slope ≥ 0.60 without
   trading MAE; keep severe ≤ baseline.
4. **+ M5 (segmentation/metadata) + M6 (denoising) + M7 (guarded distillation)** — margin & robustness.

**Plausible landing:** M1+M3 alone are quantitatively sufficient to reach MAE ≈ 4.0 / slope 0.60 / r
≈ 0.71 if severity corr goes 0.80→0.875; M2+M4 add the margin and severe-band safety to land firmly
**sub-4.0 with slope ≥ 0.6 and severe ≤ baseline**. Ambition check: MLEDL hit 3.1–3.9 fundus-only at
633 patients, and the eye-mean oracle here is 3.85 — the target is inside the achievable region.

---

## 5. Harness / environment notes (save the next session hours)

- **Memory:** 17 GB box; live encoder-gradient training OOMs. Use the **cached-prefix trainer**
  (`decoder/train_lora_cached.py`): cache the frozen prefix, free those blocks, train the suffix +
  decoder (+ any adapter/aux head) on cached (fp16, batched) features; warm-start the decoder from
  `long_global_f{i}`. Extend it for VeRA/IA³/BitFit, the RNFL aux head, structured outputs.
- **Run discipline:** exactly ONE encoder-loading process at a time; no monitors during training; save
  best every val; make eval resumable (per-fold `.npz`, see `scratchpad/eval_oof.py`); prefer many
  short (<20 min) fold jobs. `memory_pressure` (not vm_stat "free") is the true availability signal.
- **Eval:** `decoder/diagnostics.py` `pooled_metrics` + `stratified_report`; calibrate per-fold on that
  fold's train only; log every attempt to `decoder/results/auto/improvement_log.md`.
- **Assets to reuse:** `train_lora_cached.py` (cached trainer, warm-start, slope-select, denoised flag),
  `vf_autoencoder.py` (`load_ae`, d=64, encoder+decoder — for M4 latent target & M6 denoiser),
  `losses.py::balanced_mse_loss`, `build_denoised_targets.py`, `build_longitudinal_grape.py` (extend for
  M2 structural fields), `train_longitudinal.py` (M7 teacher), `garway_heath_weighting.py`,
  `eval_ckpt.py` (LoRA-aware reload), `decoder/diagnostics.py::cmd_baseline` (oracle/variance).

## 6. Integrity guardrails
Fundus-only inference; open data only; longitudinal/structural/teacher/denoising are TRAINING-ONLY;
score vs raw VF; never overwrite baseline checkpoints; keep the severity table with every pooled number;
a pooled gain that worsens severe is NOT a win; commit per method that passes its full-CV gate; branch.

## 7. Key references (from the research pass)
- Feasibility: eye-mean oracle 3.85, r\* ≈ 0.73, r_max 0.916, severity-lever table — `scratchpad/feasibility.py`.
- Medeiros et al., "From Machine to Machine" (fundus→RNFL, Ophthalmology 2019) — M2, PMC6884092.
- MLEDL, npj Digital Medicine 2025 (fundus-only, 633 pts, MAE 3.10–3.90) — M1/M5, nature s41746-025-01750-8.
- Park et al. 2026 TDV-Net (fundus→TDV, MAE 3.91, severe 9.15) — SOTA context, Graefe's 10.1007/s00417-026-07141-3.
- Yang et al., Deep Imbalanced Regression (LDS+FDS, CVPR 2021, arXiv 2102.09554); Dist-Loss (2411.15216) — M4.
- VeRA (2310.11454), IA³, BitFit (2106.10199), LoRA-Dropout (2404.09610) — M3.
- Elze archetypal VF analysis; conditional-VAE normative modeling; VF masked-AE denoiser (2411.12146) — M4/M6.
- Lopez-Paz Generalized Distillation / LUPI (ICLR 2016); privileged-features caveat (NeurIPS 2022) — M7 demotion.
- Hood RNFL structure-function floor (< −12 dB) — M4 severe-band biophysical limit.
