# Breakthrough design doc — reaching MAE < 4.0 AND slope ≥ 0.6 (fundus-only)

**Audience:** a fresh implementing session. **Status:** research/design only — nothing here is built yet.
**Goal (hard):** leak-free per-patient 5-fold OOF, scored against the RAW observed VF —
**pooled MAE < 4.00 AND line-of-best-fit slope ≥ 0.60**, without worsening the severe-band MAE and
with Pearson r ideally ≥ 0.70.

**Non-negotiable constraints (identical to prior sessions):**
- Two-stage encoder–decoder: **pretrained RETFound encoder** + **pretrained VF decoder**. LoRA/adapter/
  self-supervised adaptation of RETFound still counts as "pretrained RETFound".
- **Inference is FUNDUS-ONLY** (fundus photo + laterality). No prior VF / OCT / text at test time.
  Longitudinal & multi-visit data may be used in **TRAINING ONLY** (targets, teachers, denoising).
- **Garway–Heath anatomical sectoring** stays in the model.
- **Open data only** (UWHVF ~29k VF-only; GRAPE fundus↔VF, ~631 longitudinal records / ~250 eyes).
- Do not change the frozen folds/seeds/eval; never overwrite `long_global_f*` / `long_prior_*`
  baselines; score ONLY against raw VF; report raw AND variance-matched-calibrated; keep the
  severity-stratified table with every pooled number.

---

## 1. Where we are (read this first)

- **Baseline `long_global` (fundus-only, 5-fold OOF):** raw MAE **4.290**, slope **0.473**, r **0.657**,
  severe ~7.49; calibrated 4.536 / 0.637.
- **Best this session — cached-LoRA (`decoder/train_lora_cached.py`), 5-fold OOF:** raw **4.220 /
  0.485 / r 0.657 / severe 7.694**; calib **4.480 / 0.628**. It BEAT the baseline modestly but missed
  the target, did not raise pooled r, and slightly worsened severe.
- **What worked / didn't (so the next session doesn't repeat it):**
  - LoRA on the last K ViT blocks (cached-prefix training to fit 17 GB RAM) **raises r on favorable
    folds** (fold 0: 4.10 / 0.57 / r 0.73) — proof the r-lever is real.
  - It **overfits on the hard/small folds** (~500 train eyes/fold → fold 2: 4.43/r0.66, fold 4:
    4.40/r0.56), so the best checkpoint there collapses back to the warm-start ≈ baseline. **Overfitting
    is the #1 enemy.**
  - Balanced-MSE (BMC) loss alone is **dominated by post-hoc variance-matched calibration** at r≈0.66
    (it de-shrinks less efficiently than a linear rescale and lowers r). Revisit only AFTER r is higher.
  - Per-eye Theil–Sen target denoising (`build_denoised_targets.py`) and cached rotation-augmentation
    were **marginal-to-null** — the trend model was too weak and rotation adds little diversity.
  - Rank 16 > overfit; low decoder-LR > worse. **More capacity is not the lever; better generalization
    and cleaner targets are.**
- **The ceiling to respect:** the "predict each eye's true mean everywhere" oracle is **MAE ≈ 4.09**.
  Beating 4.0 REQUIRES genuine *within-eye spatial* accuracy (RNFL/disc → Garway–Heath → point defects),
  not just severity. Every method below is judged on whether it raises **within-eye** signal (eyeCorr,
  currently ~0.47; frozen-feature probe ceiling ~0.51) and **robust pooled r** (need ~0.72–0.75).

**Mental model of the levers** (each method is tagged L1–L5):
- **L1 Cut label noise** (raise *effective* r): test-retest σ≈2.4 dB/pt corrupts single-visit targets.
- **L2 Raise feature quality** (raise the r *ceiling*): better/adapted RETFound features.
- **L3 Cut variance/overfitting** (the thing that killed pooled LoRA).
- **L4 Constrain outputs to plausible VFs** (better slope + severe): VF-manifold / anatomy priors.
- **L5 Convert r → slope at ~no MAE cost** (finish): loss + calibration-aware selection.

---

## 2. TIER-1 methods (highest expected impact — do these first)

### T1. Distill a FUNDUS-ONLY student from the LONGITUDINAL teacher  [L1 + L3] ★ most promising
**Idea.** The longitudinal model (`decoder/train_longitudinal.py`, `long_prior`) reaches ~**3.75** MAE
because it sees each eye's prior VF — it is effectively a *denoiser* that knows the eye's baseline. It
cannot be used at inference (needs prior VF), **but its per-visit predictions on the TRAIN set are
much cleaner targets than the raw noisy VF.** Train a fundus-only student to regress the teacher's
predictions (soft targets) + the raw VF (hard targets), e.g. `loss = α·Huber(student, teacher_pred) +
(1-α)·Huber(student, raw_vf)`. Do it **per fold, teacher trained only on that fold's train** (no leak).
- Why it should work: distillation transfers the longitudinal accuracy into the fundus-only mapping;
  the teacher's targets have far less test-retest noise → the student learns true structure-function,
  raising effective r beyond the 0.66 label-noise-capped ceiling. This is the single most direct route
  to sub-4.0 fundus-only and is fully compliant (teacher = training-only device).
- Variants to try: (a) soft-target distillation as above; (b) teacher as a *target denoiser* only
  (train on teacher_pred alone); (c) ensemble-of-teachers (several longitudinal seeds) → average →
  even cleaner targets; (d) feature/attention distillation (match the student's Garway–Heath attention
  to the teacher's).
- Guardrail: teacher predictions must be **causal & fold-honest** (teacher trained on fold-train only;
  its prior-VF inputs come from the same eye's earlier visits, all within train). Student eval = raw VF.
- Gate: fold-0 student MAE ≤ 4.05 (raw) AND r ≥ 0.70. Expected pooled ~3.9–4.1.

### T2. Proper per-eye TRAJECTORY denoising of training targets  [L1] ★
**Idea.** Replace the weak Theil–Sen trend with a real per-eye smoother over the full visit series,
then evaluate at the target date. Options, best-first:
1. **Gaussian-Process / state-space (Kalman) smoother** per point per eye: model true sensitivity as a
   slowly-varying latent + measurement noise (σ≈2.4 dB), posterior-mean at the target date = denoised
   target. Borrow strength across the 52 points via the **Garway–Heath sector covariance** (points in a
   sector move together) and across the eye via a low-rank pointwise prior.
2. **VF-manifold denoising**: pass each raw field through the frozen VF autoencoder
   (`decoder/vf_autoencoder.py`, `pretrained_vf_ae.pth`) → project to the plausible-VF manifold; blend
   with the temporal smoother.
3. Combine 1+2 (temporal + manifold) → the cleanest achievable target; keep raw for eval.
- Why: directly attacks L1; the current single-visit label noise is the dominant cap on r.
- Gate: mean |denoised−raw| ~1.5–2.5 dB (not ~0); fold-0 MAE ≤ baseline_fold0 AND r not worse.
- Note: T1 (distillation) and T2 (denoising) both produce cleaner targets — **combine them** (train the
  student on the denoised+teacher-blended target). This is the core of the plan.

### T3. Self-supervised DOMAIN ADAPTATION of RETFound on glaucoma fundus  [L2] ★
**Idea.** RETFound is MAE-pretrained on general retinal images. Continue **self-supervised** pretraining
(MAE reconstruction and/or DINO/iBOT-style) on ALL available fundus (GRAPE + any open glaucoma fundus)
**without VF labels**, to make features glaucoma/RNFL-relevant BEFORE the decoder. This raises the r
*ceiling* with **zero VF-label overfitting** (the L3 killer), because it uses no VF labels. Still
"pretrained RETFound" (self-supervised adaptation of the same weights).
- Then attach the decoder (frozen or lightly adapted). Compare feature-probe r before/after.
- Gate: linear-probe r on frozen adapted features > 0.66 (the current probe ceiling).

---

## 3. TIER-2 methods (raise r robustly / cut variance)

### T4. Less-overfitting encoder adaptation + per-fold ENSEMBLING  [L2 + L3]
LoRA overfit on 500-eye folds. Try adaptation with **far fewer trainable params**, which generalizes
better on tiny data:
- **IA³** (learned per-channel rescales of K/V/FFN — hundreds of params/block) and **VeRA** (shared
  frozen random LoRA with tiny learned scaling vectors). Both are ≪ LoRA params → less overfit.
- **DoRA** (weight-decomposed LoRA) — usually generalizes better than LoRA at equal rank.
- **BitFit** (norm+bias only; already wired via `--finetune-norm`) as a cheap baseline.
- **Ensemble 3–5 adapted models per fold** (different seeds / bootstrap resamples of the fold-train) and
  average predictions → variance reduction directly fixes the "hard fold collapses to baseline" problem.
  Expected −0.05 to −0.10 MAE and steadier r. Reuse the cached-prefix trainer (fast, fits RAM).
- Gate: pooled r ≥ 0.70 AND MAE ≤ 4.10 AND severe ≤ baseline.

### T5. Manifold-constrained decoding (predict the VF-AE latent)  [L4]
Decode fundus → the frozen VF autoencoder's **64-d latent** → VF via the frozen AE decoder, instead of
52 free scalars. Predictions are guaranteed to lie on the plausible-VF manifold → cleaner spatial
structure, better slope, and (critically) **better severe band** (scotomata become coherent, not noisy).
Keep the per-point Garway–Heath attention as the fundus→latent map. Optionally blend manifold-decoded +
direct-point predictions.
- Gate: severe MAE ≤ baseline (this is the method most likely to FIX the severe regression) AND slope up.

### T6. Structure-function coarse-to-fine + ordinal severe head  [L4 + L5]
- **Coarse→fine:** predict the 6 Garway–Heath **sector means** first, then a within-sector point
  residual. Injects anatomy, regularizes, and matches the paper's weakness (spatial/severe).
- **Ordinal / distributional per-point head** (the `--head distributional` path exists): predict a
  distribution over dB bins with soft cross-entropy → deep points get full-strength gradients
  regardless of rarity → **directly targets the severe band** where we're weakest.

---

## 4. TIER-3 methods (finish / squeeze — cheap, stack last)

### T7. Multi-view / multi-scale / disc-aware input  [L2]
Feed **disc-centred crop + full image** (the `disc_crop` path exists) and/or 2 resolutions; average
their patch tokens or late-fuse. RNFL structure-function lives around the disc → more within-eye signal.

### T8. Expanded TTA + ensembling at inference  [L3]
Current TTA = 3 rotations. Add flips, small scales, disc-crop views, and average — pure variance
reduction, honest MAE gain (~−0.03 to −0.07). Combine with the T4 model ensemble.

### T9. Balanced-MSE / slope-aware loss REVISITED under higher r  [L5]
Once r ≥ 0.72 (after T1–T4), BMC (already implemented, `--loss balanced_mse`) or a CCC/slope penalty
can convert the higher r into slope *in training* (it failed at r≈0.66 because it was dominated by
calibration). Tune `--noise-sigma` upward (de-shrink grows with σ on this data; see improvement_log).

### T10. Calibration-aware, pooled-objective checkpoint selection  [L5]
Per-fold "min val-MAE" (and even "min MAE−0.5·slope") discarded the LoRA r-gains on hard folds. Select
each fold's checkpoint to optimize the **pooled, post-calibration** objective (e.g. pick the epoch that
maximizes r, since calibration then buys slope at ~no MAE cost at high r). Consider a tiny **held-out
selection set** or nested CV so selection is honest.

---

## 5. Recommended stack & expected trajectory

Build cumulatively, checking the 5-fold gate after each (keep only what survives, per the playbook loop):

1. **T2 + T1** cleaner targets (trajectory+manifold denoising, then distill the longitudinal teacher).
   → expected the biggest single jump (attacks L1, the label-noise cap). Target r → ~0.70–0.72.
2. **+ T3** SSL domain-adapted encoder (raise the ceiling, no label overfit). r → ~0.72–0.74.
3. **+ T4** IA³/VeRA/DoRA adaptation with per-fold ensembling (robust r, cut variance). MAE floor down.
4. **+ T5/T6** manifold-latent decoding + ordinal severe head (fix severe, add slope).
5. **+ T8 + T9 + T10** expanded TTA, BMC-at-higher-r, calibration-aware selection (finish to slope 0.6).

**Plausible landing:** cleaner targets + adapted encoder push pooled r to ~0.73–0.75; at that r, MAE
~3.9–4.05 raw and slope ~0.58–0.62 raw (or 0.6+ with light calibration at ~no MAE cost), severe ≤
baseline via the manifold/ordinal decoder. This is the realistic route to the target; T1 (distillation)
is the highest-leverage and should be prototyped first.

## 6. Harness / environment notes (save the next session hours)

- **Memory:** 17 GB box; live-encoder-gradient training OOMs. Use the **cached-prefix trainer**
  (`decoder/train_lora_cached.py`): cache the frozen prefix, free those blocks, train the suffix +
  decoder on cached (fp16, batched) features. Warm-start the decoder from `long_global_f{i}`. This is
  the proven-fast, RAM-safe path — extend it for IA³/VeRA/distillation/manifold-decoding.
- **Run discipline:** exactly ONE encoder-loading process at a time (concurrent python OOM-kills each
  other); no monitors during training. Runs can be killed ~mid-way — **save best every val and make eval
  resumable** (see `scratchpad/eval_oof.py`, per-fold `.npz` cache). Prefer many short (<20 min) fold
  jobs over one long job.
- **Eval:** score with `decoder/diagnostics.py` `pooled_metrics` + `stratified_report`; fit calibration
  per-fold on that fold's train only; log to `decoder/results/auto/improvement_log.md` (the scoreboard).
- **Existing assets to reuse:** `train_lora_cached.py` (cached trainer, warm-start, slope-select,
  denoised flag), `losses.py::balanced_mse_loss`, `build_denoised_targets.py`, `vf_autoencoder.py`
  (`load_ae`), `train_longitudinal.py` (the teacher), `eval_ckpt.py` (LoRA-aware reload),
  `garway_heath_weighting.py`. All three methods A/B/C are flag-gated, default OFF (reproduce baseline).

## 7. Integrity guardrails (do not break)
Fundus-only inference; open data only; longitudinal/teacher/denoising are TRAINING-ONLY; score against
raw VF; never overwrite baseline checkpoints; keep the severity table with every pooled number; a pooled
gain that worsens severe is NOT a win; commit per method that passes its full-CV gate; work on a branch.
