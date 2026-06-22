# Severe-Point & Sub-4 MAE Improvement Plan (STANDBY — not implemented)

**Status:** Tier-1 (M1 distributional head + M2 LDS) **IMPLEMENTED** behind
opt-in flags (`--head distributional`, `--reweight lds`), plus a goal-aware
self-stop (`--target-mae`). Triggered after the GH run stalled at val MAE 4.25
with a flat deep floor (~10–11 dB) and overfitting (Gap < −0.4). Tier 2 (dual
crop) and Tier 3 remain design-only standby for the next push.

**Goals, in priority order**
1. **Lower MAE on severe/deep points** — especially the 0–10 dB floor
   (`floor(0-10)` currently ~11–12 dB, flat) and `deep(<16)` (~9.5 dB, flat).
2. **Overall val MAE < 4.0 dB** (historical best 3.74; current run ~4.26).

---

## 1. Why the deep floor is stuck (root causes, grounded in the code)

**RC-1 — The loss structurally regresses to the mean.**
`compute_loss` (training.py:513) is per-point Huber (`F.huber_loss`, :579). For a
rare deep point, the Huber/L1 gradient has ~constant magnitude pointing toward
the conditional median. Because deep points are rare (~3.4 of 52 points/eye are
0–10 dB), the *aggregate* gradient over a batch favours pulling deep predictions
UP toward the bulk. `FLOOR_BOOST`/`OVERPRED_PENALTY` just rescale that same
gradient — they fight the geometry instead of changing it, which is exactly why
we saw the bias oscillate to −2.0 dB at epochs 18 and 38 (the boost briefly wins,
then the bulk pulls back). **Reweighting a mean-seeking loss cannot fix
mean-collapse; we need a loss whose geometry doesn't collapse.**

**RC-2 — Weak per-point input signal + tiny data.**
211 train eyes. Pinpointing 0 vs 5 dB at one location from a 224×224 whole-fundus
(196 patch tokens) is near the structural noise floor. The strongest structural
correlate of focal VF loss is **peripapillary RNFL at the disc** — which a single
downscaled whole-image encoding represents only coarsely.

**RC-3 — The Stage-1 prior is healthy-dominated.**
`VFDataset` (pretraining.py:60) trains the auto-decoder by reconstruction with no
severity oversampling. The decoder's learned prior is biased toward typical
(shallow) field shapes and smooths localized deep scotomata at inference.

Each method below is mapped to the root cause it attacks.

---

## 2. The plan — tiered by impact × feasibility × risk

### TIER 1 — do first (attacks RC-1; highest expected impact on Goal 1, low risk to overall MAE)

#### M1. Distributional / ordinal per-point head  ★ headline change
**What:** add a second head on the shared per-point feature
(`[attended ‖ cls]`, 2048-dim → PointHead, training.py:350) that predicts a
**distribution over dB bins** instead of a scalar. Discretize 0–35 dB into ~10–12
ordinal bins (finer near the floor: e.g. 0–2, 2–4, 4–7, 7–10, 10–14, …). Train
with **soft-label cross-entropy** (Gaussian-smoothed target around the true bin)
or **CORN/CORAL ordinal** loss. Final prediction =
`α · scalar_reg + (1−α) · E[bin]`, with the distribution's mode used to "snap"
confident deep cases.

**Why it works where FLOOR_BOOST didn't:** cross-entropy puts a strong gradient
on the *correct deep bin* regardless of that bin's rarity — there is no
median-seeking force. The model can commit to "this point is ~2 dB" without the
loss dragging it toward 22 dB. This is the standard fix for long-tailed targets:
DORN (depth estimation), CORAL/CORN (age estimation), ordinal medical grading all
adopt it precisely because L1/L2 collapse the tail. It is a **geometry change,
not a reweight** — the thing RC-1 says we actually need.

**Small-data safety:** head is shared across all 52 points (sample-efficient),
coarse bins, reuse existing dropout 0.4. Bins are *more* sample-efficient than
fine regression because each deep point contributes a full-strength
classification gradient.

**Cost/risk:** ~10–25k new params on the shared head; opt-in flag; if α=1 it
reduces exactly to today's model (safe fallback / clean ablation).

#### M2. Balanced-MSE / Label-Distribution Smoothing (LDS) — replace hand-tuned weights
**What:** replace the hand-tuned value weighting + `FLOOR_BOOST` with a
principled scheme from Yang et al. 2021 *"Delving into Deep Imbalanced
Regression."* Compute the empirical label density over dB, convolve with a
Gaussian kernel (LDS) to get an *effective* density, weight each point's loss by
inverse effective density. Optionally add **Balanced-MSE**, a closed-form loss
that injects the label prior to provably counter regression-to-mean.

**Why better than what we did:** our value weighting + `FLOOR_BOOST` are
hand-picked constants that caused the −2 dB bias swings. LDS derives the weights
from the data distribution (smooth, no cliffs) and removes the guesswork; it is
the principled generalization of exactly what we were hand-tuning. Pairs
naturally with M1 (weight the regression term; classification handles geometry).

**Cost/risk:** pure loss-side change, no new params. Low risk. Tunable kernel σ.

> **Tier-1 expected outcome:** floor(0–10) and deep(<16) MAE drop meaningfully
> with bias held near 0 (no oscillation), because the deep gradient no longer
> fights the bulk. Overall MAE should be neutral-to-better (less wasted capacity
> on the bias tug-of-war).

---

### TIER 2 — do if Tier 1 is insufficient, or for the final push to sub-4 (attacks RC-2)

#### M3. Dual disc + macula crop encoding  ★ biggest upside for BOTH goals
**What:** in addition to the whole-fundus 224×224, feed a **disc-centered crop**
and a **macula-centered crop** through the (frozen) RETFound encoder; let the
per-point attention attend over the union of patch tokens (or concatenate the two
CLS tokens into PointHead). The disc crop gives high-resolution peripapillary
RNFL detail — the literal structural basis of the Garway–Heath sector→VF mapping.

**Why it works:** attacks RC-2 (signal), not just the loss. Focal deep defects
have focal structural correlates that a downscaled whole-image blurs; a high-res
disc crop recovers them. This is the only method that raises the *information
ceiling* rather than re-allocating existing signal.

**Why it's cheap despite sounding heavy:** the encoder is frozen and we **already
cache its features** (`precompute_features` :776, `decode_latent` :456). We
pre-cache disc/macula features once, exactly like the current whole-image cache —
near-zero added train-time cost. Main work is disc/macula localization (a fixed
heuristic crop around the brightest disc region, or an existing disc detector).

**Cost/risk:** data-pipeline work + crop localization quality. Medium effort,
medium risk; highest payoff. Gate behind a flag; ablate whole-only vs +disc.

#### M4. Per-point hard-example mining
**What:** the current sampler is **per-eye** (training.py:933) — a low-mean eye
still contains many easy moderate points. Maintain a running per-point error EMA
across the dataset and upweight/oversample points that are *both* deep *and*
currently poorly predicted (finer grain than per-eye). Pairs with LDS.

**Why better:** targets the exact points failing Goal 1 rather than whole eyes.

**Cost/risk:** bookkeeping only; low risk. Watch for noise amplification — cap
the max per-point weight.

---

### TIER 3 — cheap, low-risk add-ons (safe to stack on any of the above)

#### M5. Post-hoc isotonic calibration (attacks RC-1 symptom, free)
Fit a monotonic predicted→true calibration per severity band on a train/CV fold
to undo systematic compression at the extremes. Cheap, never changes the model.
**Caveat:** the 47-eye val set is tiny — fit on a train fold or via CV, never on
the val we report, to avoid leakage.

#### M6. Severity-aware re-pretraining (attacks RC-3)
Re-run Stage-1 with severity oversampling / a deeper-field corruption curriculum
so the decoder prior isn't healthy-dominated. One-time cost; reduces the smoothing
bias the inference decoder inherits.

---

## 3. Integration order & decision gates

1. **M2 (LDS) + M1 (distributional head)** together — the core Goal-1 fix.
   Validate on `grape_test`. **Gate:** deep(<16) and floor(0–10) MAE both drop
   ≥ ~1 dB with |bias| ≤ 0.5 AND overall MAE not worse than the current best.
2. If overall MAE not yet < 4.0 → add **M3 (dual crop)**. This is the main lever
   for the last ~0.2–0.3 dB of overall MAE *and* further deep gains.
3. **M4** if specific deep points remain stubborn; **M5** as a free final polish;
   **M6** only if the prior is shown to dominate (inspect residual smoothing).

Introduce each method **one at a time, behind an opt-in flag** (mirrors the
existing `--weighting` / `--floor-boost` pattern) so every gain is attributable
and reversible — essential on 211 eyes where it's easy to fool a 47-eye val set.

---

## 4. Evaluation protocol & success criteria (the trade-off guard)

Reuse `severity_pointwise_mae` (garway_heath_weighting.py:283) and
`per_sector_mae` (:238). Every candidate must report, on held-out `grape_test`:

| Metric | Current | Target |
|---|---|---|
| floor(0–10) MAE | ~11–12 | **↓ ≥ 1 dB**, ideally < 9 |
| deep(<16) MAE | ~9.5 | **↓**, ideally < 8 |
| Overall val MAE | ~4.26 | **< 4.0** (and never regress while chasing Goal 1) |
| Global bias | ~0 | keep |bias| ≤ 0.5, **no oscillation** |
| EyeCorr | ~0.40 | keep / improve |

**Hard rule (priority encoded):** a change that improves deep MAE but regresses
overall MAE by > 0.1 dB is rejected unless the deep gain is large (> 2 dB) — Goal
1 outranks Goal 2, but we don't trade away a sub-4 result for a marginal floor
gain. Given the tiny val set, confirm the headline result with **k-fold or
repeated-seed CV**, not a single 47-eye split.

---

## 5. Review / red-team (failure modes & mitigations)

- **Distributional head over-commits to wrong deep bins on noisy labels.**
  Mitigation: soft (Gaussian) labels, coarse bins, blend with the regression
  scalar via α; α is tunable and α=1 recovers today's model.
- **LDS over-upweights the floor → re-introduces negative bias.** Mitigation:
  LDS weights are bounded by construction (inverse *smoothed* density, not raw),
  and we drop the asymmetric `OVERPRED_PENALTY` that caused the swings.
- **Dual-crop disc localization is unreliable on some images.** Mitigation:
  fall back to whole-image features when disc-crop confidence is low; ablate to
  prove the disc crop helps before committing.
- **Overfitting the 47-eye val.** Mitigation: every claim on `grape_test` +
  CV; one flag at a time; reject changes that only help val, not CV.
- **Net assessment:** M1+M2 attack the proven root cause (RC-1, mean-collapse)
  with a mechanism (classification geometry) that is the field-standard fix and
  cannot mean-collapse — high confidence they move the floor where reweighting
  could not. M3 is the highest-upside lever for both goals and is made cheap by
  the existing feature-cache infra. The plan degrades gracefully: each tier is
  independently reversible and the worst case reduces to the current model.

---

## 6. What this plan deliberately does NOT do
- Does not unfreeze RETFound (keeps the two-stage method intact).
- Does not change anything the running job imports.
- Does not rely on more global-knob tuning (the approach RC-1 shows is futile).
