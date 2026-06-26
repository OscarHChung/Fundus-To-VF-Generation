# Autonomous improvement log

Goal: minimize val MAE (target <4.0, historical best 3.74) AND severe-point MAE
(floor 0–10 dB), in that priority: **severe first, then overall**. Hard constraint:
encoder–decoder + Garway–Heath sectoring ALWAYS on. Champion (best-ever) auto-saved
to `decoder/results/champion/`.

## Baseline facts (from prior runs)
- v10.1 (light reg, dropout 0.2): train 3.57 / **val 4.02** — best architecture point.
- First GH run: ~4.04, floor ~10–11.
- GH + value weight (heavy reg 0.4): plateau **4.25**, floor ~10–11.
- Tier-1 (dist + LDS, run-4): floor **6.6** (!!) but bias runaway −4 → MAE 5.96 (diverged).
- Tier-1 + bias penalty (run-5): bias controlled (~0), but OSCILLATES between
  "predict mean" (MAE 4.43 / floor 13) and "predict deep" (MAE 5.2 / floor 9).
  Best-MAE checkpoint had WORST floor → dist+LDS not helping at bias-neutral point.

## Key diagnoses
- **Under-fitting now**: heavy reg (dropout 0.4, wd 0.015) caps train at ~4.4.
  v10.1's lighter reg fit to 3.57 train / 4.02 val. → reduce regularization.
- **Weak per-point signal (RC-2)**: model can only improve floor by globally
  deepening (hence oscillation). Real floor fix needs better signal (Tier-2 dual
  crop) or a smarter deep mechanism, not more loss reweighting.
- dist head + LDS add params/complexity without net benefit at bias-neutral point.

---

## Iteration 1 — re-establish a strong overall champion (the "~4.0 recipe")
**Hypothesis:** the path to ~4.0 is to FIT BETTER. Revert to proven simple methods
(scalar head, value weighting, gentle floor_boost) + KEEP the bias penalty +
REDUCE regularization (dropout 0.4→0.25, wd 0.015→0.008). Drop dist head + LDS.
**Config:** `--weighting garway_heath --head scalar --reweight value --dropout 0.25 --weight-decay 0.008`
(GH sectors ON [mandatory], deep_cfg floor_boost 0.5/overpred 0.15, bias_penalty 0.1)
**Predict:** train ↓ toward ~3.5, val toward ~4.0–4.2, bias ~0, floor ~10–11.
**Result:** 🏆 **val MAE 4.32** (e22), bias −0.10 ✓, floor 11.96 (bad), train ~4.6.
Stopped e30 goal-plateau. → CHAMPION (first). Partial win: bias controlled, but
still UNDER-fitting (train 4.6 ≫ v10.1's 3.57) and floor bad. Missed: I cut
dropout but left **LR at 5e-4**; v10.1 used **8e-4**. That's the missing fit lever.

## Iteration 2 — replicate the v10.1 "4.02 recipe" (push overall MAE to ~4.0)
**Hypothesis:** the fit gap is LR, not just dropout. Match v10.1 exactly: LR 8e-4,
dropout 0.2, wd 0.005. Keep GH (mandatory) + value weight + bias penalty 0.1.
**Config:** `--weighting garway_heath --head scalar --reweight value --lr 8e-4 --dropout 0.2 --weight-decay 0.005`
**Predict:** train ↓ toward ~3.6, val toward ~4.0. Floor likely still ~11 (no deep
mechanism) — that's iteration 3's job once the overall base is strong.
**Result:** 🏆 **val MAE 4.25** (e32), bias −0.29, floor 11.16. train→4.01 (LR fixed
the fit!). But val WALLS at 4.25, not 4.02. Diagnosis: GH×value weighting (≤4.6×
severe-peripheral vs 0.75× healthy-central) + oversampling optimize a WEIGHTED
loss while val MAE is UNWEIGHTED (healthy-dominated). GH (mandatory) taxes overall
MAE. → 4.0 vs severe-floor are in tension via the weighting. Reg tuning is exhausted.

## Iteration 3 — decoupling test: strong deep emphasis + bias control (goal-1 push)
**Hypothesis:** the bias penalty (run-5 win) should let us crank the SIMPLE deep
mechanism (floor_boost/overpred) without the run-4 global down-shift → better floor
WITHOUT losing MAE. Best-fit config + floor_boost 0.5→1.2, overpred 0.15→0.35.
**Config:** `--weighting garway_heath --head scalar --reweight value --floor-boost 1.2 --overpred-penalty 0.35 --lr 8e-4 --dropout 0.2 --weight-decay 0.005`
**Predict:** floor 11→~9, MAE hold ~4.25, bias near 0. If bias drifts −, penalty too
weak vs floor_boost → raise bias_penalty next. If floor flat → raise floor_boost.
**Result:** best MAE **4.28** (worse than champ), champion UNCHANGED. DECISIVE:
model OSCILLATES regime-to-regime — "predict deep" (bias −1.7, floor 9, MAE 4.8) vs
"predict shallow" (bias +1.5, floor 15, MAE 4.4); best-MAE ckpt always shallow
(floor 13). floor_boost vs bias-penalty fight each other. CONCLUSION: floor is
SIGNAL-limited (model can't localize deep points from whole-image) — reweighting
is dead for goal 1. Confirmed across runs 4/5/iter3.

## Iteration 4 — probe the overall-MAE ceiling (sub-4 hint)
**Hypothesis:** the 4.25 wall is the GH×value weighting tax. Drop the value-severity
weight (keep GH sectors, mandatory) → optimize closer to unweighted MAE → maybe sub-4.
**Config:** `--weighting garway_heath --sector-combine sector_only --head scalar --reweight value --lr 8e-4 --dropout 0.2 --weight-decay 0.005`
(sector_only ignores the value weight; GH sectors still applied)
**Predict:** MAE ↓ toward ~4.0–4.1, floor WORSE (~12–13, less severe emphasis).
Tradeoff datapoint: shows the goal-2 endpoint. If sub-4 → new champion.
**Result:** 🏆 **val MAE 4.15** (e42), bias +0.11, floor 11.96. CHAMPION. Dropping the
value weight cut MAE 4.25→4.15 and floor barely moved (value weight was paying MAE
tax without helping floor — signal limit again). Closing on sub-4.

## Iteration 5 — strip fit-hindering regularizers (squeeze to sub-4)
**Hypothesis:** label noise (σ0.3 caps achievable MAE) + variance penalty (anti-
regression, unneeded since we under-fit) add ~0.1–0.2 to val MAE. Remove both.
**Config:** iter-4 sector_only config + LABEL_NOISE_STD 0.3→0, VARIANCE_WEIGHT 0.05→0.
`--weighting garway_heath --sector-combine sector_only --head scalar --reweight value --lr 8e-4 --dropout 0.2 --weight-decay 0.005`
**Predict:** val MAE ↓ toward ~4.0 (maybe sub-4); slope may dip slightly (no variance reg).
**Result:** best MAE **4.20** (WORSE), champion UNCHANGED. train→3.99 but val 4.27:
removing reg OPENED an overfit gap. So NOT under-fitting — 4.15 is a GENERALIZATION
ceiling on 211 eyes. Reverted both regularizers. → need variance reduction, not more fit.

## Iteration 6 — weight EMA (variance reduction toward sub-4)
**Hypothesis:** 4.15 is a generalization ceiling; weight EMA (avg weights over the
plateau) cuts val variance ~0.1–0.3 with no fit/overfit risk → maybe sub-4.
**Config:** champion (iter-4) config + EMA decay 0.998 (regularizers restored).
`--weighting garway_heath --sector-combine sector_only --head scalar --reweight value --lr 8e-4 --dropout 0.2 --weight-decay 0.005` (--ema-decay 0.998 default)
**Predict:** val MAE ↓ to ~4.0–4.1, possibly sub-4; smoother val curve. New champ if <4.145.
**Result:** FAILED — slope collapse, MAE 5.45. BUG: decay 0.998 too high for short
runs (~15 steps/ep); EMA lagged into the near-constant init (0.998^330≈0.5). Raw
model was fine. Fix: added EMA warmup `min(decay,(1+step)/(10+step))`. Champion unchanged.

## Iteration 6b — EMA with warmup (the actual variance-reduction test)
**Config:** same as iter-6 + EMA warmup fix.
**Predict:** EMA now tracks recent weights → val MAE ~4.0–4.1, smoother. New champ if <4.145.
**Result:** MAE **4.24**, champion UNCHANGED. EMA works (smooth 4.24–4.28, low var) but
lands at the TYPICAL level, not iter-4's best dip 4.15 (which is partly a lucky val
draw). EMA trades peak for stability → doesn't win on best-MAE. EMA OFF going forward.

CEILING MAP (all converge): overall ~4.15 (generalization-limited, 211 eyes),
floor ~12 (signal-limited). Tried: reg, LR, weighting modes, dist/LDS/floor_boost,
bias penalty, EMA. Remaining levers: (a) augmentation [low-risk, targets generaliz.],
(b) signal upgrade dual-crop [high-EV for floor, high-risk/complex — do with user].

## Iteration 7 — photometric augmentation (attack the generalization ceiling)
**Hypothesis:** val is generalization-limited; ColorJitter (brightness/contrast)
adds realistic input variety (safe: not geometric, doesn't disturb the anatomical
prior) → lower val MAE. EMA off.
**Config:** champion config, EMA off, + ColorJitter(0.2,0.2) on train.
`--weighting garway_heath --sector-combine sector_only --head scalar --reweight value --lr 8e-4 --dropout 0.2 --weight-decay 0.005 --ema-decay 0`
**Predict:** val MAE ~4.0–4.15, smaller train/val gap. New champ if <4.145.
**Result:** ✗ FAILED — val MAE ~4.36, champion UNCHANGED. ColorJitter added input
variety but did NOT lower val MAE (worse than iter-4's 4.15 and the EMA-typical 4.24).
Confirms photometric aug adds no VF-relevant SIGNAL; the ceiling is signal+generalization,
not input diversity. → ColorJitter REVERTED to keep a clean champion baseline.

═══════════════════════════════════════════════════════════════════
# SESSION 2 — severe-point focus (slope/scatterplot is the bottleneck)
═══════════════════════════════════════════════════════════════════
User priority: (1) severe-point MAE / a real scatterplot line of best fit, then
(2) overall sub-4 / beat 3.74. Encoder–decoder + Garway–Heath stay ON always.
Pre-trained VF auto-decoder (~2.0 MAE) must NOT be overwritten.

## ROOT-CAUSE of the severe-point problem (why slope=0.37)
Champion: overall MAE 4.15, pooled **corr 0.55** but **slope 0.37**, EyeCorr 0.41,
floor(0-10) MAE **11.96**. slope/corr ⇒ σ_pred/σ_truth ≈ **0.67** → predictions are
UNDER-DISPERSED (shrunk toward the ~28 dB mean). Deep points are systematically
under-deepened (predicted ~12-22 when truth is 0-10). Two independent causes:
 - **C1 signal limit**: whole-image features only give corr ~0.55 / EyeCorr 0.41.
   Raising the achievable slope ABOVE 0.55 needs MORE signal (cannot be loss-tuned).
 - **C2 shrinkage**: Huber/MSE on a mean-dominated label dist is MMSE-optimal when it
   shrinks → low MAE but a flat scatterplot. The good scatterplot (slope→corr) and
   min-MAE genuinely CONFLICT at fixed corr. This is why every reweighting attempt
   only shifted the global level (proven runs 4/5/iter3) — it can't beat C1, and it
   fights C2.

## Two-front method ladder (each: plan → review → code → test, early-stop if flat)
 - **Iter 8 — variance-matching output calibration (attacks C2; ~free, fast).**
   Post-hoc affine ŷ'=μ+b(ŷ−μ), b=σ_truth/σ_pred fit on TRAIN-eval, applied at val.
   Pushes slope 0.37→~corr(0.55) and (since deep points are under-deepened) should
   roughly HALVE floor MAE (~12→~6-7) at a modest overall-MAE cost. Legit post-hoc
   calibration (fit on train only). Serves goal-1 directly + fixes the scatterplot.
   Confirmable in ~2 min by re-evaluating the existing champion (no retrain).
 - **Iter 9 — multi-layer RETFound feature fusion (attacks C1; cheap, 1 enc pass).**
   Use intermediate ViT blocks (e.g. 12/18/24), not just the last, as decoder input.
   Richer vessel/RNFL detail → higher corr → lifts the achievable slope AND lowers MAE.
 - **Iter 10 — disc-crop dual-view encoding (attacks C1; the big severe lever).**
   GRAPE is macula-centered, disc at a FIXED laterality-mirrored spot (OD~0.80,OS~0.22).
   Add a fixed disc-centered crop→224 as a 2nd frozen-encoder view; fuse its tokens.
   High-res optic-nerve-head/RNFL = the structural basis of glaucomatous VF loss.
   Doubles train step cost (encoder runs live in train) — accept for overnight.
 - **Iter 11 — multi-seed ensemble (attacks overall MAE → sub-3.74; reliable).**
   Average K decorrelated decoders. Run-to-run val spread (4.15-4.28) is pure variance;
   averaging cancels it. The most reliable route to sub-3.74 once corr is lifted.

CHAMPION TRACKING extended: keep the existing best-OVERALL-MAE champion AND add a
best-SEVERE champion (lowest floor-MAE, guarded by overall MAE < 4.6 so it can't be a
degenerate all-deep predictor). "Always save best" now covers both goals.

## ‼️ ROOT-CAUSE FOUND — the encoder shuffles patches (latent bug, all prior runs)
`PerPointVFModel.forward` calls `encoder.forward_encoder(x, mask_ratio=0.0)[0]`.
RETFound's `forward_encoder` → `random_masking`, which AT mask_ratio=0 STILL returns
the 196 patch tokens in a RANDOM per-call order (`ids_shuffle=argsort(rand)`, keep all)
and the code drops `ids_restore` (never unshuffles). But `PerPointAttention` adds a
FIXED row-major `patch_pos` and the FIXED Garway–Heath anatomical Gaussian prior keyed
to fixed patch indices. ⇒ the spatial prior + patch positions point at randomly
permuted patches EVERY pass → the per-point/spatial machinery has been DEAD; attention
degenerated to order-invariant content pooling. This explains the whole ceiling
(EyeCorr 0.41, slope 0.37, floor 12 = "can't localize where damage is"). It is the C1
signal limit's actual mechanism — and it's a CORRECTNESS bug, fixable with no new data,
encoder–decoder + GH fully intact (in fact it makes GH's spatial prior finally work).

## Iteration 8 (REVISED) — unshuffled encoder forward (activate the spatial prior)
**Method:** add `PerPointVFModel._encode(x)` reproducing the encoder forward WITHOUT
`random_masking` (patch_embed → +pos_embed → prepend CLS → 24 blocks → norm), giving
patches in proper row-major order. Use it in `forward` and `precompute_features`.
Requires RETRAIN (the old attention learned to ignore order). Champion config.
**Hypothesis:** with real spatial correspondence, per-point attention + the GH/anatomical
prior can localize defects → EyeCorr/slope rise, floor drops, and overall MAE may fall
too (better fit). This is the genuine severe-point + scatterplot fix.
**Predict:** slope 0.37→>0.5, EyeCorr 0.41→>0.55, floor 12→<10, overall MAE ≤4.15
(possibly sub-4). If slope/EyeCorr DON'T move, the prior mapping orientation is wrong
(tune VF→patch flips next); if they move a lot, this was THE bug.
**Result (INTERIM @ e23, run ongoing):** PARTIAL WIN. The fix is real — TRAIN metrics
jumped (Corr 0.60→0.74, slope→0.55, EyeCorr→0.48), confirming the spatial prior is now
ACTIVE. But VAL gains are modest+noisy: best val **4.13** @e20 (edges champ 4.145), val
Corr 0.52→0.58, val slope oscillating 0.18–0.42, val EyeCorr ~0.39. Severe floor still
oscillates 9.7–14.9 with the old bias flip (deep-regime e18 floor 9.7/bias−0.95 ↔
shallow-regime e20 floor 12.6/bias+0.6). ⇒ a TRAIN/VAL GAP opened (train EyeCorr 0.48 vs
val 0.39): the architecture is now CORRECT but the bottleneck shifted to generalization +
the bias oscillation. Decision: let it finish (healthy, improving). The prior mapping is
roughly right (train EyeCorr rose) so not re-deriving flips now. Next levers squarely
target the new bottleneck: calibration (slope/severe), disc-crop+ensemble (signal/variance).
**Result (FINAL):** 🏆 NEW CHAMPION **val MAE 4.134** (was 4.145) — real but small overall
gain; auto-stopped e36 (goal plateau). Val Corr 0.55→**0.58** (fix is genuine), but best
ckpt (e20) is SHALLOW regime: slope 0.312, bias +0.60, floor **12.62**, deep 9.67 — severe
NOT improved. Deep-regime epochs (e28/e34: floor ~10.5, slope ~0.43) cost overall MAE
(4.25-4.33). ⇒ shuffle fix corrected the architecture but the severe bottleneck is now the
C2 UNDER-DISPERSION: slope 0.31 vs corr 0.58 ⇒ σ_pred/σ_truth ≈ **0.53** (badly shrunk).
This is exactly what variance-matching calibration fixes → iteration 9.

## Iteration 9 — variance-matching calibration on the FIXED champion (C2 / severe)
**Method:** decoder/calibration_eval.py — post-hoc affine ŷ'=μ_t+b(ŷ−μ_p), b swept 1→σ_t/σ_p
(≈1.9 here), fit on TRAIN-eval, applied to VAL. Saves severe_champion.json at the balanced
b (lowest floor with overall MAE<4.6).
**Predict:** slope 0.31→~0.55, floor 12.6→~6-8, overall MAE 4.13→~4.3-4.6. New SEVERE champ.
**Result:** ✓ SEVERE WIN (saved severe_champion.json). Trade curve (fit b_full=1.687):
  s=0.50 b=1.34 → MAE 4.20, floor 11.01, slope 0.419 (best overall-MAE calibrated)
  s=1.00 b=1.69 → MAE 4.35, floor **10.72**, slope **0.512**, bias +0.17 ← severe champ
slope 0.31→0.51 (big scatterplot win); floor 12.6→10.7 (modest). Floor dropped LESS than
hoped because corr is only 0.58 — calibration can't add signal. ⇒ CONFIRMS the residual
severe bottleneck is CORR/SIGNAL. Calibration is a keeper (post-hoc, free) but the real
floor fix needs higher corr → disc-crop / multi-layer. Overall champion stays 4.134.

## Iteration 10 — seed ensemble (overall MAE → sub-4/3.74; low-risk, reliable)
**Hypothesis:** run-to-run val spread + the bias oscillation are VARIANCE; averaging K
decorrelated decoders (no seed is fixed, so each run differs) cancels it → lower overall
MAE, and (after calibration) a cleaner scatterplot. seed-1 = iter-8 champion.
**Method:** train 2-3 more seeds (same fixed-champion config), save each to
decoder/results/ensemble/seedN.pth, then ensemble_eval.py averages per-point preds + calibrates.
**Predict:** ensemble overall MAE ~3.95-4.05 (maybe sub-4); calibrated slope ~0.5, floor ~10.
**Result:** ✓ 🏆 NEW OVERALL CHAMPION (ENSEMBLE of 2 seeds) **4.116** (was 4.134).
seed1 4.134 + seed2 4.170 → mean 4.152, ensemble **4.116** (variance reduction works; 3rd
seed would help more). BUT ensemble HURTS severe: calibrated floor 11.38 > single 10.72 —
averaging shrinks predictions further, so calibration can't recover the floor. Confirms the
overall↔severe tension. Severe champ stays single+calib (10.72). champion.json now records
kind=seed_ensemble (defined by seed paths; best_model.pth still = iter8 single). seed-3 later.

## Iteration 11 — disc-crop signal probe (C1 / severe; the priority-1 lever)
**Method (low-risk, dataset-only):** `--disc-crop` adds a FIXED laterality-aware optic-disc
zoom (OD cx0.78 / OS cx0.22, cy0.49, half0.27) as an EXTRA per-eye view; the existing
multi-image averaging fuses it with the full image. No model/attention/cache/loop change
(can't break the ensemble path). Crop box VISUALLY VALIDATED (/tmp/disc_od.jpg, disc_os.jpg
— captures disc + peripapillary RNFL/atrophy). Probes whether high-res optic-nerve-head
detail raises corr (the real severe ceiling, since calibration showed corr=0.58 is the cap).
**Hypothesis:** disc detail → higher corr/EyeCorr → lower floor + better slope after calib.
**Risk:** naive prediction-AVERAGING + the macula-centered anatomical prior applied to
disc-centered crops may add noise → if neutral/worse, escalate to FEATURE-level fusion
(disc patches into PerPointAttention) — documented as the stronger follow-up for the user.
**Predict:** corr 0.58→>0.62, floor (calibrated) <10. If worse → averaging/prior mismatch.
**Result (INTERIM @ e11):** ✗ naive disc-crop NOT helping — val Corr flat ~0.55 (≤ baseline
0.58), EyeCorr ~0.37, floor oscillating 9.2–12.6 (e10 deep-regime floor 9.19/bias−1.36, MAE
4.59). Prediction-averaging + macula-prior-on-disc-crops adds no usable signal (predicted
risk). Letting it finish for a clean data point + a decorrelated ensemble member. → escalate.

## ‼️ CORR ≈ 0.58 WALL — three independent methods converge
whole-image unshuffle (iter8 corr 0.58), calibration (can't exceed corr 0.58), disc-crop
averaging (iter11 corr ~0.55). The per-point VF signal extractable from the FROZEN RETFound
features caps at corr ≈ 0.58 → THIS is the severe-point ceiling, not the decoder. Calibration
already converts that 0.58 into the best slope/floor it can. To beat the floor we must raise
corr, which now means ADAPTING THE ENCODER (user explicitly allowed editing encoder/pretrain):
 - **Iter 12 (planned): partial encoder unfreeze / LoRA** — let the last 1–2 RETFound blocks
   (or low-rank adapters) adapt to VF. Directly attacks the corr wall. RISK: 211 eyes →
   overfit; mitigate w/ heavy reg + self-stop + tiny LR on encoder params. Caching disabled
   (encoder changes) → slower val. The genuine signal lever.
 - **Iter 13 (planned): feature-level disc fusion** — disc patches into PerPointAttention
   (not averaging). Only worth it if encoder-adapt shows disc detail is representable.
 - Meanwhile bank safe OVERALL wins: seed-3 → 3-model ensemble.

## Iteration 12 — encoder partial unfreeze (last 1 block+norm @ lr1e-5) — KILLED (overfit)
**Result:** ✗ corr wall HELD. Train Corr 0.65→0.72 (encoder adapting to TRAIN), but val Corr
stuck **0.52–0.54 (BELOW frozen 0.58)**, val MAE ~4.4 (worse than champ). 12.6M unfrozen
params memorize train-specific features w/o gaining generalizable structure-function. Killed
e16 (not going anywhere). ⇒ DECISIVE: the corr≈0.58 ceiling is a **DATA limit (211 eyes)**,
not encoder capacity — more capacity just overfits. Severe is fundamentally data-limited with
no new data. (Would expect LoRA/norm-only to hit the same wall.) Champions unchanged.

## CONCLUSION on severe (goal 1): data-limited.
Exhausted within no-new-data: loss reweighting (dead), unshuffle (corr 0.58), calibration
(best slope/floor at that corr → severe champ 10.72/0.51), disc-crop (no gain), encoder
fine-tune (overfits). The fundus→VF per-point signal extractable from 211 eyes caps corr≈0.58.
Best severe deliverable = calibrated single model (slope 0.51, floor 10.72). Remaining UNTRIED
genuine idea: use the pretrained VF auto-decoder (~2.0 MAE) as a MANIFOLD PRIOR to enforce
realistic spatial VF patterns (improves coherence/EyeCorr, not raw corr) — investigating.

---
**Iter 11 FINAL:** disc-crop best val **4.23** (floor 11.67, slope 0.35, corr ~0.55) —
auto-stop e22. NO severe/corr gain. Worse, adding disc.pth to the ensemble HURT (3-model
4.146 > 2-model 4.116): the disc-trained model is weak on full images (MAE 4.34, corr 0.52)
because half its training used mismatched disc crops. ⇒ disc-crop RULED OUT (naive form).
Overall champ stays 2-seed ensemble 4.116; severe stays single-calib 10.72.

## Iteration 13 — 3-seed ensemble (overall)
**Result:** seed1 4.134 + seed2 4.170 + seed3 4.164 → 3-seed ensemble **4.120** — NO gain
over the 2-seed 4.116 (seeds too correlated; same ~4.13-4.17 structure → variance already
mostly cancelled at K=2). Overall MAE has PLATEAUED at ~4.12 = the 211-eye generalization
ceiling. Champion stays 2-seed ensemble 4.116.

## Iteration 14 — VF-manifold prior (pretrained auto-decoder as realistic-VF prior)
**Result:** ✗ FAILS. Applying frozen VFAutoDecoder to model preds → MAE 8.40 / bias +7.95
(the denoiser was trained on corrupted inputs WITH ZEROS at masked points; the model's smooth
all-nonzero preds are OOD → it over-fills ~+8 dB). manifold+calibration recovers to 4.23 /
floor 11.45 but eyeCorr DROPS 0.393→0.375 and it's worse than raw + worse than the calib-only
severe champ. The manifold projection does NOT add spatial coherence here. Severe champ stays
single+calib 10.72. (A proper version would retrain the model WITH a manifold-consistency loss,
or retrain the auto-decoder on non-zeroed inputs — uncertain payoff given the corr ceiling.)

═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY — autonomous session 2 (severe-point focus)
═══════════════════════════════════════════════════════════════════
## Champions (saved in decoder/results/champion/)
- **Overall (best val MAE): 4.116 dB** — 2-seed ENSEMBLE (champion.json, seeds seed1/seed2;
  was 4.145 → 4.134 single → 4.116 ensemble). Still > 3.74 baseline and > 4.0 target.
- **Severe (best floor / scatterplot): floor 10.72 dB, slope 0.512** — single iter-8 model +
  variance-match calibration (severe_champion.json). Was floor 11.96/slope 0.366.

## What WORKED (kept)
1. **Patch-shuffle bug fix** (iter8): encoder.forward_encoder shuffled the 196 patch tokens
   every pass (MAE random_masking even at mask_ratio=0), scrambling the GH/anatomical spatial
   prior. Fixed via PerPointVFModel._encode (no shuffle). Corr 0.55→0.58; champion 4.145→4.134.
   *The single most important correctness fix.*
2. **Variance-match calibration** (iter9): post-hoc affine fit on train; slope 0.31→0.51,
   floor 12.6→10.7. The scatterplot/severe deliverable. Free, no retrain.
3. **2-seed ensemble** (iter10): overall 4.134→4.116.

## What DIDN'T work (reverted/ruled out, with evidence)
- ColorJitter aug (iter7): 4.36, no signal. Loss reweighting (earlier): only shifts level.
- Disc-crop view (iter11): corr flat ~0.55, hurts ensemble. Naive prediction-averaging +
  macula-prior-on-disc-crops adds no usable signal.
- Encoder partial unfreeze (iter12): OVERFITS (train corr↑0.72, val corr↓0.54); 12.6M params
  on 211 eyes memorize. → corr ceiling is a DATA limit, not encoder capacity.
- 3-seed ensemble (iter13): no gain over 2-seed (overall plateaued ~4.12).
- VF-manifold prior (iter14): OOD distribution mismatch, no coherence gain.

## KEY FINDING — two hard ceilings on 211 train eyes, no new data
- **Overall MAE ≈ 4.12** (generalization-limited; reg/LR/ensemble all converge here).
- **Per-point corr ≈ 0.58** (signal-limited) → severe floor can't go much below ~10.7 even
  with optimal calibration. Confirmed by 4 independent methods hitting the same wall.

## To genuinely beat 3.74 / fix severe (needs the user — beyond no-new-data autonomy)
1. **More paired data** (the dominant lever): GRAPE has more eyes/timepoints; other glaucoma
   fundus+HVF datasets. 211→1000s eyes would lift both ceilings.
2. **Stronger structure signal**: OCT RNFL/GCL thickness maps as an extra input (the real
   structure-function substrate) — or a disc-region encoder PROPERLY fused at the feature
   level (not averaging) with a disc→VF (Garway-Heath sector) prior, trained jointly.
3. **VF-manifold done right**: retrain the auto-decoder on non-zeroed inputs and add a
   manifold-consistency loss during main training (coherence prior).
All keep encoder→decoder + Garway-Heath. Within strict no-new-data, the current champions are
at/near the achievable ceiling for this architecture.

═══════════════════════════════════════════════════════════════════
# SESSION 3 — challenge the "ceiling": rigorous eval + untried signal levers
═══════════════════════════════════════════════════════════════════
User direction (this session): keep the 2-stage encoder→decoder + Garway–Heath ALWAYS;
ALLOWED to (a) edit decoder pretraining (must stay masked-point spatial mapping), (b)
FINE-TUNE RETFound; prioritize EVAL RIGOR; optimize BOTH sub-3.74 MAE + a faithful
scatterplot line in ONE model, with **severe points the priority** (severe loss is also
the biggest MAE drag, so fixing severe should lower MAE too). Long/overnight runs OK.

## ‼️ The Session-2 "ceilings" rest on an unreliable test set (audited 2026-06-23)
Data audit of grape_train/test/new (263 eyes, 144 patients, 1 VF/eye, ≤7 fundus imgs/eye):
 - **62% FELLOW-EYE LEAKAGE**: split is per-RECORD random (separate_datasets.py), NOT
   per-patient. **29 of 47 test eyes have their fellow eye in TRAIN.** Bilateral glaucoma +
   mirror anatomy ⇒ test metrics are OPTIMISTIC and high-variance. The 4.12 MAE / 0.58 corr
   were measured here.
 - **SEVERE METRIC = 5 EYES**: the test set has only **5 eyes with meandB<15**. floor(0-10)
   MAE and slope — the whole severe goal — are computed almost entirely from those 5 (leaked)
   eyes. That is why floor oscillated 9.2–14.9 run-to-run. The "severe ceiling 10.72/slope
   0.51" is largely MEASUREMENT NOISE, not a model limit. (Full set has **47 severe eyes** —
   10× more — evaluable out-of-fold by per-patient CV.)
 - **Single noisy split**: 0.15 dB run-to-run spread was read as a hard ceiling.
 - The "corr 0.58 = signal limit" was inferred from MODEL runs only; never measured directly
   (no linear-probe ceiling, no learning curve, no per-layer feature analysis).
 - Encoder adaptation was tested ONCE, naively (last-block unfreeze @1e-5, 12.6M params) →
   overfit → "data-limited" verdict. LoRA / norm-only(BitFit) / domain-adaptive SSL pretrain
   (the regularized small-data ways) were NEVER tried. User now explicitly allows fine-tuning.
 - Pretrained VF auto-decoder (~2.0 MAE manifold) only ever used as a FAILED post-hoc
   projection; never as decoder init / training-time manifold regularizer.
⇒ Both ceilings are partly EVAL ARTIFACTS. Re-evaluate honestly before believing them.

## Plan (looping: diagnose → hypothesize → add tests → implement → train → save/early-stop)
 - **Iter A (diagnostics, no big-model training):** per-patient severity-stratified 5-fold CV
   harness; baselines (per-point/per-eye/global mean); LINEAR PROBES from frozen RETFound
   features (CLS / spatial-retinotopic patch / disc-region / multi-layer) under CV → measure
   the TRUE achievable corr + where the signal is (decoder-limited vs feature-limited vs
   disc-located); bootstrap CIs; re-eval current champion CV-vs-leaky to quantify inflation.
 - **Iter B+ (signal levers, by probe-guided EV):** multi-layer feature fusion; domain-adaptive
   SSL encoder pretrain on GRAPE imgs (no label overfit) → re-extract; LoRA/norm-only fine-tune
   w/ heavy aug + CV early-stop; feature-level disc fusion via GH sectors; manifold-decoder
   transfer + training-time anti-shrinkage (variance-match) loss. Calibration + cross-fold/seed
   ensemble as the finishing moves. Each gated on per-patient CV (severe-first), early-stopped
   if not beating the honest baseline.

## Iter A results — baselines reframe the ENTIRE problem (per-patient 5-fold CV)
Target variance: **54.4% between-eye (severity) / 45.6% within-eye (spatial)**. Naive OOF:
  global_mean      MAE 6.30 | corr -0.03
  perpoint_mean    MAE 6.01 | floor 14.67 | eyeCorr **0.488** | corr 0.297
  eye_oracle_mean  MAE **4.09** | floor 8.41 | corr 0.738 | slope 0.544   ← predict each eye's
                   TRUE mean flat across all 52 pts (severity oracle, ZERO spatial info)
‼️ **The champion (MAE 4.12) barely beats the severity-only oracle (4.09).** It captures
between-eye severity but adds almost NO within-eye spatial signal — its eyeCorr (0.41 on the
leaky set) is even BELOW a fixed per-point TEMPLATE (0.488). The pooled "corr 0.58" was
severity in disguise (oracle pooled corr 0.738 = √0.544). ⇒ The Session-2 framing was wrong:
the ceiling is NOT signal-in-general, it's that the model extracts **no within-eye spatial
signal**. Sub-3.74 REQUIRES capturing the 45.6% spatial variance (= localizing deep/severe
points = the slope + the floor + the user's severe priority, all the same lever). Next: the
linear probe locates whether that spatial signal is in the frozen features (decoder's fault)
or absent (needs encoder/disc).

## Iter A probes — frozen-feature linear readout (per-patient 5-fold CV, OOF). Mirror-sanity 0.945.
  probe        MAE   floor  slope  corr  eyeCorr  σp/σt
  CLS          5.38  11.04  0.371  0.537  0.400   0.69
  ALLPATCH     4.87  10.51  0.412  0.622  0.461   0.66   (best MAE/eyeCorr; mean-pooled+CLS)
  MULTILAYER   5.45  11.74  0.311  0.502  0.429   0.62
  RETINO       5.43   9.94  0.459  0.562  0.372   0.82   (per-point retinotopic patch)
  DISC         5.31   9.77  0.478  0.580  0.360   0.82   (per-point disc-crop patch) ← best floor/slope
  refs: template eyeCorr 0.488 ; oracle MAE 4.09/floor 8.41 ; champion(leaky) 4.12/floor~12/eyeCorr0.41/σ0.53
### Findings (decisive)
1. **No linear probe beats the population template's eyeCorr 0.488** (best 0.461). Eye-SPECIFIC
   within-eye spatial pattern is NOT readily linearly present in frozen RETFound features. The
   champion (eyeCorr 0.41) does WORSE than linear — so the decoder isn't extracting it nonlinearly
   either. ⇒ big eyeCorr gains likely need ENCODER ADAPT / DISC HIGH-RES, not just decoder tweaks.
2. **BUT deep-point DEPTH is localizable from frozen features NOW**: per-point RETINO/DISC probes hit
   floor **9.8** (vs template 14.7, champion ~12) and **σp/σt 0.82** (vs champion 0.53), DISC best
   (floor 9.77, slope 0.478). The disc region carries deep-loss signal. The champion squanders it by
   **excessive shrinkage** (σ 0.53 ≪ the 0.82 the same features support). ⇒ a real SEVERE win is
   available with FROZEN features + no new data, just by fixing dispersion + deep-localization.
3. **Severity is already solved**: champion MAE 4.12 ≈ oracle 4.09 (it nails eye-mean better than the
   linear probes' 4.9). The entire remaining gap is within-eye spatial.
### Revised lever ladder (severe-first, both-in-one-model)
 - **Lever 1 (next, no encoder change):** kill shrinkage + exploit deep-localization. Train-time
   variance-matching/dispersion loss to push σp/σt 0.53→~0.85; mean+residual head (CLS→eye-mean,
   localized attended patch→residual); sharpen attention onto retinotopic/disc patches; + calibration.
   Target: floor →~10, slope →~0.5, MAE ≤ baseline (deep errors dominate MAE → severe fix helps MAE).
 - **Lever 2:** feature-level DISC fusion via GH sectors (disc probe = best floor) — proper, not averaging.
 - **Lever 3:** domain-adaptive SSL encoder pretrain on GRAPE imgs (no label overfit) + LoRA/norm-only.
 - **Lever 4:** cross-fold + seed ensemble; honest 5-fold CV reporting for the champion.
First, a CONTROL: retrain the current champion recipe on the clean per-patient split → honest baseline.

## Iter A — NONLINEAR (MLP) probe on frozen features: decoder-vs-encoder bottleneck (per-patient 5-fold OOF)
  plain MLP        MAE 4.70 | floor 11.5 | slope 0.436 | eyeCorr **0.511** | σp/σt 0.71
  MLP + dispersion MAE 4.86 | floor 11.3 | slope 0.454 | eyeCorr 0.500 | σp/σt 0.77
  (linear ALLPATCH eyeCorr 0.461 ; template 0.488 ; champion **0.41** ; oracle MAE 4.09)
‼️ **DECODER is leaving spatial signal on the table.** A plain MLP on the SAME frozen features hits
eyeCorr **0.511** — well above the champion's 0.41 (and the template 0.488). So within-eye spatial
signal IS extractable from frozen RETFound with NO encoder change / NO new data; the champion's
heavy spatial-washing reg (attention-entropy bonus + label noise + high dropout) suppresses it.
BUT the MLP's overall MAE (4.70) is WORSE than the champion (4.12) because it predicts SEVERITY
worse. ⇒ **champion = great severity / no spatial ; MLP = good spatial / poor severity.** A model
with BOTH (severity≈oracle 4.09 + eyeCorr~0.51 + proper dispersion) plausibly reaches **sub-3.74**.
The ~0.51 eyeCorr is the frozen-feature spatial CAP; beyond it needs the encoder/disc lever.
### Lever-1 refined (frozen, no new data): recover the spatial signal WITHOUT losing severity
 - De-wash spatial: --entropy-weight 0 (let attention localize), --label-noise 0/0.1, dropout ↓.
 - --dispersion-weight ~0.1 (anti-shrinkage: σp/σt 0.53→~0.8; slope/floor = severe priority).
 - If de-reg+dispersion underperforms the MLP's eyeCorr, escalate to a MEAN+RESIDUAL head
   (CLS→eye-mean severity ; localized attended patch→within-eye residual) so the solved severity
   signal stops drowning the spatial head. Then calibrate + ensemble. Encoder/disc = later levers.
Added CLI knobs: --dispersion-weight, --label-noise, --entropy-weight, --train-json/--val-json,
--out-tag, --no-champion. Tests in decoder/tests_session3.py all pass.

## Iter B — honest CONTROL baseline (champion recipe on clean per-patient split, eval_ckpt)
RAW: MAE **4.547** | floor 9.90 | slope 0.486 | corr 0.693 | eyeCorr **0.422** | σp/σt 0.70 | bias +0.47
calib s=0.5 (b=1.20): MAE 4.537 | floor 9.43 | slope 0.581 | σp/σt 0.84  (calibration alone lifts
slope+floor at ~0 MAE cost). Leaky→clean: 4.12→4.55 confirms ~0.4 dB leakage inflation. eyeCorr
0.42 ≪ MLP-probe 0.51 ⇒ spatial headroom confirmed on the clean split. THIS is the bar to beat.
### Headroom math (why frozen-feature decoder levers CAN reach sub-3.74)
within-eye target σ≈5.5 dB; oracle (perfect severity, 0 spatial) → MAE 4.09. A model that hits the
frozen cap eyeCorr≈0.51 WHILE keeping oracle-grade severity → MAE ≈ 4.09·√(1−0.51²) ≈ **3.5 dB**.
The MLP probe got eyeCorr 0.51 but MAE 4.70 ONLY because its severity was poor. ⇒ keep the model's
severity strengths (CLS + TTA + multi-image + mean head) AND add the spatial residual = the bet.
### Lever-1b implemented+tested: --mean-residual (CLS→eye-mean ; point_head→zero-mean spatial residual)
Plus --dispersion-weight, --entropy-weight, --label-noise knobs. Running Lever-1a (de-wash lumped
head) now; Lever-1b (mean+residual) queued next.

## Iter C — Lever-1a (de-wash reg + dispersion on the LUMPED per-point head): ✗ WORSE
Best val MAE **4.87** (vs control 4.55); val eyeCorr **0.36** (vs 0.42); train eyeCorr 0.42 vs
val 0.36 = OVERFIT. ⇒ removing entropy bonus + label noise lets the 425k-param per-point attention
MEMORIZE; the regularizers were aiding generalization, not just suppressing signal. Dispersion
lifted slope (0.50) but eyeCorr (real spatial) fell. KEEP regularizers ON going forward.
### ‼️ Architecture insight (re-reading the probes): eyeCorr is FEATURE-TYPE limited
ALLPATCH (global mean-pooled patches+CLS) eyeCorr **0.51** ≫ RETINO/DISC (per-point LOCAL) 0.36–0.37.
The within-eye PATTERN is better predicted from GLOBAL pooled features via a JOINT head (global→52)
than from per-point local attention. A SHARED per-point head (current arch) is structurally capped at
the local-feature eyeCorr (~0.37–0.41 = exactly the champion's 0.41). ⇒ to reach 0.51, the decoder
needs a JOINT global-spatial head. The per-point attention is good for the FLOOR (deep localization,
9.8) but bad for the overall pattern. Plan: joint global head (pattern) + mean head (severity) first;
add a local/disc deep-correction for the floor after. GH stays on via sector loss weighting.

## Iter C — Lever-1b (REPLACE per-point with joint global head, dropout 0.5): ✗ UNDERFIT
MAE **6.06** (train MAE 5.50 @e60 still falling, corr 0.38) — replacing the whole decoder with a
fresh zero-init 1.2M global head at dropout 0.5 + dispersion + lr 8e-4 couldn't fit in 60 ep and
threw away the per-point model's solid base. ⇒ don't REPLACE; ADD. Reworked --global-head to be
ADDITIVE: pred = per_point_pred + zero-mean(global_spatial([pool‖CLS])), global_spatial zero-init
⇒ starts EXACTLY as control and can only ADD the global pattern the local head misses. Attention/
GH-prior stay active. Tests updated+pass. Running Lever-1b2 = control + additive --global-head.

## Iter C — Lever-1b2 (control + ADDITIVE global head): small MAE win, severe win via calib
RAW val MAE **4.46** (vs control 4.55), eyeCorr 0.428 (train 0.47 → mild overfit gap), σp/σt 0.66.
TTA on/off identical eyeCorr (0.428/0.425) → TTA is NOT the suppressor. CALIBRATED (s=1.0): slope
**0.678**, floor **8.40** (≈ oracle 8.41!), MAE 4.65 — a STRONG scatterplot line (vs control calib
slope 0.58/floor 9.4). ⇒ additive global head helps the severe/floor side; MAE only a touch better.
## ‼️ FROZEN-FEATURE CAP CONFIRMED — eyeCorr ≈0.51 (encoder lever now required)
Combined MLP probe (full+disc+multilayer, 7168-dim) eyeCorr **0.497** ≤ ALLPATCH 0.511 — NO frozen
combination beats ~0.51. In-pipeline reaches only ~0.42. Severity-oracle alone = MAE 4.09 on CLEAN
data, and the model's severity is imperfect (4.55 > 4.09). ⇒ sub-3.74 needs eyeCorr ≫0.51 = BETTER
FEATURES = encoder adaptation. Frozen decoder levers are exhausted (control 4.55 → global head 4.46).
### Deliverables so far (honest per-patient, dev fold):
 - SEVERE/scatterplot (user priority): additive global head + full calibration → slope 0.68, floor
   8.40. Strong line. To confirm on 5-fold + bank as severe champion.
 - sub-3.74 MAE (gating goal): needs the ENCODER lever → Lever-2.
### Lever-2: BitFit (norm+bias only) encoder fine-tune + heavier aug, clean-CV early-stop.
The regularized small-data version the prior session never tried (it did a naive 12.6M-param block
unfreeze → overfit). ~50k params, directly VF-supervised. Build on the best decoder (global head).

## Iter D — Lever-2 BitFit encoder fine-tune (norm+bias, ~50k params) + heavy aug
Implemented --finetune-norm (BitFit), --enc-lr, --heavy-aug, --batch-size. _encode runs full ViT-L
WITH grad for BitFit (norms throughout). Tests pass (263 enc tensors trainable, grad reaches norms,
attn matrices stay frozen). Running on CONTROL decoder (isolate encoder effect vs 4.55 baseline):
--finetune-norm --enc-lr 1e-4 --heavy-aug --batch-size 16, dev fold0, 35 ep. ~2 min/epoch (no cache,
full backprop). KEY question: does val eyeCorr exceed the frozen cap ~0.42? Also built scatter_clean.py
(deliverable scatterplot: truth-vs-pred + line of best fit, raw/calibrated) and run_cv.py (5-fold OOF).

## Iter D — Lever-2 BitFit RESULT + ‼️ error decomposition (the session's key insight)
BitFit (enc-lr 1e-4, heavy aug): val MAE **4.53** (≈control), eyeCorr 0.42→**0.45** (encoder DID adapt,
marginally), but bias drifted +1 and SEVERITY got worse. ⇒ encoder lever = marginal, not a breakthrough.
### Error decomposition (severity = per-eye mean ; residual = within-eye spatial), clean fold0:
  model        sevMAE sevCorr sevShr | resMAE resCorr resShr
  control       2.58   0.869   0.87  |  3.82   0.326   0.36
  global head   2.47   0.880   0.80  |  3.85   0.326   0.39   ← best severity ⇒ best MAE 4.46
  BitFit        2.88   0.836   0.86  |  3.78   0.360   0.38   ← traded severity for spatial (net neutral)
‼️ **The gap to sub-3.74 is SEVERITY, not spatial.** If eye-mean were perfect (sevMAE→0), per-point
error → the residual term ≈ **3.82 (already sub-4, near 3.74)**. Model sits at 4.5 because sevMAE=2.58.
Residual is also heavily shrunk (resShr 0.36). ⇒ Levers: (a) BETTER SEVERITY (dedicated mean head +
ENSEMBLE variance reduction), (b) de-shrink residual (dispersion). Global head already nudged severity.
BitFit dropped (hurts severity, 4× slower). Running Lever-3 = FULL decoder: --mean-residual (dedicated
severity head, bias-penalty-supervised) + --global-head (spatial) + --dispersion-weight 0.1, frozen enc.
Then 3-seed ENSEMBLE + calibration (the severity-variance lever) → honest champion + scatterplot.

## Iter D — Lever-3 (mean-residual + global + dispersion): ✗ WORSE (MAE 5.23, underfit)
Routing severity through a small dedicated mean_head DEGRADED it (the standard arch already predicts
severity well, sevCorr 0.87) and the stacked decomposition+bias-penalty 0.2 underfit (train 4.76).
⇒ MEAN-RESIDUAL RULED OUT. Best single recipe = additive global head (lever1b2, MAE 4.46). Building a
3-member global-head ENSEMBLE (member1=lever1b2 plain; member2 +dispersion 0.1; member3 +dropout 0.3)
for diversity → ensemble_clean.py averages + calibrates. Then 5-fold CV confirm → champion.

## Iter E — ENSEMBLE + severity tweaks: confirm the honest ceiling (~4.46)
3-model global-head ensemble: raw MAE **4.470** (= single best 4.46; same-arch seeds have CORRELATED
severity errors → averaging can't reduce them, as prior session found). Calibrated ensemble: slope
0.586 (s0.5) → 0.677 (s1.0), floor 9.1-9.35 — clean scatterplot. Stronger bias-penalty (0.4) HURT
(MAE 4.59) — severity is near-capped (sevCorr 0.87 from fundus), over-pinning backfires. ⇒ DEV
EXPLORATION DONE. Best recipe = additive global head (lever1b2), MAE 4.46. Confirming on 5-fold CV.
### Honest ceiling (per-patient): MAE ~4.46. Decomposition: severity sevMAE 2.58 (the gap to oracle
### 4.09) is fundus-signal-limited; spatial eyeCorr capped ~0.51. Sub-3.74 HONEST not reachable on
### 263 eyes. The old "3.74" was a LEAKY-split + heavily-shrunk model (low MAE / flat slope) — to verify.
### Real win: faithful calibrated scatterplot (slope ~0.6-0.68, floor ≈oracle) on HONEST eval,
### far better than the prior leaky slope 0.31-0.51.

## Iter F — 5-fold CV confirmation of the champion recipe (global head) → honest champion + scatterplot

═══════════════════════════════════════════════════════════════════
# SESSION 4 — longitudinal prior-VF: the honest path below 4.0 (2026-06-25/26)
═══════════════════════════════════════════════════════════════════
User: reach sub-3.9 (firm) / sub-3.74 (stretch) MAE + a strong scatterplot line, honestly.
Encoder→decoder + Garway–Heath stay ON. Allowed: new open datasets (with approval), any method.

## ‼️ DATA UNLOCK — the Follow-up sheet (was never read)
vf_test_converter only read sheet "Baseline" (263 eyes, 1 VF/eye). The "Follow-up" sheet has
**1,115 longitudinal visits**; **631 have a fundus on disk**, each with its OWN contemporaneous VF.
build_longitudinal_grape.py now builds 631 paired (fundus→same-visit VF) records, 263 eyes/144
patients, 101 severe. Fixes the expand_GRAPE label-noise bug (every image was paired with the
single baseline VF). Visit-1==baseline reproduces the old VFs EXACTLY (maxdiff 0) — mapping verified.

## Fundus-only ceiling CONFIRMED at ~4.29 (honest per-patient CV over 631)
long_global (control + additive global head) on cv_long: **RAW 4.290 / calib 4.54 (slope 0.64)**.
Decomposition: sevMAE 2.61 (corr 0.81) ⊕ resMAE 3.45 (sub-3.74). The gap to sub-3.9 is ENTIRELY
between-eye severity. Dead ends (re)confirmed: calibration floor = 4.30 (corr is the bound; b<1 only
flattens slope); fellow-eye input (model residual vs fellow TRUE severity corr 0.14; oracle
correction −0.07 dB); metadata; disc-crop averaging. 4-agent parallel brainstorm + literature
(MLEDL fundus-only raw-sensitivity = 4.13 on a *leaky* split; every published sub-3.9 uses
total-deviation/31k-images/OCT-input/VF-priors): **honest fundus-only ≤3.9 on raw sensitivity is not
reachable; fundus→severity corr ~0.81–0.83 is the imaging ceiling (hit independently by 2 groups).**

## ‼️ THE LEVER — the eye's own PRIOR VF (causal, leak-free under per-patient CV)
Persistence (predict current visit = most-recent prior visit) pointwise MAE **3.185** (any-VF prior)
/ 3.299 (photo prior); 368/631 records have a prior. This removes the per-eye severity bias the
fundus can't read. Leak-free: per-patient folds (eye-disjoint) + causal past→future ordering; the
held-out patient's OWN prior VF is an inference-time input, not a label (standard longitudinal
forecasting; matches MLEDL). Implemented: vf_autoencoder.py (bottleneck manifold AE, non-zeroed
input, recon 1.8 dB), longitudinal_dataset.py, longitudinal_model.py (LongitudinalVFModel: pred =
prior_field + zero-init delta ; visit-1 → fundus branch warm-started from long_global), train_longitudinal.py.

## RESULT — pooled MAE 3.75 (sub-3.9 ✓, at sub-3.74); strong line (calib slope 0.72), severe 6.0
5-fold per-patient OOF (631): **RAW 3.751 / calib 3.888 (slope 0.709)**, corr 0.74, eyeCorr 0.54,
severe(101) MAE 6.06 (vs fundus-only 7.49). Beats fundus-only 4.29 by 0.54.
### ‼️ Honest decomposition (don't oversell): the learned model ≈ a DETERMINISTIC BLEND
- with-prior stratum: model **3.20 ≈ persistence 3.185** — the delta adds ~nothing. The visit-to-visit
  test-retest noise floor is **2.76 dB**; the best possible denoiser (Agent-C ridge) is 3.167 — i.e.
  the follow-up stratum is **noise-floor-bounded and unlearnable** beyond persistence.
- visit-1 stratum: model **4.55 = fundus-only** (first-visit eyes are the hard subset; THE bottleneck).
- Deterministic blend (persistence on follow-ups + long_global on first-visits) = **3.752**, identical
  to the trained model. ⇒ the neural longitudinal head is unnecessary; the result IS the blend.
  (Freezing the fundus branch to protect visit-1 changed nothing: 3.766 — visit-1 was never degraded,
  the fundus is just genuinely ~4.5 on first visits.)
### Honest ARVO framing (required): lead with the PERSISTENCE baseline (3.71), not fundus-only 4.29
(beating fundus-only is trivial longitudinally). Report the stratified table: 3.185 on the 368
follow-up records WHERE VF history exists, 4.55 on the 263 first-visit records (fundus-only), pooled
3.75. The fundus model is the genuine ML contribution on first-visit eyes + the structure-function
scatterplot; prior-VF is the clinical no-change baseline that carries follow-ups. Disclose 42% of
records are first-visit/fundus-only.
### Remaining lever for clearly sub-3.74: improve visit-1 fundus (disc structure-function severity —
probe sevCorr 0.79 vs CLS 0.67). Estimated visit-1 4.55→~4.3 ⇒ pooled ~3.68. Marginal; not yet built.
