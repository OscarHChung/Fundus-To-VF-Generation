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
**Result:** _pending_
