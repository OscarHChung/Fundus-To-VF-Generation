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
**Result:** _pending_
