# Fundus-only model improvement playbook (self-correcting loop)

**Audience:** a coding agent/session. **Mode of work:** implement one method at a time behind a flag,
run the fast dev loop, check the numeric gate, and **self-correct or revert** using the decision trees
below. Only promote a change to the full 5-fold CV once it clears the fast gate. Never keep a change
that fails its full-CV gate.

**Hard constraints (do not violate):** the model stays an **encoder–decoder, two-stage** design with a
**pretrained RETFound encoder** + **pretrained VF decoder**; **inference is fundus-only** (no prior VF,
text, or OCT at test time). Longitudinal/multi-visit data may be used in **training only**. Adapting
RETFound with LoRA still counts as "pretrained RETFound". Open data only (UWHVF + GRAPE).

---

## 0. Objective / definition of done
Beat the current fundus-only baseline on the leak-free per-patient 5-fold CV, primarily on **slope**
(line of best fit) and **MAE**, without degrading severe-band MAE.

- **Baseline (`long_global`, `decoder/results/auto/long_global_cv.json`):** raw MAE **4.290**,
  slope **0.473**, Pearson r **0.657**, per-eye r 0.447, severe(<15 dB eyes) MAE ≈ 7.49.
- **Definition of done (full 5-fold OOF, raw sensitivity):** MAE **≤ 4.05** AND slope **≥ 0.65** AND
  severe-band MAE **≤ baseline** AND r **≥ 0.70**. A partial win (e.g. slope ≥0.65 at MAE ≈ baseline)
  is still worth keeping — record it; do not discard a slope win just because MAE is flat.

## 1. The scoreboard (single source of truth for "is this working?")
All numbers come from `decoder/diagnostics.py::pooled_metrics` / `stratified_report` on **out-of-fold**
predictions, scored against the **RAW observed VF** (never against denoised labels — see Method B).
Keep a running table in `decoder/results/auto/improvement_log.md`:

| run tag | change | eval | MAE | slope | r | severe MAE | verdict |
|---|---|---|---|---|---|---|---|

Log every attempt, including failures and the reason. This log IS the "am I going in the right
direction" signal.

## 2. The self-correcting loop (how to work)
```
for each method (A → B → C, in order):
  1. implement behind a flag (default OFF = exact current behavior)
  2. write/ði run its unit test (fast, no encoder) — must pass before any training
  3. INNER LOOP (fast, ~minutes): train ONE fold (fold 0) or the grape_long fast split,
     evaluate, read the fast gate.
        - if fast gate PASS  → go to full CV
        - if fast gate FAIL  → consult the method's decision tree, make ONE targeted change,
                               re-run the inner loop. Cap at ~4 inner attempts.
        - if still failing after the cap → REVERT the method (flag OFF), log why, move on.
  4. FULL CV (slow): run all 5 folds, score pooled OOF, check the full-CV gate.
        - PASS → keep the flag ON as the new baseline for the next method; update the scoreboard.
        - FAIL → revert; log.
  5. STOP CONDITIONS you may hit mid-run and should act on:
        - train MAE ≪ val MAE and widening  → overfitting; stop, apply the overfit fix.
        - slope not moving after the loss change → σ/weighting wrong; stop, apply the slope fix.
        - loss NaN/spikes → lower LR, add/keep grad-clip (already 1.0), raise σ (BMC); stop & fix.
```
Iterate methods independently first; stack only what individually passed (Section 7).

## 3. Shared harness & guardrails
- **Fast inner loop:** train a single fold. If `run_cv.py` lacks a single-fold option, add
  `--folds 0` (mirror the `--folds` arg already in `train_longitudinal.py`), or use the fast split
  `data/vf_tests/grape_long_{train,val}.json` (produced by `python decoder/diagnostics.py split-long`).
  Use fewer epochs (e.g. 25) for inner-loop speed; full runs use the standard 60.
- **Full eval:** `python decoder/run_cv.py --tag <tag> --cv-dir decoder/results/cv_long --epochs 60 -- <recipe>`
  writes `decoder/results/auto/<tag>_cv.json` with pooled raw+calib metrics. Compare to `long_global`.
- **Everything behind a flag, default OFF.** A run with all new flags OFF MUST reproduce `long_global`
  within noise (sanity check before trusting any gain).
- **Fixed seeds & fixed folds** (`decoder/results/cv_long/` is frozen) so runs are comparable.
- **Checkpoints:** never overwrite `long_global_f*_best.pth`. New runs use new tags.
- **Git:** work on a branch; commit after each method that passes its full-CV gate, so a failed later
  method can be reverted cleanly.

---

## 4. METHOD A — Balanced MSE loss (the slope fix)  ★ do first, lowest effort
**Goal:** raise slope 0.47 → ~0.65–0.80 and cut severe-band MAE, at ~flat pooled MAE, by replacing the
mean-seeking Huber with a distribution-balanced loss (Ren et al., *Balanced MSE*, CVPR 2022 — BMC
variant, no label prior needed).

**Why our earlier attempt misfired:** it was stacked on top of the variance-matching penalty (which
double-counts spread) and `noise_sigma` was untuned. Fix both.

**Files:** `decoder/training.py` (`compute_loss`, loss-config globals), `decoder/run_cv.py` (pass the flag).

**Implementation sketch (BMC):** flatten the batch's valid (pred, target) pairs to 1-D vectors
`p, t` (apply the existing mask; keep laterality/query order irrelevant here since it's pointwise).
```
# logits[i,j] = -(p[i]-t[j])^2 / (2*sigma^2);  target label = i (the matching index)
logits = -(p[:,None] - t[None,:])**2 / (2*sigma*sigma)
loss_bmc = F.cross_entropy(logits, arange(len(p)))   # = BMC
```
- Keep the **Garway-Heath sector weighting** (multiply the per-point term as today) and the **per-eye
  CCC** term. **Disable the variance term** when BMC is on (set `VARIANCE_WEIGHT` contribution to 0).
- `sigma` = `--noise-sigma` (start 1.0). Optionally make it a learnable `nn.Parameter` (log-space,
  as in the BMC paper) — try fixed first.
- Add `--loss {huber,balanced_mse}` (default `huber` = current behavior).

**Unit test (`decoder/tests_session3.py` or new):** on synthetic data `t ~ N(0,5)`,
`p = 0.5*t + noise` (a deliberately regressed-to-mean predictor), one step of BMC must increase the
fitted slope of `p` vs `t` relative to MSE. Assert BMC recovers slope closer to 1 than MSE on a small
optimization. Also assert `balanced_mse` OFF ⇒ identical loss value to current Huber path.

**Fast gate (fold 0 val):** slope ≥ 0.58 AND MAE ≤ baseline_fold0 + 0.10.
**Full-CV gate:** slope ≥ 0.60 AND MAE ≤ 4.35 AND severe MAE ≤ baseline severe.

**Decision tree (self-correct):**
- slope barely moves → `sigma` too large (loss ≈ MSE): halve it (1.0 → 0.5 → 0.25). Confirm the
  variance term is actually disabled.
- loss NaN / unstable / MAE explodes → `sigma` too small or LR too high: raise `sigma`, lower LR to
  4e-4, keep grad-clip 1.0, add 1–2 epoch warmup.
- slope up but MAE regresses > +0.15 → reduce the CCC weight slightly, or blend: `loss = α·BMC +
  (1-α)·Huber` with α=0.5; tune α.
- severe MAE worse → increase `sigma` a touch (less aggressive balancing) or add mild deep-location
  weighting; re-check.

**Keep/revert:** keep if full-CV gate passes (expected). This is the foundation the other methods
build on — do not skip.

---

## 5. METHOD B — Per-eye target denoising (lower the label-noise floor)  ★ training-only
**Goal:** train against a cleaner estimate of each eye's true field (perimetric test-retest σ ≈ 2.4 dB
per point corrupts single-VF labels), so the model learns structure-function instead of noise.
Expected: MAE −0.1 to −0.3 and better slope/severe band. **Eval is unchanged — always score against
the RAW observed VF**, so the metric stays honest and comparable to Park/Huang.

**Files:** new `build_denoised_targets.py` (or extend `build_longitudinal_grape.py`), and the training
`Dataset` (in `decoder/training.py` / `decoder/longitudinal_dataset.py`) to load a `hvf_denoised`
field when a `--denoised-targets` flag is set.

**Implementation sketch:**
- Reuse the per-eye VF **timeline** already assembled in `build_longitudinal_grape.py::read_followup`
  (all visits per (subject, laterality) with `interval_years`).
- For each target record, for each of the 52 points: fit a **robust linear trend** (Theil–Sen or
  Huber regression) of sensitivity vs `interval_years` over that eye's visits, then **evaluate the
  trend at the TARGET visit's date** → the denoised target. This detrends progression; do **not**
  naively average (that biases toward the mean date).
- **Fallbacks:** eyes with < 3 visits, or an unstable/positive-progression fit, → use the raw VF (no
  denoising). Clip denoised values to plausible dB range; keep masked points masked.
- Alternative/augment: pass the raw field through the frozen `decoder/pretrained_vf_ae.pth`
  (`load_ae`) as a manifold denoiser and blend with the trend estimate. Try the trend first (simpler).
- Write `data/vf_tests/grape_longitudinal_denoised.json` (raw kept for eval; denoised used for train
  targets only).

**Unit test:** construct a synthetic eye with a known linear trend + added noise across 4 visits;
assert the denoised target at the target date is closer to the (noise-free) truth than the raw VF, and
that a <3-visit eye falls back to raw exactly.

**Fast gate (fold 0):** MAE ≤ baseline_fold0 (raw eval) AND slope ≥ baseline_fold0 slope.
**Full-CV gate:** MAE ≤ 4.20 (raw eval) AND slope not worse; severe MAE not worse.

**Decision tree (self-correct):**
- MAE worse → likely progression bias (didn't detrend to date) or over-smoothing: verify the trend is
  evaluated at the TARGET date; widen the fallback (require ≥4 visits); reduce blending with the AE.
- No change at all → denoised ≈ raw (labels barely moved): increase denoising strength (add the AE
  manifold blend; or robustify the trend). Check that `hvf_denoised` is actually being loaded (log a
  mean |denoised−raw| per batch; it should be ~1–2 dB, not ~0).
- Slope worse → the estimator flattened variance: lower AE blend weight; keep more of the trend slope.

**Keep/revert:** keep if full-CV gate passes; combine with Method A (denoised targets under BMC).

---

## 6. METHOD C — LoRA adaptation of RETFound (raise r = the real MAE+slope lever)
**Goal:** lift Pearson r from 0.66 toward ≥0.70 (fundus→severity ceiling ≈ 0.81), which lowers both
MAE and slope, by low-rank-adapting the frozen RETFound features (MAE-pretrained ViT features are
weak for regression). Expected: MAE 4.29 → ~4.0–4.1, slope → ~0.55–0.62. Highest overfit risk (n≈500
train/fold) — guard it.

**Files:** `decoder/training.py` (`PerPointVFModel.__init__` / `_encode`), small `LoRALinear` wrapper,
`decoder/run_cv.py` (flags). Keep the encoder otherwise frozen.

**Implementation sketch:**
- Minimal custom LoRA (avoid a new heavy dependency): `LoRALinear(base_linear, r, alpha, dropout)`
  computes `base(x) + (alpha/r) · dropout(x) @ A @ B`, with `base` frozen, `A,B` trainable
  (`A: in×r` ~N(0,·), `B: r×out` zero-init so it starts as a no-op).
- Inject into the **last K ViT blocks'** attention projections: wrap `encoder.blocks[-K:].attn.qkv`
  (and optionally `.attn.proj`). Optionally also unfreeze those blocks' LayerNorms.
- Flags: `--lora` (default off), `--lora-rank 8`, `--lora-blocks 8`, `--lora-alpha 16`,
  `--lora-dropout 0.1`. Give LoRA params their own (smaller) LR and **high weight decay**.
- Ensure `_encode` still omits the MAE random-patch-shuffle (unchanged). LoRA must be active in both
  train and eval (it's part of the model now); it's included in the saved checkpoint.

**Unit test:** assert (a) base RETFound weights have `requires_grad=False` and only LoRA A/B (+ chosen
norms) are trainable; (b) trainable-param count matches `r`, `K` expectation and is « full-block count;
(c) forward output shape unchanged; (d) at init (B=0) the encoded features equal the frozen-encoder
features (LoRA is a no-op at start).

**Fast gate (fold 0):** r ≥ 0.69 AND MAE ≤ baseline_fold0 − 0.05 AND (train MAE − val MAE) within a
sane band (overfit guard; compare to the baseline's own train/val gap).
**Full-CV gate:** r ≥ 0.70 AND MAE ≤ 4.15 AND slope ≥ 0.55.

**Decision tree (self-correct):**
- Overfit (train ≪ val, val worse than baseline) → lower `--lora-rank` (8→4), fewer `--lora-blocks`
  (8→4, last blocks only), raise weight decay, raise `--lora-dropout`, rely on 5-fold ensembling.
- No gain / underfit → raise rank (8→16) or blocks (8→12), also unfreeze the LoRA blocks' norms, or
  raise the LoRA LR.
- Unstable / diverges → lower LoRA LR, add warmup, confirm B is zero-init (no-op start).
- Helps r but MAE/slope flat → combine with Method A (BMC) which converts higher r into slope.

**Keep/revert:** keep only if the full-CV gate passes AND severe MAE is not worse; LoRA that only
helps mild eyes while hurting severe is not a win here.

---

## 7. Sequencing & stacking
1. **A (Balanced MSE)** — establishes the slope fix and becomes the loss for everything after.
2. **B (denoised targets)** under A — lowers the MAE floor.
3. **C (LoRA)** under A+B — the feature lever; BMC (A) turns the higher r into slope.
- After each individually passes, run the **stacked** config on full CV. If the stack underperforms
  the best single method (interactions), bisect: turn off the most overfit-prone piece (usually LoRA
  rank/blocks) and re-tune. Keep the best-scoring config as the new champion; write it to
  `decoder/results/auto/<champion_tag>_cv.json` and update `iterations.md` + the scoreboard.
- Stretch (only if A–C land and you want more): Tier-2 items from `RESEARCH_SYNTHESIS.md`
  (structure-function auxiliary head; severity+residual decomposition; manifold-latent decoding;
  multi-view resolution). Each gets its own flag + gate + decision tree in the same pattern.

## 8. Rollback & integrity rules
- Every method is a flag; the all-OFF config must reproduce `long_global`. If it doesn't, fix that
  before trusting any result.
- **Score only against the raw observed VF.** Denoised labels are a training device; reporting a
  denoised-eval MAE would be dishonest and is forbidden.
- Report raw AND variance-matched-calibrated numbers (no post-hoc calibration can improve MAE and
  slope together — it only moves along the frontier bounded by r).
- Keep the honest severity-stratified table with every pooled number (`stratified_report`); a pooled
  MAE that improves while severe MAE worsens is NOT a win.
- If a method fails its gate after the capped inner-loop attempts, revert it and log the reason in
  `improvement_log.md`. A smaller honest gain that survives full CV beats a large gain that doesn't
  replicate.
- Do not change the frozen folds, seeds, or the definition of the eval mid-project.

**Success = a new champion checkpoint set whose full 5-fold OOF beats MAE 4.29 / slope 0.47 per
Section 0, verified against the raw VF, with the scoreboard and iterations.md updated.**
