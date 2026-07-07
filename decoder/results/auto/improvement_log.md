# Improvement log — fundus-only model (self-correcting loop)

Eval = leak-free per-patient 5-fold CV, scored against the **RAW** observed VF
(`diagnostics.pooled_metrics` / `stratified_report`). Definition of done (full 5-fold OOF):
MAE ≤ 4.05 AND slope ≥ 0.65 AND severe-band MAE ≤ baseline AND r ≥ 0.70. Slope-only wins at
flat MAE are kept and logged.

## Baselines (reference)

| scope | MAE | slope | r | severe MAE | notes |
|---|---|---|---|---|---|
| **long_global (full 5-fold OOF, raw)** | 4.290 | 0.473 | 0.657 | ~7.49 | the target to beat |
| long_global (full 5-fold, calib) | 4.536 | 0.637 | 0.656 | — | variance-matched line |
| **long_global fold-0 val (raw)** | 4.268 | 0.485 | 0.724 | 7.651 | fold-0 inner-loop reference (measured on `long_global_f0_best.pth`) |

Fold-0 reference strata: severe(n24) 7.651 · moderate(n24) 4.898 · mild(n61) 2.690. σp/σt=0.67.

## Scoreboard

| run tag | change | eval | MAE | slope | r | severe MAE | verdict |
|---|---|---|---|---|---|---|---|
| long_global | baseline | 5f OOF raw | 4.290 | 0.473 | 0.657 | ~7.49 | ref |
| **loraC** | **cached-LoRA champion, 5-FOLD OOF raw** | 5f OOF raw | **4.220** | 0.485 | 0.657 | 7.694 | modest MAE −0.07; r/slope flat; severe WORSE. Fold-0 gain didn't generalize (overfit on hard folds) |
| loraC calib | variance-match | 5f OOF calib | 4.480 | 0.628 | 0.659 | 7.332 | slope 0.63 but MAE 4.48 — neither raw nor calib hits 4.0 & 0.6 |
| (baseline) | Huber | fold0 raw | 4.268 | 0.485 | 0.724 | 7.651 | fold0 ref |
| bmc_s3 | BMC σ=3, 25ep | fold0 raw | 4.596 | 0.429 | 0.698 | 8.126 | ✗ FAIL (slope<baseline, MAE+0.33, severe worse, σp/σt 0.61) — σ too small; raise it |
| bmc_s7 | BMC σ=7, 25ep (killed @ep8) | fold0 raw | ~5.27 | ~0.42 | ~0.60 | — | ✗ FAIL (MAE stuck 5.2–5.4, r collapses 0.72→0.60, slope no better than σ=3) — high σ degrades the fit without buying slope |
| raw25 | baseline recipe, 25ep | fold0 raw | 4.433 | 0.422 | 0.711 | 8.260 | epoch-matched CONTROL for B (35 fewer epochs than the 60ep ref → +0.16 MAE) |
| denoise25 | +denoised targets, 25ep (best@ep18) | fold0 raw | 4.474 | 0.430 | 0.707 | 8.082 | ~ MARGINAL vs raw25: severe −0.18 ✓, slope +0.008 ✓, MAE +0.04 ✗ (epoch-confounded: best@18 vs raw@~24) |
| **lorac_f0** | **cached-LoRA r8×4blk, 40ep (TTA)** | **fold0 raw** | **4.199** | **0.536** | **0.729** | **7.666** | **✓ BEATS baseline on MAE −0.07, slope +0.05, r +0.005, severe tied — LoRA breaks the frozen r-ceiling** |
| **lorac8_f0** | **+warm-start +8blk +slope-sel (TTA)** | **fold0 raw** | **4.105** | **0.574** | **0.734** | **7.545** | **✓✓ beats baseline on ALL: MAE −0.16, slope +0.09, r +0.01, severe −0.11, eyeCorr 0.447→0.552** |
| lorac8_f0 +calib s=0.5 | variance-match calib | fold0 | 4.108 | **0.612** | 0.736 | — | slope ≥0.6 for ~free at r=0.734 (MAE +0.003) — only MAE gap to 4.0 remains |
| lorac16_f0 | rank 16, 8blk | fold0 raw(no-TTA best) | 4.193 | 0.554 | 0.725 | — | ✗ more rank OVERFITS (r 0.738→0.725); rank 8 better. Bottleneck = overfit (no-aug), not capacity |
| **m1m3_f0** | **M1 severity head + M3 EMA (lorac8 recipe)** | **fold0 raw TTA** | **4.206** | **0.552** | **0.721** | **7.612** | ~ beats BASE not lorac8. Sev de-shrunk (sev_shr 0.74→0.83) ✓ but EMA blurred spatial (res_corr 0.495→0.441). See SESSION-M1 section. Fold-0 severity is SATURATED (weak gate) |
| **m1sev_f0** | **M1 severity head, NO EMA** | fold0 no-TTA (interrupted@E24) | ~4.16 | ~0.56 | ~0.73 | — | best@E10; TTA eval + full CV TODO on faster box. Isolates M1 (spatially neutral). See SESSION-M1 section |

---

## Method A — Balanced MSE (BMC)  [in progress]

**Implementation (flag, default OFF):** `decoder/losses.py::balanced_mse_loss` (pure, unit-tested,
no encoder) + `training.py` `--loss {huber,balanced_mse}` / `--noise-sigma σ`. When ON: the pooled
batch's valid points go through BMC cross-entropy (Ren et al. CVPR'22); Garway–Heath sector
weighting and per-eye CCC are kept; the variance-match penalty is disabled (it double-counts spread).
Default OFF is byte-identical & σ-independent to the Huber path (test_balanced_mse_gating).

**Fast unit tests (pass):** `tests_method_a.py` — BMC de-shrinks a regression-dilution slope above
MSE and nearer the true slope of 1; larger σ ⇒ stronger de-shrink; helper shape/weights.
`tests_session3.py::test_balanced_mse_gating` — OFF ≡ Huber, ON differs.

**σ note (evidence, deviates from playbook prose):** an init-independent optimisation experiment on
the playbook's own BMC formula shows de-shrink strength GROWS with σ (σ scales with target spread;
pooled VF σ≈7.7 dB). So if slope barely moves, RAISE σ (the playbook's decision-tree said halve —
inverted for this formulation). σ search will start low and go up.

**Fast gate (fold-0 val):** slope ≥ 0.58 AND MAE ≤ 4.368 (baseline_fold0 + 0.10).

### Method A verdict: REVERT (flag stays default OFF)  ✗

Two fold-0 inner-loop attempts bracket the σ space and both fail the fast gate; the σ direction is
exhausted (σ=3 too weak, σ=7 degrades the fit), and the decision-tree "blend" branch's precondition
("slope UP but MAE regresses") is never met — slope never rises above baseline at any σ. Per the
capped-attempt rule, Method A is reverted. Root cause (evidence, not a tuning miss):

- **The encoder is frozen, so r is capped** (~0.72 fold-0 / 0.66 pooled). BMC only *redistributes*
  the existing signal into more spread; it cannot raise r. On this setup it de-shrinks *less
  efficiently than a plain linear rescale* while also lowering r (σ=7: r 0.72→0.60).
- **BMC is dominated by the baseline's own post-hoc variance-matched calibration**, which already
  reports slope **0.637 @ MAE 4.54** (long_global calib) with NO training change. Every BMC point
  (σ=3: 0.43@4.60; σ=7: ~0.42@5.2) lies *inside* that frontier on RAW slope AND RAW MAE, i.e. BMC
  improves neither raw slope nor raw MAE over the Huber baseline (raw fold-0 0.485 @ 4.27).
- Secondary: the best-checkpoint selection is min-val-MAE, which discards BMC's later higher-slope
  epochs — but this is moot because even BMC's best epoch is dominated.

**Implication for the plan:** the real slope+MAE lever is **higher r** (Method C — LoRA — unfreezes
encoder capacity; Method B — denoised targets — cleans label noise). BMC (kept in the codebase,
default OFF, unit-tested) becomes worth revisiting only *after* r is raised, exactly as the playbook's
"BMC converts higher r into slope" note anticipates. Proceeding to Method B, then C, then re-test the
BMC stack. The slope deliverable, meanwhile, is honestly reportable via the variance-matched calib row.

---

## Method B — per-eye target denoising (training-only)  [in progress]

**Implementation:** `build_denoised_targets.py` fits a robust per-point Theil–Sen trend over each
eye's visit series and evaluates it at the TARGET visit's date (interpolation → denoises test-retest
noise without biasing progression). Writes `grape_longitudinal_denoised.json` (631 keys; mean
|denoised−raw| = **1.24 dB**, 0 whole-eye fallbacks). `training.py --denoised-targets` swaps TRAIN
targets only; val/eval always score RAW. Unit tests: `tests_method_b.py` (denoise beats raw on
synthetic truth; <3-visit + sparse/masked fallbacks) and `tests_session3::test_denoised_target_swap`.

**Note on eval fairness:** the fold-0 gate compares to the **epoch-matched** raw25 control
(MAE 4.433 / slope 0.422 / severe 8.260 @25ep), NOT the 60-epoch reference — the epoch gap alone
costs ~0.16 MAE. Fast gate: denoise25 MAE ≤ 4.433 AND slope ≥ 0.422 AND severe ≤ 8.260.

### Method B verdict: MARGINAL — carry into the C stack, don't promote alone

denoise25 (best@ep18, TTA raw eval) vs the raw25 control: **severe 8.082 vs 8.260 (−0.18 ✓)**,
slope 0.430 vs 0.422 (+0.008 ✓), MAE 4.474 vs 4.433 (+0.04, within the epoch-18-vs-24 confound), r
flat. Denoising moved labels by only 1.24 dB and the improvement is small because — as with Method A
— the **frozen encoder caps r**, so label-noise cleanup isn't the binding constraint. B does not
clearly clear its fast gate alone (MAE not below control), so it is NOT promoted to a standalone full
CV. It mildly helps the severe band without hurting slope, so I'll retest it **stacked under Method C**
(the r-lever) and keep it only if B+C ≥ C. Proceeding to Method C.

**Infra note:** 17 GB RAM is saturated (swap full); training jobs OOM-die if ANY other Python
process runs concurrently (even light tests) — all encoder work is strictly serialized, one process
at a time. denoise25 was killed at ep22 but its ep18 best checkpoint was scored directly via
eval_ckpt (salvaged without a re-run).

## Method C — LoRA adaptation of RETFound  [implemented, default OFF]

**Implementation:** `LoRALinear` (base frozen; A~N(0,1/r), B=0 ⇒ no-op at init) injected into the
last K blocks' attention qkv via `inject_lora`; `training.py --lora --lora-rank/-blocks/-alpha/
-dropout/-lr`. The encoder is deep-copied so the shared `base_model` is never mutated. LoRA blocks
run with grad in `_encode`; `eval_ckpt.load_model` rebuilds the LoRA arch from ckpt metadata.
Unit test `tests_session3::test_lora_adapter` (A/B-only trainable, base frozen, no-op at init, shape
preserved, grad→B, base_model unpolluted). Memory-optimized: gradient checkpointing on the graphed
suffix blocks + periodic `torch.mps.empty_cache()` + train-eval subsample + skip the training-time
encoder deep-copy.

### Method C verdict: WORKS (via cached-prefix training) — beats the baseline ✓✓

The live-encoder LoRA path OOM'd on this box (per-step graph over a 1.2 GB encoder). The fix is
`decoder/train_lora_cached.py`: cache the frozen prefix once, FREE those blocks (~1 GB), warm-start
the decoder from long_global, and train only the LoRA suffix + decoder on cached features. This fits
memory and finishes in ~15–20 min. Fold-0 (TTA, RAW): **MAE 4.105 / slope 0.574 / r 0.734 / severe
7.545 / eyeCorr 0.552** — beats the baseline (4.268 / 0.485 / 0.724 / 7.651 / 0.447) on EVERY metric.
LoRA raises r above the frozen ceiling; with light variance-match calibration slope reaches 0.612 at
MAE 4.108 (nearly free at high r). Levers that helped: warm-start decoder, 8 LoRA blocks, slope-aware
selection. Rank 16 overfits (worse). Remaining gap to the 4.0/0.6 target is ~0.10 MAE, addressed by
restoring augmentation (multi-view cached prefixes) to cut the train/val overfit gap (~0.9 → ?).

## Environmental limits (why full CV was not reached)

- **Wall-clock:** background runs are killed at ~25 min. The frozen denoise25 (25 epochs) was killed
  at epoch 22 (~24 min). A **60-epoch** fold (needed to match the long_global baseline) therefore
  cannot complete, so the **full 5-fold CV @60ep — the definition-of-done eval — is infeasible here.**
- **Memory:** encoder-gradient training (Method C, the r-lever) OOMs at ~90 s (above).
- **Concurrency:** only ONE RETFound-loading process may run at a time; any second python process (a
  test, an eval, or a second training job) OOM-kills the first. Monitors/background waiters count.

## Honest best result

No config beat the baseline on the (locally-infeasible) full 5-fold CV. On the epoch-matched fold-0
signals the baseline is not beaten on raw MAE by A or B; A is dominated by calibration and B is
marginal. The binding constraint is the **frozen-encoder r-ceiling** (r≈0.66 pooled / 0.72 fold-0):
- post-hoc variance-matched calibration already delivers the slope headroom (baseline calib slope
  **0.637** @ MAE 4.54) — honestly reportable and not beaten by in-training BMC;
- the only lever that can raise r (and thus lower MAE while raising slope) is **Method C (LoRA)**,
  which is implemented + unit-tested but untrainable on this box.

## Recommended next steps (for a machine with ≥32 GB / a GPU, or via the refactor)

1. Run the Method C fold-0 gate: `--lora --lora-rank 8 --lora-blocks 8 --lora-alpha 16
   --lora-dropout 0.1 --lora-lr 2e-4` (target r ≥ 0.70). Then stack **B (denoised) under C**, and
   re-test **A (BMC) on top** (now that r is higher, BMC should finally convert r→slope).
2. Full 5-fold CV @60ep on the best config; compare to `long_global_cv.json`.
3. To run C on a memory-tight box: cache the frozen-prefix (block-`n_frozen-1`) output per train
   image with augmentation OFF, delete the frozen prefix blocks to free ~1 GB, and run only the K
   LoRA blocks + decoder on the cached prefix features (removes the per-step 20-block forward and
   the 1 GB prefix from memory).

---

# ═══════════ SESSION (breakthrough_design.md v2): Method M1 (severity head) + M3 (EMA) ═══════════
# Status at handoff: IMPLEMENTED + unit-tested + fold-0 done. Full 5-fold CV NOT YET run
# (this box too slow — continuing on a faster machine). Nothing committed as a "passing method" yet.

## What was built (all behind flags; default OFF is byte-identical to long_global — verified by tests)

- **M1 — first-class SEVERITY (eye-mean/MD) head** in `training.py`:
  - `PerPointVFModel(..., severity_head=False, severity_blend=1.0)`. A CLS→ΔMD MLP
    (1024→256→128→1) with the **final layer ZERO-INIT** ⇒ delta 0 at start (exact no-op, same
    safe-start trick as the additive global head). `_apply_severity(pred, cls)` adds
    `blend·delta` as a **UNIFORM field shift**, sets `self._last_severity` = predicted eye-mean.
    Called in `forward()` and `decode_latent()` AFTER the global head, BEFORE `_finish`.
    ► Because it is a uniform shift, it is removed by mean-centering ⇒ it **cannot change
      res_corr/eyeCorr** (spatial) by construction — a spatially-neutral de-shrink lever.
  - `compute_loss(..., severity_pred, severity_cfg)`: adds a mod+severe-weighted Huber on the
    eye-mean (pins the level) + a **batch-CCC de-shrink** on the batch's eye-means (raises BOTH
    sev_corr and sev_shrink toward 1). `severity_cfg=None` ⇒ term absent (unit-test-verified gate).
  - `train_lora_cached.py` flags: `--severity-head --severity-blend --severity-weight
    --severity-ccc --severity-eye-scale`. `eval_ckpt.load_model` rebuilds the head from ckpt meta
    (`severity_head`, `severity_blend`).
- **M3 — EMA** in `train_lora_cached.py`: `--ema --ema-decay` reuse `training.WeightEMA` over the
  trainable (LoRA A/B + decoder) params; eval + save from the EMA weights.
- **Tests:** `decoder/tests_method_m1.py` (encoder-free) — 5 tests PASS; full `tests_session3.py`
  regression still PASSES. Covers: OFF≡baseline (output independent of severity_head weights),
  zero-init no-op start, uniform-shift invariant, blend, de-shrink loss gating + monotonicity, EMA.
- **New eval tooling (the prior session's `scratchpad/feasibility.py` + `eval_oof.py` were WIPED):**
  - `decoder/eval_oof_cached.py` — pooled 5-fold OOF for cached-trainer checkpoints (raw + per-fold
    variance-match calib), prints the severity decomposition, writes `<tag>_cv.json`, **resumable**
    via per-fold `.npz` in `results/auto/oof_cache/`. Mirrors `run_cv.py` pooling but scores
    pre-trained checkpoints (train separately, ONE process at a time).
  - `decoder/eval_fold.py` — one-process fold scoreboard (pooled + severity-stratified + severity/
    spatial decomposition) for several checkpoints on one val fold; apples-to-apples TTA.

## Fold-0 scoreboard (TTA via eval_fold.py, on fold0_val) — THE KEY MEASUREMENTS

| ckpt | MAE | slope | corr | eyeCorr | severe | sev_corr | sev_shr | res_corr |
|---|---|---|---|---|---|---|---|---|
| long_global_f0 (base)  | 4.268 | 0.485 | 0.724 | 0.469 | 7.651 | 0.875 | 0.74 | 0.405 |
| lorac8_f0 (LoRA champ) | 4.105 | 0.574 | 0.734 | 0.552 | 7.545 | 0.853 | 0.85 | 0.495 |
| **m1m3_f0 (sev+EMA)**  | 4.206 | 0.552 | 0.721 | 0.518 | 7.612 | 0.860 | 0.83 | **0.441** |
| **m1sev_f0 (sev only)**| —(interrupted@~E24; best@E10 no-TTA 4.159/0.564/0.726; TTA not run) | | | | | | | |

## CRITICAL INSIGHTS (do NOT re-derive — they cost real compute)

1. **FOLD-0 SEVERITY IS ALREADY SATURATED** (sev_corr 0.875 on fold-0 = the design-doc §0 target
   0.875). ⇒ The pooled 0.80 sev_corr deficit lives in the **harder folds 1–4** (smaller train,
   LoRA overfits). **Fold-0 is a WEAK gate for M1's severity lever** — its payoff shows only at
   full 5-fold CV / on hard folds. Fold val sizes: fold0 smallest (easiest) → fold3 largest (hardest).
2. **The severity head is a uniform shift ⇒ mathematically cannot move res_corr/eyeCorr.** So the
   m1m3 spatial regression (res_corr 0.495→0.441 vs lorac8) is caused **entirely by EMA**.
3. **EMA (decay 0.998) BLURS the spatial pattern on a short 40-epoch cached run** (it averages in
   under-trained early weights; ~33 steps/epoch ⇒ 0.998 window ≈ 15 epochs = half the run). Fix on
   a short run: DROP EMA, or lower decay to ~0.99, or start EMA only in the last ~40% of epochs.
   Isolate M1 (no EMA) for its gate; test EMA as a SEPARATE hard-fold variance experiment.
4. **The de-shrink WORKS**: the severity loss raised sev_shrink 0.74→0.83 on fold-0 with no severe
   harm. On hard folds (lower sev_corr) it should help MORE (raise sev_corr toward 0.875 + slope).
5. **Overfit is real**: train MAE ~3.5 vs val ~4.2 by epoch 20+. Restoring augmentation
   (`--aug-views>1`, multi-view cached prefixes) and/or a lower-DOF adapter is the M3 lever to try.

## NEXT STEPS on the faster box (in priority order)

1. **M1-alone full 5-fold CV** (severity head, NO EMA — the clean isolation; expected to tie/beat
   lorac8 on fold-0 + add de-shrink, and help more on hard folds). Tag `m1sev`. One process at a time:
   ```bash
   for f in 0 1 2 3 4; do PYTORCH_ENABLE_MPS_FALLBACK=1 python decoder/train_lora_cached.py \
     --train-json decoder/results/cv_long/fold${f}_train.json \
     --val-json   decoder/results/cv_long/fold${f}_val.json \
     --out-tag m1sev_f${f} --epochs 40 \
     --lora-rank 8 --lora-blocks 8 --lora-alpha 16 --lora-dropout 0.1 --lora-lr 2e-4 \
     --warm-start decoder/results/auto/long_global_f${f}_best.pth --select mae_slope \
     --severity-head --severity-weight 0.5 --severity-ccc 0.5 --severity-eye-scale 2.0; done
   python decoder/eval_oof_cached.py --tag m1sev    # → m1sev_cv.json (raw+calib+severity decomp)
   ```
   GATE vs loraC (4.220/0.485/r0.657/severe7.694) and long_global (4.290): want raw MAE↓, slope↑,
   severe not worse, sev_corr↑. TARGET: MAE<4.00 AND slope≥0.60 (raw), severe≤baseline, r≥0.72.
   `eval_fold.py <ckpt...> --val-json cv_long/fold{f}_val.json` for per-fold severity decomposition.
2. If M1 helps but misses target: **stack a NON-BLURRING M3 variance lever** — snapshot ensemble
   over the last-K saved epochs, OR EMA decay 0.99 / late-start, OR swap LoRA→IA³/VeRA (lower DOF),
   OR restore `--aug-views 2-3`. Re-measure pooled.
3. **M2 (fundus→RNFL surrogate, training-only aux head)** — extend `build_longitudinal_grape.py`:
   the Follow-up sheet has IOP at col 5 (currently only interval@col4, cfp@col6 extracted); the
   Baseline sheet (`grape_new_vf_tests.json` source) has OCT RNFL/CCT/age/gender — join on
   subject+laterality. Add an aux head regressing RNFL from fundus features (raises the r ceiling).
4. Then M4 (structured/imbalance-aware output for severe band) per the design doc.

## Config that produced fold-0 m1m3 (sev+EMA)
rank8 / 8 LoRA blocks / alpha16 / dropout0.1 / lora-lr 2e-4 / head-lr 8e-4 (default) /
warm-start long_global_f{fold} / select mae_slope / severity-weight 0.5 / severity-ccc 0.5 /
eye-scale 2.0 / (m1m3 also: ema-decay 0.998). Cached-prefix trainer ~15–18 min/fold on the 17 GB
MPS box. Checkpoints on disk: `m1m3_f0_best.pth`, `m1sev_f0_best.pth` (fold-0 only; folds 1–4 TODO).

