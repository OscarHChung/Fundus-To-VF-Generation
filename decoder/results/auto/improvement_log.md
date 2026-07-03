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
| (baseline) | Huber | fold0 raw | 4.268 | 0.485 | 0.724 | 7.651 | fold0 ref |
| bmc_s3 | BMC σ=3, 25ep | fold0 raw | 4.596 | 0.429 | 0.698 | 8.126 | ✗ FAIL (slope<baseline, MAE+0.33, severe worse, σp/σt 0.61) — σ too small; raise it |
| bmc_s7 | BMC σ=7, 25ep (killed @ep8) | fold0 raw | ~5.27 | ~0.42 | ~0.60 | — | ✗ FAIL (MAE stuck 5.2–5.4, r collapses 0.72→0.60, slope no better than σ=3) — high σ degrades the fit without buying slope |
| raw25 | baseline recipe, 25ep | fold0 raw | 4.433 | 0.422 | 0.711 | 8.260 | epoch-matched CONTROL for B (35 fewer epochs than the 60ep ref → +0.16 MAE) |
| denoise25 | +denoised targets, 25ep (best@ep18) | fold0 raw | 4.474 | 0.430 | 0.707 | 8.082 | ~ MARGINAL vs raw25: severe −0.18 ✓, slope +0.008 ✓, MAE +0.04 ✗ (epoch-confounded: best@18 vs raw@~24) |

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

### Method C verdict: BLOCKED by the local environment (not by the method)

Encoder-gradient training could not be run on this machine. 4 distinct LoRA configs
(rank8/4-blocks/batch16; +grad-checkpointing; +MPS empty_cache; rank8/2-blocks/batch2) all OOM-die
at ~90 s (epoch 2, ~step 53), and a fresh process eventually failed at import (base_model load).
Cause: 17 GB RAM with swap saturated (~1.2 GB free) cannot hold the 1.2 GB RETFound base + a per-step
encoder graph + gradients. The frozen-decoder runs work only because they CACHE features once and
never hold an encoder graph. The LoRA code is correct and unit-tested; it needs a higher-memory
machine (or the prefix-caching refactor below) to train.

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

