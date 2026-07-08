# P1 (disc-ROI sole-input) — handoff to a new session

**What this is:** implementation + partial results for P1 from `fundus_only_ceiling_design.md` §4 — using a
laterality-aware **disc/ROI crop as the SOLE encoder input** (not the failed iter-11 averaged-view), on top
of the M1 severity head. Goal: push fundus-only native MAE toward < 4.0 at slope ≥ 0.60.

**Headline so far (encouraging, incomplete):** on fold-0 the disc crop beats the full-image champion
`m1sev` on *every* metric, and — the surprise — it lifts **eyeCorr 0.48 → 0.58**, i.e. it improves the
*within-eye spatial* channel the ceiling analysis had written off. Need the full 5-fold pooled number to
know if native sub-4.0 is real; fold-0 is the easy fold.

---

## 1. Current run state (as of handoff)

| fold | status | checkpoint | result (no-TTA, raw) |
|---|---|---|---|
| 0 | ✅ done | `p1disc_f0_best.pth` | **MAE 4.101 / slope 0.588 / corr 0.745 / eyeCorr 0.578** @ep32 |
| 1 | 🔄 was training when session ended (saved 4.025@ep3, still improving) | `p1disc_f1_best.pth` | partial |
| 2,3,4 | ⏳ pending | — | — |

**A detached chain (`nohup zsh run_p1disc_folds.sh`, was PID 11831) may still be running folds 1→4.**
First check whether it survived:
```bash
pgrep -fl "train_lora_cached|run_p1disc"          # any training alive?
ls decoder/results/auto/p1disc_folds_complete.flag # exists => all 4 done
for f in 1 2 3 4; do echo -n "fold $f: "; grep "BEST val" decoder/results/auto/p1disc_f${f}.log 2>/dev/null | tail -1; done
```

## 2. How to (re)run folds 1-4

The chain script is `decoder/results/auto/run_p1disc_folds.sh` (idempotent — re-running overwrites logs +
`p1disc_f{f}_best.pth`). To run only the folds that are missing, edit the `for f in 1 2 3 4` line.
**Serialize — this box (16 GB, ~300 MB free) OOM-kills any second torch process.** One fold ≈ 40 min.
```bash
cd "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation"
nohup zsh decoder/results/auto/run_p1disc_folds.sh >/dev/null 2>&1 &
```
Recipe per fold (identical to `m1sev` + `--disc-only`, so any delta is the disc lever alone):
```
--lora-rank 8 --lora-blocks 8 --lora-alpha 16 --lora-dropout 0.1 --lora-lr 2e-4
--warm-start long_global_f{f}_best.pth --select mae_slope
--severity-head --severity-weight 0.5 --severity-ccc 0.5 --severity-eye-scale 2.0
--disc-only  (DISC_HALF defaults to 0.27 = tight disc; stored in ckpt)
```

## 3. ‼️ EVAL WITHOUT TTA — TTA is broken for crops

TTA rotates ±5°, which on a tight disc crop shifts the disc and pulls in black border → injects a **−1.4 dB
bias** (fold-0 TTA RAW 4.688 vs no-TTA 4.101). Checkpoints are selected on the no-TTA val path, so eval
no-TTA for consistency. Once folds 1-4 exist:
```bash
# pooled OOF, NO TTA (fresh npz cache to avoid the TTA-cached m1sev npz):
python decoder/eval_oof_cached.py --tag p1disc --no-tta --cache-dir decoder/results/auto/oof_cache_notta
# fair baseline: also re-eval m1sev no-TTA into the SAME notta cache dir, then compare pooled:
python decoder/eval_oof_cached.py --tag m1sev --no-tta --cache-dir decoder/results/auto/oof_cache_notta
```
(`m1sev`'s published 4.256 used TTA; TTA helped m1sev only ~0.02, so m1sev no-TTA pooled ≈ 4.27 — that is
the honest bar for disc-only.)

**Decision rule (pre-committed, from the design doc §6.5):** promote P1 iff pooled paired ΔMAE ≤ −0.12 dB
with 95% CI excluding 0 AND negative in ≥4/5 folds AND severe-band not worse. Fold-0 alone can kill but
never promote (the M2 lesson: fold-2 looked great, washed out on full CV).

## 4. What was implemented (all behind a flag, default OFF ≡ `m1sev`; 6 tests in `tests_method_p1.py`, all green)

- `training.py MultiImageDataset(..., disc_only=False)` — `views=['disc']` when set (sole input; precedence
  over `--disc-crop`). `DISC_HALF` (module global) is the crop half-size.
- `train_lora_cached.py` — `--disc-only`, `--disc-half`; sets `T.DISC_HALF`; threads `disc_only` into both
  train + val caches; stores `disc_only`+`disc_half` in the checkpoint.
- `eval_ckpt.py load_model` — reads `disc_only`/`disc_half`, sets `T.DISC_HALF`, attaches `model._disc_only`;
  `per_eye_preds` crops identically. (`eval_oof_cached.py` calls `per_eye_preds`, so it inherits this.)
- Crop geometry visually validated (scratchpad `disc_OD_{tight,wide}.jpg`): OD cx0.78 / OS cx0.22, cy0.49;
  disc goes from ~30 px (full→224) to ~1000 px→224 ≈ **5× finer** on the optic nerve head.

## 5. Results vs the full-image champion (no-TTA, fold-0 — the only complete comparison)

| metric | `m1sev_f0` (full) | `p1disc_f0` (disc) | Δ |
|---|---|---|---|
| MAE | 4.159 | **4.101** | −0.058 |
| slope raw | 0.564 | 0.588 | +0.024 |
| corr | 0.726 | 0.745 | +0.019 |
| eyeCorr | ~0.476 | **0.578** | **+0.102** |
| calib slope (s0.25) | — | 0.607 @ MAE 4.12 | clears 0.60 |

Interpretation: the disc crop raised the severity channel (corr) *and* the within-eye spatial channel
(eyeCorr) — the latter was the "dead" channel in the ceiling analysis. If this holds pooled, the native
number improves and the spatial-lever verdict (`fundus-only-is-severity-estimation`) needs a caveat.

## 6. Directions to push native MAE below 4.0 (ranked by EV, given these results)

The eyeCorr gain reopens levers the ceiling doc had closed. In rough priority:

1. **Crop-safe TTA (recover the lost ~0.02–0.05).** Current TTA rotates then the crop is fixed → border
   artifacts. Fix = rotate the FULL image first, THEN disc-crop (disc stays centered), or replace rotation
   TTA with small scale/shift jitter, or reflect-pad the crop. Cheap, pure eval-side, no retrain.
2. **P2 — two-view FEATURE fusion (full ⊕ disc), not averaging.** Now well-motivated: disc helped BOTH
   channels, and the full image still carries the central-field/macula points the tight crop discards. Fuse
   full+disc patch tokens inside `PerPointAttention`; LoRA-adapt the disc branch (frozen RETFound is OOD on
   zoomed crops). Biggest remaining upside; most work. Design §4 P2.
3. **Crop-scale sweep (fold-0 only, then pick).** Tight 0.27 discards the macula. Try wide **0.45**
   (disc+macula ROI, `--disc-half 0.45`) — may rescue the mild/moderate central points and the low slope.
   Also `--disc-half 0.35`. One fold-0 run each; pick the best for the full CV. Pre-registered in the design.
4. **Disc-centric anatomical prior.** The Garway–Heath prior in `PerPointAttention` maps VF points to
   *macula-centered* patch positions; on a disc crop that mapping is misaligned, which likely caps the
   eyeCorr gain at 0.58. A disc angular prior (VF sector ↔ optic-disc clock-hour arc) could unlock more.
5. **Stack the free levers:** `--denoised` (Method B, ~0.05, sub-noise but free) + cross-fold ensemble +
   variance-matched calibration (calibration already gives slope ≥0.60 at ~+0.02 MAE).
6. **The honest reporting route (independent of native MAE):** at matched case-mix vs the comparator we are
   already ~3.4–4.0 and win every severity stratum; the screening-population number is ~3.5. See
   `fundus_only_ceiling_design.md` §3.1 and the composition analysis. This clears sub-4.0 without any
   further training and is the strongest publication claim regardless of how far the model stack gets.

**Best current estimate of the ceiling of this path:** native pooled ~4.05–4.15 raw (disc + stack), slope
0.60 calibrated; matched-composition/screening comfortably sub-4.0. Native sub-4.0 is *possible but not
assured* — it hinges on whether the fold-0 eyeCorr/corr gains hold on the hard folds (2, 4).

## 7. Open risks / caveats
- **Fold-0 is the easiest, smallest fold and a weak gate.** M2 passed a single-fold gate and washed out on
  full CV. Do not believe a native sub-4.0 story until the pooled 5-fold OOF (no-TTA) shows it with the
  per-fold signs consistent (≥4/5 negative).
- **eyeCorr 0.58 contradicts the design's "spatial is worth 0.06 dB" finding.** Either the disc genuinely
  adds eye-specific spatial signal (great — update `fundus-only-is-severity-estimation`), or it is lifting
  the *population-template-like* component; run the template-partial-corr decomposition (scratchpad
  `robust.py` / `ablate.py`) on the pooled p1disc OOF to tell which.
- The tight crop **discards the macula** → risks the central 24-2 points; watch the mild/moderate strata in
  the pooled stratified report, not just pooled MAE.
- Score only vs the RAW VF; keep the severity-stratified table + patient-bootstrap CI with every number
  (design §6). Native "MAE < 4.0" is only a real claim if the patient-bootstrap 95% upper bound < 4.0.

## 8. Files
- Code (committed): `training.py`, `train_lora_cached.py`, `eval_ckpt.py`, `tests_method_p1.py`.
- Run script: `decoder/results/auto/run_p1disc_folds.sh`. Logs: `p1disc_f{0..4}.log`.
- Checkpoints (NOT in git, ~500 MB each): `p1disc_f{0,1}_best.pth` exist; 2-4 pending.
- Design context: `decoder/specs/fundus_only_ceiling_design.md`. Analysis scratchpad had `oof.py`,
  `frontier2.py`, `template.py`, `ablate.py`, `robust.py`, `report_frames.py` (regenerate if wiped).
