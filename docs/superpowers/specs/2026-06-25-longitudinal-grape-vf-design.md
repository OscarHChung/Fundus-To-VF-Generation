# Longitudinal GRAPE expansion for sub-3.9 honest VF prediction

**Date:** 2026-06-25
**Status:** Design — awaiting user review before writing the implementation plan
**Owner:** Oscar Chung

## 1. Goal & success criteria

Predict HVF 24-2 (52-point) sensitivity from a fundus image, using the existing
frozen/adapted RETFound encoder → decoder with Garway–Heath sectoring (both
**mandatory, always on**).

Two targets, both measured on **leak-free per-patient 5-fold CV**:

1. **Overall MAE ≤ 3.9 dB** (firm bar; stretch: sub-3.74).
2. **A clinically strong line of best fit** on the truth-vs-prediction scatterplot:
   target calibrated slope ≥ ~0.7 with severe points (0–10 dB) placed near the
   severity oracle floor (~8.4 dB), materially better than the prior best (leaky
   slope 0.31–0.51).

Severe points are the stated priority; the error decomposition shows fixing
severity also lowers overall MAE, so the two goals are aligned, not in tension.

### Reporting honesty (non-negotiable in this spec)
- The **headline number is honest per-patient CV.** A fast single split may be used
  for *development velocity only*, always labeled as a dev signal, never reported as
  the clean metric.
- Because the task is being redefined to "predict the VF for *any* visit's fundus,"
  the pooled per-visit MAE will mechanically be lower than the old per-eye number
  (the visit distribution is milder on average). This is legitimate, but every
  report must **also** include severity-stratified MAE (mild / moderate / severe)
  and a per-eye breakdown so the pooled number is never misleading.

## 2. Root-cause recap (what we are fixing)

From the existing iterations log and Session-3 audit:

- The honest per-patient ceiling is **MAE ~4.46**, not the leaky-split 4.12/3.74.
- Error decomposition (clean fold): **sevMAE 2.58** (per-eye-mean severity) ⊕
  **resMAE 3.82** (within-eye spatial residual). **Severity is the dominant gap** —
  a perfect eye-mean would put MAE at ~3.82 on its own.
- The "data-limited, corr≈0.51, encoder overfits" verdict was **explicitly
  conditioned on "263 eyes / no new data."**

## 3. The unlock: the unused longitudinal data

`data/vf_tests/grape_data.xlsx` has **two** sheets. `vf_test_converter.py` reads
only `Baseline` (263 eyes, 1 VF/eye). The **`Follow-up` sheet** holds:

- **1,115 visit records** (mean 4.2 visits/eye, range 3–9), columns: Subject
  Number, Laterality, Visit Number, Interval Years, IOP, **Corresponding CFP**
  (fundus filename), Acquisition Device, Resolution, and a **61-value G1 VF grid**
  (cols 9–69) — the identical format the existing G1→24-2 KD-tree mapper consumes.
- **631 of those visits have their fundus image on disk** — and those 631 are
  *exactly* the files in `data/fundus/grape_fundus_images/`. Rows with CFP `"/"`
  have a VF but no photo and are skipped.
- Severity spread of the 631 paired visits: **104 severe (<15 dB) / 200 moderate /
  327 mild**, median 22.1 dB — i.e. **2.2× more severe examples** than the
  baseline's 47, plus a full progression range per eye.

The current `expand_GRAPE.py` globs every `<id>_<eye>_*` image into one list and
the model averages them — but pairs all of them with the **single baseline VF**.
Since glaucoma progresses across visits, a severe late-visit fundus is mislabeled
with the milder baseline field. Pairing each fundus with its **contemporaneous**
VF fixes both the 2.4× data loss and this label-noise bug at once, and the
progression range is exactly what teaches structure→severity (the bottleneck).

### Built-in correctness check
Follow-up `Visit Number == 1` rows correspond to the Baseline rows (e.g. subject 1
OD visit 1 → `1_OD_1.jpg`). The converter must reproduce `grape_new_vf_tests.json`
when restricted to visit 1; this is the unit test that validates the column→G1
alignment before any training.

## 4. Architecture (unchanged core, kept)

Reuse the honest best recipe verbatim — no architecture rewrite:

- **RETFound encoder**, frozen, with the **patch-shuffle fix** (`_encode`, no
  `random_masking`) — mandatory correctness fix from iter-8.
- **PerPointAttention** with the **Garway–Heath anatomical prior** — mandatory.
- **PointHead** + **additive zero-init global-spatial head** (lever1b2): the best
  honest single recipe (MAE 4.46), starts exactly as the per-point control and can
  only add the global pattern the local head misses.
- **Post-hoc variance-match calibration** as the scatterplot deliverable.

## 5. Staged plan (gated; stop early when the bar is met)

Sequencing: **staged & gated** (user choice). Encoder fine-tuning is **held in
reserve** (user choice) — only run if Stages 0–2 fall short.

### Stage 0 — Data + honest harness
- Extend `vf_test_converter.py` (or a new `build_longitudinal_grape.py`) to read the
  `Follow-up` sheet → one record per visit:
  `{PatientID, Laterality, VisitNumber, IntervalYears, FundusImage:[<its CFP>],
   hvf, mean_dB}`. Skip CFP `"/"`. Reuse the existing spiral_order + KD-tree mapping.
- Validate against the visit-1==baseline check above.
- Replace the `expand_GRAPE.py` eye-glob with per-visit pairing (each record keeps
  only its own contemporaneous image; multi-image averaging is dropped per record,
  net positive given 2.4× more records).
- **Per-patient grouped 5-fold split** (all visits of a patient → exactly one fold),
  severity-stratified at the patient level. Extends `run_cv.py` / `separate_datasets.py`.
- **Fixes:** 2.4× data, label noise, leak-free eval, more severe support.

### Stage 1 — Retrain the proven recipe on expanded data
- Train additive-global-head recipe on the 5 longitudinal folds; pool OOF preds.
- **Read-out:** expect MAE to fall from 4.46 toward ~3.9–4.1, slope/floor to improve
  from the added severe support. Measured in the first run.
- **Fixes:** severity (the dominant MAE term) directly.

### Stage 2 — Push the line of best fit
- Add a light training-time dispersion term (anti-shrinkage; safer against overfit
  now that data is larger) + retune the post-hoc variance-match calibration on OOF.
- **Fixes:** under-dispersion / flat scatterplot — the explicit #2 goal.
- **Gate:** if honest CV MAE ≤ 3.9 **and** calibrated slope ≥ ~0.7 with good severe
  placement → ship; otherwise proceed to reserve levers.

### Reserve levers (only if Stage 2 short of target)

- **Stage 3 — Light encoder adaptation:** LoRA on the last 1–2 ViT blocks or
  norm/bias BitFit, with heavy augmentation, CV-gated early-stop. The prior
  "encoder overfits → data-limited" result was on 211 eyes; 631 cleaner pairs with
  progression diversity changes the overfit math. This is the lever that lifts
  eyeCorr past the 0.51 frozen-feature cap. 4× slower (no feature cache). Kept only
  if it beats Stage 2 on honest CV.
- **Stage 4 — Finishing moves:**
  (a) Initialize the decoder from the **pretrained VF auto-decoder** (~2.0-MAE
      manifold) and/or add a manifold-consistency regularizer during training — the
      genuine encoder–decoder alignment and the one auto-decoder use never tried
      properly (prior attempts were post-hoc only, OOD-failed). Retrain the
      auto-decoder on non-zeroed inputs first to remove the OOD mismatch.
  (b) Cross-fold + diverse-member ensemble (frozen + adapted) + final calibration.

## 6. Risks & honest caveats

- **Visit non-independence:** 631 pairs come from 263 eyes / 144 patients. Effective
  N for *between-eye* severity is still ~144 patients — per-patient grouping is
  mandatory or metrics inflate. The genuine new signal is the *within-eye
  progression* range, which is what helps.
- **Pooled-MAE distribution shift:** the per-visit set is milder on average, so
  pooled MAE drops partly from distribution, not only model skill. Mitigation:
  always report severity-stratified + per-eye MAE next to the pooled number.
- **G1 column alignment:** the Follow-up header is offset/messy; the visit-1==baseline
  unit test gates correctness before any training spend.
- **Optional, not in scope unless asked:** sheet also has Age/Gender/CCT/IOP/glaucoma
  type — cheap auxiliary severity features, but they change inference inputs and
  dilute the fundus-only story. Listed, not adopted.

## 7. Files touched (anticipated)

- `vf_test_converter.py` — add Follow-up reader (or new `build_longitudinal_grape.py`).
- `decoder/expand_GRAPE.py` — replace eye-glob with per-visit pairing.
- `decoder/separate_datasets.py` / `decoder/run_cv.py` — per-patient grouped folds
  over the longitudinal records.
- `decoder/training.py` — no architecture change for Stages 0–2; consumes new JSON.
  Dispersion term already has a CLI knob (`--dispersion-weight`).
- `decoder/generate_scatterplot.py` / `scatter_clean.py` — stratified + line-of-fit
  reporting on OOF.
- New tests extending `decoder/tests_session3.py` (visit-1==baseline; per-patient
  fold disjointness; no fundus file shared across folds).

## 8. Out of scope
- Adding external datasets or OCT (none present here).
- Any concealed-leakage or deliberately-biased reporting.
- Architecture rewrites beyond the additive global head already validated.
