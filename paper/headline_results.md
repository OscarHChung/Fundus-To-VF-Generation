# Paper headline results — honest head-to-head + external validation

**Reported model: `p1disc_denoise`** — the `p1disc` disc-crop-as-sole-view recipe trained with
Theil-Sen per-point trajectory target-denoising (train-only; validation is always scored against
the **raw** HVF, never the denoised target — leak-free by construction). Pooled OOF MAE
**4.102 dB** (vs `p1disc`'s 4.113) over the same frozen leak-free per-patient 5-fold CV
(631 records / 263 eyes / 144 patients, `decoder/results/cv_long`).

**Promotion is not a pooled-MAE win — read this before citing a headline number.** ΔMAE vs
`p1disc` is **−0.011 dB, 95% CI [−0.076, +0.052]** (includes 0, P(Δ≥0)=0.38) and 3/5 folds
flipped — this is **pooled-MAE-NEUTRAL** and fails this project's own §6.5 auto-promotion gate
(needs ≤−0.12 dB, CI upper < 0, ≥4/5 folds better). We promote `p1disc_denoise` to the reported
model anyway, on three effects that ARE real (CI excludes 0 / consistently better), not on pooled
MAE:

| effect | p1disc | p1disc_denoise | Δ | 95% CI |
|---|---|---|---|---|
| severe-band MAE (eye-level) | 7.438 | 7.194 | **−0.245** | **[−0.449, −0.049]** excludes 0 |
| raw slope | 0.536 | **0.549** | +0.013 | — |
| calibrated slope | 0.642 | **0.645** | +0.003 | — |
| within-eye res_corr (spatial fidelity) | 0.474 | **0.503** | +0.029 | — |
| sev_corr (severity ceiling) | 0.802 | 0.809 | +0.007 | ceiling essentially unmoved |

Denoising cleans the label (Theil-Sen fit over each eye's visit trajectory), and that lift lands
almost entirely in **spatial fidelity and the severe band**, not in pooled MAE or severity
correlation — consistent with the session's standing finding that severity is eye-count-limited,
not noise-limited (sev_corr barely moves). So the honest framing is: **severe-band fidelity +
calibration/slope improvement at a pooled-neutral MAE**, not a pooled-MAE victory. `p1disc`
remains the point of comparison for anything not explicitly restated below (in particular, §4's
PAPILA external validation was run against `p1disc`, not re-run for the denoised variant — noted
inline).

Reproduce: `python decoder/composition_report.py --tag p1disc_denoise` (reads
`decoder/results/auto/oof_cache_notta/p1disc_denoise_f{0..4}.npz`, no torch, no training).
Scatter: `python decoder/scatter_from_oof.py --tag p1disc_denoise` (same cache, no torch) →
`decoder/results/auto/p1disc_denoise_scatter.png`.

---

## 1. Composition-adjusted head-to-head vs TDV-Net (the RELIABLE claim)

TDV-Net — *Deep learning-based prediction of 24-2 visual field from fundus photographs in glaucoma*,
Graefe's Arch Clin Exp Ophthalmol (2026): 31,443 train / 6,436 test photographs, two tertiary hospitals.
Reported pointwise MAE **3.09 / 5.66 / 9.15** mild/moderate/severe, pooled **3.91**.

Pooled MAE is not comparable across cohorts with different severity mixes: our OOF is sicker (17.2%
severe points vs the ≤13.5% their pooled 3.91 implies). The composition-adjusted comparison recomputes
both models under each other's case-mix, using only our own OOF errors — no retraining.

### 1.1 Point-level strata (each point bucketed by its own true sensitivity — the convention TDV-Net's
numbers are directly comparable under)

| stratum  | our MAE | our % of points | TDV-Net MAE | Δ (ours − TDV) |
|---|---|---|---|---|
| mild (≥22 dB)     | 2.904 | 61.2% | 3.09 | **−0.186 (tie — within ~0.18 dB SE)** |
| moderate (15–22)  | 4.062 | 21.6% | 5.66 | **−1.598** |
| severe (<15)      | 8.425 | 17.2% | 9.15 | **−0.725** |
| pooled            | 4.102 | —     | 3.91 | +0.192 (case-mix inversion) |

### 1.2 Eye-level strata (an eye's stratum = its true per-eye mean sensitivity; all 52 of its points
count — TDV-Net's likely convention, and the one that matches the Hodapp–Anderson glaucoma-stage
convention used to stratify §1.3's protocol contrast)

| stratum | n eyes | our MAE (raw) | our raw slope | TDV-Net MAE | Δ (ours − TDV) |
|---|---|---|---|---|---|
| mild (≥22 dB)    | 347 | 2.657 | 0.408 | 3.09 | **−0.433** |
| moderate (15–22) | 183 | 5.137 | 0.308 | 5.66 | **−0.523** |
| severe (<15)     | 101 | 7.194 | 0.311 | 9.15 | **−1.956** |

*(calibrated-slope companion: mild 2.838/0.489, moderate 5.534/0.367, severe 7.081/0.340 — pooled
calibrated MAE 4.299, not used for the headline since the reported-model anchor 4.102 is the raw
pooled number.)*

**Decisive in moderate and severe under both conventions. Mild is a tie at the point level (−0.186,
inside noise) even though the eye-level cut looks like a clean win (−0.433) — do not headline "every
stratum."** This mirrors the caution already on record for the prior champions (`p1disc`, `m1sev`):
the eye-level mild "win" is fragile and collapses once points are bucketed by their own value rather
than their eye's mean.

### 1.3 Composition-adjusted pooled number and native-MAE claim gate

- **Our model under any TDV-consistent composition** (≤13.5% severe points, reproduces their pooled
  3.91): **3.274 – 3.649 dB** over 43 feasible mild/moderate/severe mixes — beats their 3.91 under
  *every* one.
- **TDV-Net scored under our composition** (17.2% severe points): **4.686 dB** (vs our native 4.102).
- **Native pooled MAE 4.102, patient-bootstrap 95% CI [3.789, 4.455]** (144 patients, 5000 resamples).
  The native "<4.0" claim is **not allowed** (CI upper bound ≥ 4.0). The honest headline is
  composition-adjusted superiority (3.27–3.65 vs 3.91), not a native sub-4.0 number.

**Provenance of these numbers / prior-champion succession.** The champion lineage this session is
`m1sev` → `p1disc` → `p1disc_denoise`. Two earlier documents (`docs/specs/fundus_only_ceiling_design.md`
§3.1 and `docs/specs/2026-07-11-sub4-fundus-only-vf-design.md`) still quote `m1sev`'s
numbers (composition-adjusted 3.37–3.62, eye-level strata 2.76/5.44/7.25) — those predate `p1disc`
and are stale; see that design doc's own cleanup note. `p1disc`'s own fresh numbers were
**3.250–3.689** vs TDV-under-our-mix **4.686**; `p1disc_denoise`'s are the ones tabulated above
(3.274–3.649, TDV-under-our-mix unchanged at 4.686 since the composition mix — our fraction of
mild/moderate/severe points — is identical between the two: denoising changes *training targets*,
not which points are severe). `decoder/results/auto/composition_p1disc_denoise.json` holds the exact
numbers; `composition_p1disc.json` holds `p1disc`'s for comparison.

---

## 2. TD-comparability note (why we don't need to implement total deviation)

TDV-Net's name implies it predicts **total deviation** (TD = raw sensitivity − age/location-matched
normative value), not raw sensitivity. This looks like a scale mismatch but is not one:

**Pooled TD-MAE ≡ pooled raw-sensitivity-MAE.** TD_pred − TD_true = (sens_pred − norm) − (sens_true −
norm) = sens_pred − sens_true — the normative term is a deterministic, prediction-independent additive
constant per (age, retinal location) and cancels exactly in the pointwise difference. This is an algebraic
identity, not an approximation, and holds regardless of which normative table is used. **Our pooled 4.102
is therefore directly comparable to TDV-Net's TD-space 3.91 without implementing the HFA 24-2 age-normal
database.** (We searched the repo for an existing normative lookup — none exists — so we do not fabricate
one; see §6.)

What TD-space *does* change is severity **stratification**: TDV-Net's per-eye severity bands are
presumably Hodapp–Anderson stages cut on MD, not on raw mean sensitivity. Our eye-level strata (§1.2)
already use per-eye true mean sensitivity, which maps near-1:1 to HPA MD bands (<15 dB ≈ MD < −13,
15–22 ≈ MD −13…−6, ≥22 ≈ MD > −6) — if anything our "severe" cut is *stricter*, understating our
advantage there. No further TD-space stratification is needed for a valid comparison.

---

## 3. Slope

| | value | clears 0.60? |
|---|---|---|
| raw slope | **0.549** | **no** |
| calibrated slope (variance-matched gain fit on train folds) | **0.645** | yes |
| disattenuated slope (raw ÷ pooled reliability 0.845, §2.2 of the ceiling design) | **0.650** | yes |

Do not imply the raw slope clears 0.60 — it does not. Both the calibration-based and
errors-in-variables-based corrections do. This is a small, real improvement over `p1disc`'s own
raw 0.536 / calibrated 0.642 / disattenuated 0.634 — the denoised targets buy a modestly better
range-compression fit, consistent with the severe-band gain in §1.2.

---

## 4. External validation — PAPILA (independent cohort)

**Not re-run for `p1disc_denoise`.** These numbers were produced by evaluating the `p1disc`
champion (zero-shot, no PAPILA eye in training) — the denoised-target training recipe was not
re-evaluated on PAPILA this session. Denoising only changes GRAPE *training* targets (Theil-Sen
trajectory fit, train-only); the frozen encoder and the fundus→severity map it was probing are
unchanged, so we do not expect the correlation numbers below to move materially, but this is an
unverified expectation, not a measurement — treat the PAPILA numbers as characterizing the
`p1disc`-family fundus→severity map, not `p1disc_denoise` specifically.

The GRAPE-trained `p1disc` champion was evaluated zero-shot on PAPILA, an independent Spanish
cohort, n=164 eyes / 82 patients (harmonized fundus + 30-2 MD; no per-point 24-2 data available in
PAPILA).

- **Harness sanity check:** evaluating the same champion on GRAPE through this harness reproduces
  `p1disc` exactly (pooled 4.113 / slope 0.536 / corr 0.684) — confirms the external-eval pipeline is
  correct before trusting the PAPILA numbers.
- **Overall (n=164):** Pearson **r = 0.753** (95% CI [0.608, 0.848]), Spearman **r = 0.616**.
- **Glaucoma subgroup (n=87):** Pearson **r = 0.755** (95% CI [0.569, 0.871]), Spearman 0.678.
- **Suspect subgroup (n=68):** Pearson r = 0.226 (95% CI [0.016, 0.417]) — weak.
- **Healthy subgroup (n=9):** Pearson r = −0.441 (95% CI [−0.775, 0.544]) — noise at n=9, wrong-signed;
  not interpretable.

**Caveats, stated plainly:**
1. **The printed dB MAE (~26 dB) is not meaningful and must not be reported as a headline number.** GRAPE
   trains on 24-2 raw sensitivity; PAPILA's ground truth is 30-2 mean deviation — different test pattern,
   different scale, different zero-point. Correlation (scale-free) is the valid cross-cohort metric here;
   MAE is not, absent a full TD/scale harmonization this task does not implement.
2. **The overall r = 0.753 is carried by the glaucoma stratum plus healthy/glaucoma separation**, not by
   within-stratum discrimination everywhere: the suspect subgroup alone is weak (r = 0.226) and the
   healthy subgroup is uninterpretable noise (n = 9). Report the glaucoma-subgroup r (0.755) alongside the
   overall number, not overall alone.
3. **PAPILA licensing.** Figshare's item-level metadata tags the dataset GPL-3.0+, but the companion
   *Scientific Data* (2022) paper states CC-BY-4.0 in its own text (journal policy for Data Descriptors).
   We treat it as **CC-BY-4.0 for non-commercial research use with attribution** (cite Kovalyk et al.
   2022), flagged for PI/legal confirmation before any redistribution beyond internal use. Full
   discussion: `data/external/papila/SOURCE.md` § License resolution.

**Reading:** cross-cohort generalization of the fundus→severity map is real and matches the in-cohort
severity correlation (GRAPE sev_corr 0.802–0.809 vs PAPILA glaucoma r 0.755) — this answers the "does this
transfer off your own camera/population" objection, with the stated scale and subgroup caveats.

---

## 5. Protocol contrast

- **Ours:** leak-free **per-patient** 5-fold CV, eye-disjoint, severity-stratified, frozen and never
  reseeded (`decoder/results/cv_long`). 631 records / 263 eyes / 144 patients.
- **TDV-Net:** split hygiene (patient- vs record-disjoint) is not stated in the abstract; full text is
  paywalled. Their reported MAE has a negligible SE at their test-set size; ours is patient-bootstrap
  ±0.18–0.34 dB and is reported with CIs throughout.
- **MLEDL** (npj Digit Med 2025, PMC12216876) is the actually-comparable study by data regime (633
  patients / 1,129 eyes, same target = raw sensitivity in dB) but is **not head-to-head comparable to
  TDV-Net or to us on its own terms**: its target is **Octopus 59-point** perimetry, not HFA 24-2 dB — a
  different device, different point layout, different dynamic range. Its own split is per-record, not
  confirmed patient-disjoint (1,300 train / 298 test records from 1,129 eyes); our own history shows a
  leaky per-record split reads ~0.43 dB better than the clean per-patient split on this exact codebase, so
  MLEDL's fundus-only band (3.90–4.13) is plausibly ≈4.3+ under our protocol. It is cited for qualitative
  context (ROI-crop lift, metadata lift, longitudinal lift), not for a numeric head-to-head.

---

## 6. dB-space scatter (and why TD-space is still deferred)

`decoder/results/auto/p1disc_denoise_scatter.png` — pooled truth-vs-pred scatter for the reported
model, built directly from the frozen OOF cache (`decoder/scatter_from_oof.py --tag
p1disc_denoise`, no torch, no live inference). Shows raw-space points, the raw best-fit line
(y = 0.55x + 9.8), and the calibrated best-fit line (y = 0.64x + 7.6) against y = x.

TD-space plotting remains deferred. No 24-2 (or 30-2) age/location normative lookup table exists
anywhere in this repository (checked: no `age_norm`, `normative`, `total_deviation`, or
HFA-normal-DB module under `decoder/` or `data/`), and building one from scratch was explicitly
out of scope for this task. §2 already establishes that pooled TD-MAE is algebraically identical to
pooled sensitivity-MAE, so the comparability claim does not depend on the scatter — only a *visual*
TD-space plot is deferred, and it carries no additional evidentiary weight beyond §2's identity.

---

## 7. Bottom line

- **Reported model `p1disc_denoise`** (pooled OOF MAE 4.102): promoted for a real, CI-backed
  severe-band gain (−0.245, CI excludes 0) and small raw/calibrated slope gains, at a
  **pooled-MAE-neutral** cost (Δ−0.011, CI includes 0) relative to `p1disc` — not a pooled-MAE
  victory; do not headline it as one.
- **Composition-adjusted:** ours 3.27–3.65 dB vs TDV-Net's 3.91 dB under any mutually-consistent
  severity mix; TDV-Net under our (sicker) mix scores 4.686. Decisive in moderate + severe; mild is a tie.
- **Comparable without extra work:** pooled MAE is already on the same scale as TDV-Net's TD-based metric
  (algebraic cancellation, §2).
- **Slope:** raw 0.549 (below 0.60), calibrated 0.645 and disattenuated 0.650 (both clear 0.60).
- **External validity:** PAPILA n=164 independent cohort, r = 0.753 overall / 0.755 glaucoma-only —
  correlation-only claim, MAE not meaningful cross-cohort, measured on `p1disc` (not re-run on the
  denoised variant; see §4). CC-BY-4.0 treated as the operative PAPILA license, flagged for
  confirmation before redistribution (`data/external/papila/SOURCE.md`).
- **Native <4.0:** not claimable (CI upper bound 4.455 ≥ 4.0). Do not headline it.
