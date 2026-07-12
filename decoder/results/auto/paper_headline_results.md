# Paper headline results — honest head-to-head + external validation

**Champion:** `p1disc` (disc-crop-as-sole-view, no-TTA). Pooled OOF MAE **4.113 dB** over the frozen
leak-free per-patient 5-fold CV (631 records / 263 eyes / 144 patients, `decoder/results/cv_long`).
Native pooled MAE does **not** clear 4.0 dB at 95% confidence (patient-bootstrap CI upper bound ≥ 4.0,
see §1.3), so the reliable headline is the **composition-adjusted head-to-head**, not a native-<4.0 claim.
Four of five candidate MAE levers (P1-jitter, M2/RNFL, high-res, PAPILA-severity-transfer, ensemble) failed
their pre-registered gates this session; this document reports what survives.

Reproduce: `python decoder/composition_report.py --tag p1disc` (reads
`decoder/results/auto/oof_cache_notta/p1disc_f{0..4}.npz`, no torch, no training).

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
| mild (≥22 dB)     | 2.929 | 61.2% | 3.09 | **−0.161 (tie — within ~0.18 dB SE)** |
| moderate (15–22)  | 3.934 | 21.6% | 5.66 | **−1.726** |
| severe (<15)      | 8.558 | 17.2% | 9.15 | **−0.592** |
| pooled            | 4.113 | —     | 3.91 | +0.203 (case-mix inversion) |

### 1.2 Eye-level strata (an eye's stratum = its true per-eye mean sensitivity; all 52 of its points
count — TDV-Net's likely convention, and the one that matches the Hodapp–Anderson glaucoma-stage
convention used to stratify §1.3's protocol contrast)

| stratum | n eyes | our MAE (raw) | our raw slope | TDV-Net MAE | Δ (ours − TDV) |
|---|---|---|---|---|---|
| mild (≥22 dB)    | 347 | 2.628 | 0.425 | 3.09 | **−0.462** |
| moderate (15–22) | 183 | 5.093 | 0.305 | 5.66 | **−0.567** |
| severe (<15)     | 101 | 7.438 | 0.286 | 9.15 | **−1.712** |

*(calibrated-slope companion: mild 2.823/0.516, moderate 5.531/0.375, severe 7.428/0.313 — pooled
calibrated MAE 4.345, not used for the headline since the champion anchor 4.113 is the raw pooled number.)*

**Decisive in moderate and severe under both conventions. Mild is a tie at the point level (−0.161,
inside noise) even though the eye-level cut looks like a clean win (−0.462) — do not headline "every
stratum."** This mirrors the caution already on record for the prior champion (`m1sev`): the eye-level
mild "win" is fragile and collapses once points are bucketed by their own value rather than their eye's
mean.

### 1.3 Composition-adjusted pooled number and native-MAE claim gate

- **Our model under any TDV-consistent composition** (≤13.5% severe points, reproduces their pooled
  3.91): **3.250 – 3.689 dB** over 43 feasible mild/moderate/severe mixes — beats their 3.91 under
  *every* one.
- **TDV-Net scored under our composition** (17.2% severe points): **4.686 dB** (vs our native 4.113).
- **Native pooled MAE 4.113, patient-bootstrap 95% CI [3.786, 4.474]** (144 patients, 5000 resamples).
  The native "<4.0" claim is **not allowed** (CI upper bound ≥ 4.0). The honest headline is
  composition-adjusted superiority (3.25–3.69 vs 3.91), not a native sub-4.0 number.

**Reconciliation vs the design-doc / plan placeholder numbers.** `decoder/specs/fundus_only_ceiling_design.md`
§3.1 and the task-15 plan entry quote **ours 3.37–3.62 vs 3.91, TDV-Net 4.69–4.81 under our mix** and
eye-level **2.760 / 5.440 / 7.251** — those are the prior champion `m1sev`'s numbers (pooled 4.256, 16.0%
severe points, WITH-TTA cache), not `p1disc`'s. Re-running `composition_report.py` fresh on the current
champion (`p1disc`, no-TTA, 17.2% severe points — slightly sicker mix, slightly better mild/moderate MAE,
slightly worse severe MAE) gives a shifted but still-decisive range: **3.250–3.689** (vs the quoted
3.37–3.62) and TDV-Net-under-our-mix **4.686** (a single value, vs the quoted 4.69–4.81 range — inside
that range at its low end). Direction and conclusion are unchanged; only the champion-specific point
estimates moved. `decoder/results/auto/composition_p1disc.json` holds the exact fresh numbers.

---

## 2. TD-comparability note (why we don't need to implement total deviation)

TDV-Net's name implies it predicts **total deviation** (TD = raw sensitivity − age/location-matched
normative value), not raw sensitivity. This looks like a scale mismatch but is not one:

**Pooled TD-MAE ≡ pooled raw-sensitivity-MAE.** TD_pred − TD_true = (sens_pred − norm) − (sens_true −
norm) = sens_pred − sens_true — the normative term is a deterministic, prediction-independent additive
constant per (age, retinal location) and cancels exactly in the pointwise difference. This is an algebraic
identity, not an approximation, and holds regardless of which normative table is used. **Our pooled 4.113
is therefore directly comparable to TDV-Net's TD-space 3.91 without implementing the HFA 24-2 age-normal
database.** (We searched the repo for an existing normative lookup — none exists — so we do not fabricate
one; see §4.)

What TD-space *does* change is severity **stratification**: TDV-Net's per-eye severity bands are
presumably Hodapp–Anderson stages cut on MD, not on raw mean sensitivity. Our eye-level strata (§1.2)
already use per-eye true mean sensitivity, which maps near-1:1 to HPA MD bands (<15 dB ≈ MD < −13,
15–22 ≈ MD −13…−6, ≥22 ≈ MD > −6) — if anything our "severe" cut is *stricter*, understating our
advantage there. No further TD-space stratification is needed for a valid comparison.

---

## 3. Slope

| | value | clears 0.60? |
|---|---|---|
| raw slope | **0.536** | **no** |
| calibrated slope (variance-matched gain fit on train folds) | **0.642** | yes |
| disattenuated slope (raw ÷ pooled reliability 0.845, §2.2 of the ceiling design) | **0.634** | yes |

Do not imply the raw slope clears 0.60 — it does not. Both the calibration-based and
errors-in-variables-based corrections do. *(Note: an earlier note quoted disattenuated = 0.643; that
number is `m1sev`'s figure (raw 0.543 ÷ 0.845), not `p1disc`'s. Recomputed directly from `p1disc`'s own
raw slope, 0.536 ÷ 0.845 = 0.634. Both 0.634 and 0.642 clear 0.60; the conclusion is unchanged, only the
third decimal moves.)*

---

## 4. External validation — PAPILA (independent cohort)

The GRAPE-trained `p1disc` champion (no PAPILA eye in training) was evaluated zero-shot on PAPILA, an
independent Spanish cohort, n=164 eyes / 82 patients (harmonized fundus + 30-2 MD; no per-point 24-2 data
available in PAPILA).

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

**Reading:** cross-cohort generalization of the fundus→severity map is real and matches the in-cohort
severity correlation (GRAPE sev_corr 0.802 vs PAPILA glaucoma r 0.755) — this answers the "does this
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

## 6. TD-space scatter

Deferred. No 24-2 (or 30-2) age/location normative lookup table exists anywhere in this repository
(checked: no `age_norm`, `normative`, `total_deviation`, or HFA-normal-DB module under `decoder/` or
`data/`), and building one from scratch was explicitly out of scope for this task. §2 already establishes
that pooled TD-MAE is algebraically identical to pooled sensitivity-MAE, so the comparability claim does
not depend on the scatter — only a *visual* TD-space plot is deferred, and it carries no additional
evidentiary weight beyond §2's identity.

---

## 7. Bottom line

- **Composition-adjusted:** ours 3.25–3.69 dB vs TDV-Net's 3.91 dB under any mutually-consistent
  severity mix; TDV-Net under our (sicker) mix scores 4.686. Decisive in moderate + severe; mild is a tie.
- **Comparable without extra work:** pooled MAE is already on the same scale as TDV-Net's TD-based metric
  (algebraic cancellation, §2).
- **Slope:** raw 0.536 (below 0.60), calibrated 0.642 and disattenuated 0.634 (both clear 0.60).
- **External validity:** PAPILA n=164 independent cohort, r = 0.753 overall / 0.755 glaucoma-only —
  correlation-only claim, MAE not meaningful cross-cohort.
- **Native <4.0:** not claimable (CI upper bound 4.474 ≥ 4.0). Do not headline it.
