# Fundus-only 24-2 VF prediction: feasibility, proposals, and a pre-committed eval protocol

**Status:** design only. Nothing here is implemented or trained.
**Constraint:** RETFound encoder + custom VF decoder with Garway–Heath sectoring (both freely editable
internally). **Inference consumes a single fundus image.** Training may use any additional signal.
**Eval:** the frozen leak-free per-patient 5-fold OOF over 631 records (`decoder/results/cv_long`).

---

## 0. Verdict

**No — conditional on one unrun diagnostic (D1). Pooled MAE < 4.0 dB at slope ≥ 0.60 is very unlikely to
be reachable fundus-only on these 631 records, and the severe-band thresholds are not reachable by any
model — including one that is handed a real prior VF of the same eye.** The pooled verdict is stated as a
strong prior, not a proof: it becomes a proof only if D1's learning curves (§4) are flat at n = 210 eyes.
Every load-bearing measurement below has been independently reproduced; the residual uncertainty is
whether the information ceiling is real or a 144-patient sample-size artifact (Open Risks R1/R3).

The binding limit differs by stratum:

| stratum | binding limit | evidence |
|---|---|---|
| **pooled** | **structure–function information content of the photograph** (conditional on R1) | the image's *eye-specific* spatial contribution is **0.056 dB** of MAE (95% CI [−0.121, +0.007]) — real (partial-corr 0.196, p < 0.001) but economically negligible; the severity channel is saturated at r = 0.813, at/above the best comparable published fundus-only result |
| **severe band** | **perimetric label noise** | σ_ε ≈ 6.0 dB where 44% of severe-band points live (5–15 dB); Bayes MAE floor 3.16 dB; a same-eye prior VF scores only 4.46 dB there |
| — | data volume is **secondary, not binding** (moderate confidence — see Open Risk R1) | 50× more training data (TDV-Net, 31,443 photos) does **not** beat us in any severity stratum at matched case mix |

**Label noise is not the pooled binding limit.** The "2.76 dB test-retest noise floor" in the brief is a
misread (§2.1): 2.76 dB is the MAE between *two noisy VF exams*. A deterministic predictor of the true
field has a Bayes MAE floor of **≈2.0–2.25 dB** pooled. We are ~2.0 dB above it. Noise is not what stops us
— and this rests on *measured* near-zero test-retest error correlation (excess ρ = −0.036, §2.1), not an
assumption.

**What actually stops us**, quantitatively: at the current information levels, the best pooled MAE
attainable at slope = 0.60 (sweeping output gain, i.e. anywhere on the calibration frontier) is
**4.387 dB**. Clearing 4.0 requires a **−0.39 dB** improvement. To calibrate that: it is ~2.2 unpaired
MAE standard errors (SE 0.179) and ~3.7 fold-level SEs (SE 0.106) — a **large, detectable** effect, but
*not* the "11 SEs" a naive division by the paired-clone SE (0.037) would suggest (that SE only separates
two co-trained near-identical models; §2.6). Reaching −0.39 requires *either* fundus→severity correlation
0.813 → **0.875**, *or* within-eye spatial correlation 0.406 → **0.60**. Neither has been demonstrated in
this codebase across seven method families, nor anywhere in the comparable literature. A *simultaneous,
realistic* gain on both axes (sev 0.84, res 0.46) still lands at ≈ 4.08.

---

## 1. Correcting the starting point

The brief conflates two models on the severe band. Restated from the artifacts:

| quantity | value | source |
|---|---|---|
| fundus-only pooled MAE (champion `m1sev`) | **4.256** | `m1sev_cv.json`, reproduced from OOF caches |
| fundus-only raw slope | 0.543 | ditto |
| fundus-only **severe band (n=101)** | **MAE 7.251, slope 0.251** | ditto — *not* 6.06/0.482 |
| longitudinal severe band (n=101) | MAE 6.06, slope 0.482 | `longitudinal_champion.json` (prior-VF model) |
| fundus-only baseline `long_global` | 4.290 / 0.473 | `long_global_cv.json` |

The severe-band thresholds fail *much* harder fundus-only than the brief implies: slope 0.251, not 0.482.

I verified the OOF caches align with the fold JSONs by asserting each cached ground-truth vector equals
the record's own 52-vector (`scratchpad/oof.py`). Pooled metrics reproduce 4.256 / 0.543 / 0.665 exactly.

---

## 2. Measurements made for this design (all reproducible, none require training)

### 2.1 Label noise, measured from GRAPE's own repeat visits
368 records carry a causal prior VF (19,136 paired points). Minimum inter-test interval is 0.42 yr, so
every pair contains some true progression; regressing retest MAE on interval and extrapolating to Δt = 0
removes it.

| quantity | value |
|---|---|
| retest MAE \|curr − prior\|, all intervals | 3.185 dB |
| retest MAE extrapolated to Δt = 0 | **2.815 dB** (this is the brief's "2.76") |
| ⇒ per-point σ_ε | ≈ 2.49 dB |
| **Bayes MAE floor for a deterministic predictor** (≈ retestMAE / √2) | **2.01 dB (Δt→0) / 2.25 dB (all)** |

The ÷√2 conversion is *exact* only under independent, equal-variance, Gaussian point errors; it is an
**upper bound** on the floor under the two ways VF noise violates that, and both violations were checked:
- **Leptokurtosis** (the difference has excess kurtosis +2.97, sd/MAD-σ = 1.56): a single exam is heavier-
  tailed than the difference, so E|e₁−e₂|/E|e₁| > √2 ⇒ ÷√2 *over*estimates the floor. Conservative for us.
- **Independence of e₁, e₂** (same patient/perimeter — the assumption a reviewer would attack): **measured,
  not assumed.** On the 105 eyes with ≥3 visits, the lag-1 autocorrelation of detrended per-point residuals
  is −0.617 vs a white-noise detrending-artifact null of −0.581, i.e. **excess ρ = −0.036 (z = −6.3)** — if
  anything slightly *negative*, in every severity band. Positively-correlated errors would raise the floor
  (floor = retestMAE/[√2·√(1−ρ)]: ρ = 0.3 → pooled 2.69 / severe 3.78; ρ = 0.5 → 3.18 / 4.47). The data do
  not support ρ > 0, so no upward correction applies. **A perfect model scores ≈2.0–2.25 dB, not 2.76.**

**Noise is strongly heteroscedastic** — this is what kills the severe band:

| true sensitivity | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–28 | 28–32 |
|---|---|---|---|---|---|---|---|
| retest MAE (dB) | 4.09 | **6.97** | **6.92** | 4.92 | 2.86 | 2.32 | 1.57 |
| σ_ε (dB) | 3.51 | **6.04** | **6.14** | 4.54 | 2.74 | 2.18 | 1.54 |

(Matches the classic perimetry result — variability peaks at moderate damage, falls at the 0 dB floor.)

Eye-mean (MD) retest σ = 1.55 dB ⇒ MD reliability 0.928 ⇒ **sev_corr ceiling from noise = 0.963**.
Residual (within-eye) corr ceiling from noise ≈ **0.88**. Neither is close to binding.

### 2.2 The oracle slope ceiling
A prior VF is an unbiased noisy read of (approximately) the same true field. Regressing it on the current
VF yields the **reliability**, which is exactly the slope ceiling for *any* predictor of the true field:

| stratum | reliability = max attainable slope | Bayes MAE floor |
|---|---|---|
| pooled | **0.845** | 2.25 |
| severe (<15 dB) | **0.768** | **3.16** |
| moderate | 0.713 | 2.78 |
| mild | 0.649 | 1.75 |

Consequences: (i) `m1sev`'s *disattenuated* pooled slope is 0.543/0.845 = **0.643** — it already clears
0.60 once you correct for errors-in-variables; the raw 0.543 is partly a measurement artifact. (ii) In
the **mild** band, even a perfect model has slope 0.649 — a slope target of 0.6 there is nearly maxed.
(iii) In the severe band the disattenuated slope is 0.251/0.768 = **0.327**.

### 2.3 The decisive ablation: what does the image actually contribute?
Decompose every prediction and target into (eye-mean) + (within-eye residual). Build a **population
template** = the mean residual field of each fold's *training* eyes, laterality-specific, with its gain
also fit on train only (strictly out-of-fold; per-fold gains 0.80–0.88).

| predictor (5-fold OOF, 631 records) | MAE | slope | r | severe MAE | severe slope |
|---|---|---|---|---|---|
| A. `m1sev` (the model) | 4.256 | 0.543 | 0.665 | 7.251 | 0.251 |
| B. model eye-mean **+ population template** (no image spatially) | **4.313** | 0.514 | 0.666 | 7.345 | 0.199 |
| C. model eye-mean + flat field (no spatial at all) | 4.642 | 0.447 | 0.609 | 7.825 | 0.114 |
| D. **true** eye-mean + population template | **3.451** | **0.629** | 0.798 | 6.171 | 0.264 |
| E. **true** eye-mean + flat field (severity oracle) | **3.848** | 0.561 | 0.749 | 6.695 | 0.178 |

- **A − B = −0.056 dB** (paired patient bootstrap, 95% CI **[−0.121, +0.007]**, P(model better) = 0.958).
  *The eye-specific spatial contribution of the fundus photograph is worth ~0.06 dB of pooled MAE — real
  but economically negligible, and below the paired detection threshold at 144 patients.*
- **The eye-specific spatial signal is statistically present, just small.** Partial corr(model residual,
  true residual | template) = **0.196** pooled, patient-bootstrap 95% CI **[0.144, 0.246], p < 0.001**
  (per-eye mean 0.219). Incremental R² of the image over the template = **0.0315**. So the honest claim is
  *"present but MAE-negligible,"* **not** *"absent."* The tension is pure shrinkage: when the model
  residual and the template are each variance-matched (un-shrunk) to the truth, the **model beats the
  template by 0.36 dB** (4.83 vs 5.19), and an OOF-fit linear blend `a·template + b·model_residual` lifts
  res_corr to **0.453** (> template 0.420, CI on the gain excludes 0). The image and the template are
  *complementary*; the model simply operates at a shrinkage where its extra signal buys almost no MAE.
- Pooled residual correlation at the operating shrinkage: model 0.4065 vs template 0.4196 — the "model
  slightly below template" figure is an amplitude artifact (per-eye residual corr is a tie: model 0.471,
  template 0.473; Δ = −0.003, 95% CI [−0.022, +0.014]).
- **Row E is the headline finding: a model that knew each eye's MD perfectly and emitted a flat field
  would score 3.85 dB pooled — clearing the MAE target with literally zero spatial information.**
  Row D (perfect MD + template) clears *both* thresholds (3.451 / 0.629).

⇒ **Pooled fundus-only performance is, to first order, a fundus→MD problem.** The decoder's eye-specific
spatial signal is real but worth ~0.06 dB and is nearly reproducible with a lookup table.

> This also retires a claim the repo has been steering by. Session 3 concluded there was "spatial
> headroom" because a frozen-feature MLP probe reached eyeCorr 0.511 vs the champion's 0.41. But the
> **template alone reaches eyeCorr 0.488**. The "frozen-feature spatial cap of 0.51" was the population
> template in disguise, not eye-specific signal.

### 2.4 The frontier: what correlations would we need?
Simulated on the *real* OOF error fields (rescaling the actual, heavy-tailed, spatially-structured error
components rather than adding Gaussian noise — the anchor cell reproduces 4.265/0.543 vs the true
4.256/0.543, and severe 7.204/0.251 vs true 7.251/0.251).

Minimum pooled MAE subject to raw slope ≥ 0.60, sweeping output gains:

| sev_corr ↓ / res_corr → | 0.406 (now) | 0.510 | 0.600 | 0.700 |
|---|---|---|---|---|
| **0.813 (now)** | **4.387** | 4.122 | *3.911* | 3.663 |
| 0.875 | *3.998* | 3.794 | 3.605 | 3.357 |
| 0.920 | 3.780 | 3.580 | 3.379 | 3.111 |
| 0.963 (noise ceiling) | 3.569 | 3.372 | 3.160 | 2.870 |

Severe band, same construction, minimum severe MAE subject to **severe** slope ≥ 0.60:

| sev_corr ↓ / res_corr → | 0.406 | 0.510 | 0.600 | 0.700 | 0.879 (noise ceiling) |
|---|---|---|---|---|---|
| **0.813** | **slope 0.60 unreachable at any gain** | 8.08 | 6.90 | 6.01 | 4.91 |
| 0.875 | **unreachable** | 7.58 | 6.36 | 5.46 | 4.28 |
| 0.920 | 9.62 | 7.21 | 6.01 | 5.09 | *3.79* |
| 0.963 | 9.00 | 6.87 | 5.66 | 4.73 | *3.27* |

**The severe band clears 4.0 dB at slope 0.6 only when the model is an information oracle on both axes**
(sev_corr ≥ 0.92 *and* res_corr ≥ 0.879 = the label-noise ceiling). At today's spatial channel, severe
slope 0.60 is unreachable at *any* output gain: raising the gain amplifies the error field faster than
the signal within that stratum.

### 2.5 The strongest single argument on the severe band
**Persistence — handed the eye's own real prior VF, which strictly dominates any information a photo
could carry about the current field — scores severe-band MAE 4.461 dB and slope 0.768.**
Fundus-only is at 7.251 / 0.251. The Bayes floor is 3.16 / 0.768. Clearing severe MAE < 4.0 therefore
requires *beating a same-eye VF measurement* from a photograph. That is not happening.

### 2.6 Statistical power at 631 records / 263 eyes / 144 patients
Cluster bootstrap over the 144 PatientIDs (2,000–3,000 reps):

| quantity | estimate | SE | 95% CI |
|---|---|---|---|
| pooled MAE | 4.256 | **0.179** | [3.90, 4.62] |
| pooled slope | 0.543 | 0.026 | [0.487, 0.593] |
| severe MAE | 7.251 | 0.191 | [6.88, 7.63] |
| severe slope | 0.251 | 0.029 | [0.192, 0.306] |
| **paired** ΔMAE (m2rnfl − m1sev) | −0.034 | **0.037** | [−0.106, +0.033] |

- **Two different MDEs, and the distinction matters.** The *paired-bootstrap* MDE (80% power, α = .05,
  two-sided = 2.80 × 0.037) = **0.10 dB**, but that SE only measures test-set resampling for **two fixed,
  co-trained models** whose per-point errors are 94% correlated (m1sev vs m2rnfl). It does **not** capture
  retraining variance. The *method-level* MDE — what a genuinely new method must clear to survive
  refitting — is governed by the fold spread: per-fold paired ΔMAE (m2rnfl − m1sev) ranges −0.19…+0.05,
  SE-of-mean 0.042 (4 df) ⇒ **method-level MDE ≈ 0.12 dB**. Use 0.12, not 0.10, as the go/no-go floor.
  The paired bootstrap's resolution advantage over an unpaired comparison is **~5× on the SE** (0.179 /
  0.037), not "30×."
- **Absolute claims are far weaker than paired ones.** To assert *population* MAE < 4.0 with 95%
  confidence you need a point estimate ≤ 4.0 − 1.96(0.179) = **3.65**. A headline of "3.99" would be
  scientifically empty at this sample size — its CI runs to 4.34. (This is a claim about the *population*;
  a claim only about *these 631 records* is stronger but less interesting — §6.5 rule 5 uses the
  population framing deliberately, since a method that only wins on this fixed dataset is not a result.)
- This retroactively validates the M2 verdict: ΔMAE −0.034 with CI [−0.106, +0.033] is a genuine wash.

---

## 3. (a) Where we genuinely stand vs the comparator

### 3.1 TDV-Net — *Deep learning-based prediction of 24-2 visual field from fundus photographs in glaucoma*
Graefe's Arch Clin Exp Ophthalmol (2026), doi 10.1007/s00417-026-07141-3.
Train 31,443 fundus photographs / 27,364 VFs; test 6,436 photographs / 5,647 VFs, two tertiary hospitals.
Reported pointwise MAE **3.91 dB overall; 3.09 / 5.66 / 9.15 for mild / moderate / severe**.

**Their metrics are not computed on a comparable stratum. We win moderate + severe decisively; the
pooled inversion is case-mix.** The stratum comparison depends on whether strata are cut per *eye* or per
*point*, and the abstract does not say which they use — so both are shown:

| stratum | ours **eye-level** (`m1sev`) | ours **point-level** | TDV-Net | Δ (eye / point) |
|---|---|---|---|---|
| mild | **2.760** | 3.070 | 3.09 | −0.33 / **−0.02 (tie)** |
| moderate | **5.440** | 4.293 | 5.66 | −0.22 / −1.37 |
| severe | **7.251** | 8.438 | 9.15 | **−1.90** / −0.71 |
| pooled | 4.256 | 4.256 | **3.91** | +0.35 |

- *Eye-level* (an eye is severe if its mean sensitivity < 15 dB; all 52 of its points count as severe):
  we beat them in all three, but the mild "win" is fragile.
- *Point-level* (each point bucketed by its own true sensitivity): the mild win **collapses to a tie**
  (3.07 vs 3.09, well inside ±0.18 SE) and the severe advantage shrinks to −0.71. **Moderate and severe
  wins survive under both conventions; the mild claim does not.** Do not headline "every stratum."

The pooled inversion is pure composition. Our OOF is **16.0% severe points**; *any* mix of their three
stratum MAEs that yields 3.91 requires **≤13.5% severe**.

- TDV-Net scored under **our** composition: **4.805 dB**.
- Our model scored under **any** composition consistent with their 3.91: **3.37 – 3.62 dB** (all 22
  feasible mixes; ours beats 3.91 under every one).

So: **our fundus-only model is ~0.3–0.55 dB better than the 31,443-photograph SOTA at matched case mix**,
on a strictly patient-disjoint 5-fold OOF. The severe-eye advantage is −1.90 dB eye-level / −0.71 dB
point-level; report the convention. This is the honest headline.

*Assumptions I could not verify (paywalled full text; abstract only):* (i) TDV-Net's name implies it
predicts **total deviation**; MAE on TD equals MAE on sensitivity for the equivalent predictor, since the
age/location normal reference is a deterministic lookup — but I could not confirm they score it that way.
(ii) Their severity cut-points are presumably Hodapp–Parrish–Anderson on MD; ours are mean-sensitivity
(<15 / 15–22 / ≥22 dB), which map to MD < −13 / −13…−6 / > −6 — near-identical, and if anything our
"severe" is *stricter*, understating our advantage. (iii) Their split hygiene is not stated.
(iv) Their test MAE has a tiny SE; ours is ±0.18.

### 3.2 MLEDL — the *actually* comparable study (npj Digit Med 2025, PMC12216876), 633 patients / 1,129 eyes
This is our exact data regime. Five models, target = raw sensitivity in dB:

| model | inputs at inference | pointwise MAE |
|---|---|---|
| EDL (original CFP) | **fundus only** | **4.131** |
| EDL (ROI) | **fundus only** (disc-centred ROI crop) | **3.903** |
| EDL (ROI + OD/OC) | fundus + disc/cup segmentation | 3.980 |
| MEDL | + sex, age, IOP, CCT, medical history | 3.575 |
| LEDL | + time interval (longitudinal) | 3.098 |

**This corrects a load-bearing error in `decoder/specs/breakthrough_design.md` §0/§4**, which cites
"MLEDL hit 3.1–3.9 fundus-only at 633 pts — the target is inside the achievable region" and used it as
the ambition check justifying the whole sub-4.0 program. It is wrong: **3.098 is longitudinal and 3.575
uses clinical metadata. The strictly fundus-only band is 3.90–4.13.** Our 4.256 sits inside it — on a
sicker population (16% severe points) and a strictly patient-disjoint split, where MLEDL's abstract and
methods never state patient-disjointness (1,300 train / 298 test *records* from 1,129 eyes). Our own
history quantifies exactly this: the leaky per-record split read 4.12 where the clean per-patient split
read 4.55 — a **0.43 dB** inflation. MLEDL's 3.903 is plausibly ≈4.3 under our eval.

Two further readings, both material:
- **ROI cropping is the only demonstrated fundus-only gain in the literature: −0.228 dB.** That is the
  single best-supported lever available to us (→ P1).
- **Adding OD/OC segmentation made it worse** (3.980 > 3.903). Rules out the segmentation branch that
  `breakthrough_design.md` §M5 proposes.

### 3.3 The severity channel is already at the literature ceiling
Sci Rep 2020 (PMC7712913) predicts **MD only** from monoscopic disc photos, 563 eyes / 327 patients,
patient-disjoint external test: r = 0.755, MD MAE = 1.94 dB, test MD = −1.37 ± **3.96** dB.

Correlations are not comparable across cohorts with different MD spreads (ours: SD **5.912** dB). The
scale-free comparison is MD MAE / SD(MD):

| | MD MAE | SD(MD) | MAE/SD | implied r |
|---|---|---|---|---|
| Sci Rep 2020 (disc photo only) | 1.94 | 3.96 | 0.490 | 0.755 |
| **ours (`m1sev`, fundus only)** | **2.629** | 5.912 | **0.445** | 0.813 |

*(The arithmetic self-checks: their reported r and MAE imply SD(MD) = 3.71 under an MMSE predictor;
their actual test SD is 3.96.)*

**Our fundus→severity channel is ~9% better than the best comparable published fundus-only result**, and
theirs was on a cohort that was 40% normals (an easier, partly classification-like problem). Nobody has
demonstrated the sev_corr ≈ 0.875 that our frontier requires. **Open Risk R2** records the counter-reading.

---

## 4. (b) Surviving proposals

Effect sizes are stated against the **method-level MDE = 0.12 dB** (the fold-spread threshold of §2.6; the
finer 0.10 dB paired-bootstrap MDE applies only when comparing two already-trained checkpoints). Anything
below 0.12 is **sub-noise / unfalsifiable at n = 144 patients** and must not be run as a standalone experiment.

None of these clears 4.0 at slope 0.60. They are listed because they are the only things with positive
expected value, and because two of them (D1, D2) can *overturn the verdict*.

### D1 — Learning curve for the two information channels ★ run this first; it is the verdict's weak point
**Claim under test:** the verdict says "information-limited, not data-limited." That is the least
defensible of the three claims (Open Risk R1).
**Method (no training of the proposed model):** cache frozen RETFound features once over the 631 images
(one encoder pass). Fit the *severity* probe (features → eye-mean) and a *spatial* probe (features →
within-eye residual, with the population template as a fixed offset) at n = 40 / 80 / 120 / 170 / 210
training **eyes**, 20 patient-disjoint resamples each, scored OOF on the frozen folds. Plot sev_corr and
partial-corr(pred, eye-specific residual | template) versus n.
**Falsification of the verdict:** if sev_corr is still rising by ≥ 0.02 per doubling at n = 210, or the
spatial partial-corr is rising at all, the ceiling is a sample-size artifact and §0 is wrong.
**Falsification of D1 itself:** flat curves beyond n = 120 ⇒ information-limited, verdict stands.
**Cost:** one encoder pass + minutes of ridge/MLP fitting. **Expected effect on MAE: none — it is a
diagnostic.** This is the highest-value experiment in the document.

### D2 — Is *any* eye-specific spatial signal extractable from the frozen features? ★ cheap, decisive
**Claim under test:** partial-corr = 0.196 is the *model's*; it may not be the *features'*.
**Method:** with the cached features, fit an OOF probe of the true within-eye residual **using the
population template as a fixed offset**, i.e. predict only the eye-specific part. Report OOF partial-corr
and its patient-bootstrap CI. Do it for CLS, all-patch, retinotopic-patch, disc-crop, and multi-layer
features, and for their concatenation.
**Falsification:** OOF partial-corr ≤ 0.25 for every feature set ⇒ the eye-specific spatial channel is
empirically absent from RETFound features at this n; P2 and every spatial-decoder idea die with it.
If some feature set reaches ≥ 0.35, P2 becomes the priority and §0's pooled verdict weakens materially.
**Note:** Session 3's combined frozen probe (full ⊕ disc ⊕ multilayer, 7168-d) reached eyeCorr 0.497 —
*below* the template's 0.488 + noise. Prior probability of D2 succeeding is low. Run it anyway: it is
hours, and it is the load-bearing premise of everything spatial.

### P1 — ROI/disc-centred crop as the **sole** encoder input, at native resolution ★ the only lever with a literature-anchored, above-MDE effect
**Mechanism:** GRAPE images are 2136×2136; we resize the *whole* image to 224×224, so the optic disc
occupies ~30×30 px and the peripapillary RNFL is destroyed. A disc-centred crop (existing
`DISC_HALF = 0.27` box, OD cx 0.78 / OS cx 0.22) is ~1153×1153 native px → 224, i.e. **5.1× finer on the
disc**. MLEDL's EDL(ROI) beat EDL(original CFP) by exactly this substitution: **4.131 → 3.903**.
**What is different from Iteration 11 (which failed):** iter-11 added the disc crop as an *extra view*
and fused by **prediction averaging**, training one shared model on a mixture of macula-centred and
disc-centred inputs while applying the macula retinotopic Garway–Heath prior to disc pixels. It also
polluted the ensemble (the disc-trained model scored 4.34 on full images). **P1 does not average and does
not mix views**: the disc ROI *replaces* the input, so the anatomical prior is re-derived once, for one
view. This exact configuration is untried here and is the only published fundus-only gain.
**Target:** sev_corr (the disc carries the axon-count signal that determines MD).
**Expected effect:** −0.10 to −0.25 dB pooled (literature point estimate −0.23); sev_corr 0.813 → 0.83–0.85.
**Above MDE** in the upper half of its range; the lower half is at the boundary.
**Falsification:** full 5-fold OOF pooled MAE > 4.15 **or** sev_corr < 0.83 ⇒ dead, revert.
**It does not clear 4.0.** Best case (sev_corr 0.85) puts MAE-at-slope-0.60 at ≈ 4.16.
**Known risk:** the crop discards the macula, which subserves the central 24-2 points. MLEDL's ROI may be
wider than our disc box. Pre-register a second crop scale (`DISC_HALF` 0.27 and 0.40) as the *only*
permitted tuning, decided on fold 0 before the full CV.

### P2 — Dual-view feature-level fusion with a disc-specific angular prior — **conditional on D2**
**Mechanism:** fuse full-image and disc-crop patch tokens *inside* `PerPointAttention`, with a
**disc-specific angular prior** (each Garway–Heath VF sector ↔ its optic-disc clock-hour arc) rather than
the macula retinotopic Gaussian, plus a LoRA adapter on the disc branch only (frozen RETFound is
out-of-distribution on zoomed crops).
**Gate:** do not build unless D2 shows a feature set with OOF partial-corr ≥ 0.35.
**Expected effect:** −0.05 to −0.15 dB. **Borderline; the lower half is sub-noise.**
**Falsification:** OOF partial-corr(pred, eye-specific residual | template) does not exceed 0.196 by
≥ 0.05, or severe-band MAE 95% CI upper bound > +0.15 dB vs baseline ⇒ dead.
**Contrary evidence on the record:** the frozen combined probe already failed (§3.2, Session 3).

### P3 — The reductive model: severity head + fixed population template, **no learned spatial head** ★ highest information per GPU-hour
**Not an improvement — a decisive experiment.** §2.3 shows the learned spatial head is worth 0.056 dB.
Delete it. Emit `pred = MD_head(image) + g · template(laterality)`, with `g` fit on train folds.
**Why run it:** (i) it removes ~425k parameters from a 144-patient problem, so if severity is
variance-limited rather than information-limited, sev_corr should *rise*; (ii) if sev_corr does **not**
rise, that is the strongest available evidence for the information ceiling, and it is evidence a reviewer
cannot dismiss; (iii) it is the honest model of what the fundus actually tells us.
**Expected effect:** pooled MAE 4.25–4.35 (a wash by construction), slope ≈ 0.51 (worse).
**Falsification / interpretation:** sev_corr ≥ 0.84 ⇒ the severity channel was capacity/variance-limited
and P1+P3 stack becomes interesting. sev_corr ≤ 0.82 ⇒ verdict §0 confirmed, stop.
**This is the experiment I would run second, after D1.**

### P4 — Evaluation-protocol change (no MAE effect, largest scientific payoff)
Report, for every headline number: patient-bootstrap 95% CI; the per-stratum table; the
severity/spatial decomposition; **the disattenuated slope** (raw slope ÷ stratum reliability: 0.845
pooled, 0.768 severe); and **MAE normalised to the stratum Bayes floor**. Under this protocol our
existing model reports a pooled disattenuated slope of **0.643** (clears 0.60) and a mild-band MAE of
2.76 against a Bayes floor of 1.75. See §6.

---

## 5. (c) Ruled out — and why

Each entry states the *evidence*, not a hunch. "Sub-noise" = expected effect < 0.12 dB method-level MDE (§2.6).

| # | proposal | why it is dead |
|---|---|---|
| 1 | **Any learned within-eye spatial mechanism** (attention sharpening, cross-point refinement, multi-layer fusion, retinotopic priors) | The image's eye-specific spatial contribution is worth **0.056 dB, CI [−0.121, +0.007]** — real (partial-corr 0.196, p < 0.001) but MAE-negligible. Per-eye residual corr: model 0.471 vs no-image template 0.473 (Δ = −0.003, CI [−0.022, +0.014]). The channel is *present but economically dead* at this shrinkage/n; the frontier (§2.4) needs res_corr 0.60, and no method here or in the literature has reached it. **Ruled out only for MAE, and only conditional on D2** — the un-shrunk blend does lift res_corr to 0.453, so a slope/scatterplot-focused variant is not excluded. |
| 2 | **Optic disc/cup segmentation branch** (`breakthrough_design.md` §M5) | MLEDL measured it: ROI + OD/OC = 3.980 vs ROI alone = 3.903. Segmentation **hurt**. |
| 3 | **fundus→RNFL auxiliary head (M2)** | Full 5-fold OOF measured. Paired ΔMAE −0.034, CI [−0.106, +0.033]; pooled r flat 0.665→0.666; severe worse +0.12. A true wash. Nothing about the mechanism has changed. |
| 4 | **Severity de-shrink / mean-residual / bias heads (M1 family)** | M1 already drove sev_shrink 0.71 → **0.98**. A uniform shift cannot manufacture correlation; sev_corr stayed 0.813. The lever is exhausted. |
| 5 | **Balanced-MSE / dispersion / any slope-targeting loss** | Three independent runs. All lie *inside* the frontier already reachable by free post-hoc variance-matched calibration (slope 0.646 at MAE 4.495). Slope ≥ 0.60 is already free; MAE is the constraint, and no reweighting of a fixed-r predictor lowers MAE. |
| 6 | **Post-hoc calibration as an improvement** | Moves along an r-bounded frontier; cannot lower MAE and raise slope together. Keep it as a *reporting* device only. |
| 7 | **Seed / snapshot / K-model ensembles, EMA, SWA** | Two nulls (2- vs 3-seed: 4.116 vs 4.120; 3-model global-head: 4.470 vs single 4.46). *Caveat:* both are from the **leaky per-record split era** (~0.4 dB easier regime than the current per-patient `cv_long`), so the levels don't transport to the m1sev anchor — but the *mechanism* (same-architecture members share correlated severity errors, which are ~94% of the loss) does. Expected effect < 0.08 dB ⇒ **sub-noise**. A cross-*fold* ensemble on the clean split is the one untested variant; low prior. |
| 8 | **Encoder capacity: full-block unfreeze, LoRA rank ↑, more LoRA blocks** | Full unfreeze: train r 0.72, val r 0.54. Rank 16 < rank 8. Rank 8 pooled: r 0.657, unchanged; severe worse. r is pinned at 0.657–0.666 across frozen / LoRA / M1 / M2 — four families, spread ±0.005, far below the fold spread. |
| 9 | **Target denoising (Theil–Sen, VF-AE, GP/Kalman)** | We sit **2.0 dB above** the pooled Bayes floor. Denoising is a second-order variance reduction when you are that far from the noise ceiling. Method B measured it: labels moved 1.24 dB, MAE moved +0.04. **Sub-noise.** |
| 10 | **Manifold / archetype structured output head** | Two independent kills. (i) The existing `pretrained_vf_ae.pth` has a **64-d latent for a 52-d target — it is not a bottleneck at all**, so its 1.8 dB reconstruction error is pure information loss. (ii) A real basis imposes a hard floor: PCA on 14,433 complete UWHVF fields gives reconstruction MAE **1.579 dB at k=16, 1.017 dB at k=32**. More fundamentally, constraining output *shape* cannot supply scotoma *location*, and location is precisely what partial-corr = 0.196 says we don't have. It is a reparameterisation of the calibration frontier: slope ↑, MAE ↑. |
| 11 | **Heteroscedastic / inverse-variance point weighting** using the §2.1 σ_ε(s) profile | Genuinely untried and well-motivated (the 5–15 dB band is 9.9% of points at σ_ε ≈ 6 dB). But the current recipe uses `--sector-combine sector_only`, so the per-point weight is the GH **spatial sector** weight and is already value-agnostic — the premise "we up-weight noisy points" is false. Efficiency gain 0.02–0.08 dB ⇒ **sub-noise / unfalsifiable**. It would also worsen severe MAE by construction. |
| 12 | **Distillation from the longitudinal teacher** | The teacher *is* persistence (its delta head adds nothing above 3.185). Its entire edge is a prior-VF **level** anchor the student cannot recover from a photo — the textbook LUPI worst case, whose failure mode is regression-to-mean, i.e. *lower* slope. Distilling only its residual field reduces to target denoising (#9). |
| 13 | **Chasing severe-band MAE < 4.0 or slope ≥ 0.60** | Persistence, holding the eye's own real prior VF, achieves severe MAE **4.461** and slope 0.768. The Bayes floor is 3.16. Severe slope ≥ 0.60 is unreachable at any output gain while res_corr ≤ 0.41 (§2.4). Not a model problem. **Report it; do not chase it.** |
| 14 | **Photometric augmentation** (ColorJitter) | Measured **worsening** of +0.12 to +0.21 dB (4.36 vs 4.15/4.24), above the MDE — not a null. Conclusion (don't use it) is if anything firmer. |

---

## 6. (d) Pre-committed evaluation protocol — fixed **before** any training

Deviating from this after seeing results invalidates the experiment. Write it down, commit it, then train.

### 6.1 Folds and data
- The **frozen** `decoder/results/cv_long/fold{0..4}_{train,val}.json`: per-patient, eye-disjoint,
  severity-stratified 5-fold over 631 records / 263 eyes / 144 patients. **Never re-split. Never reseed.**
- Score **out-of-fold**, pooled over all 631 records, against the **RAW observed VF**. Denoised or
  smoothed targets are a training device; reporting a metric against them is forbidden.
- Any per-fold quantity (calibration gain, template, normalisation) is fit on that fold's **train** split
  only. The template gain must be re-fit per fold (measured: 0.80–0.88); using a globally-fit gain
  biases the baseline optimistically.

### 6.2 Primary endpoint (exactly one)
**Pooled OOF MAE against the raw VF.** Everything else is secondary.

### 6.3 Secondary and mandatory-companion metrics
Reported together, always, in one table — never a raw slope beside a calibrated MAE:
- raw slope `polyfit(T, P, 1)[0]`, Pearson r, eyeCorr, bias, σ_p/σ_t;
- variance-matched-calibrated MAE and slope (gain fit on the fold's train);
- **disattenuated slope** = raw slope / stratum reliability (pooled 0.845, severe 0.768, moderate 0.713,
  mild 0.649 — measured in §2.2, frozen as constants);
- severity-stratified table (severe < 15 / moderate 15–22 / mild ≥ 22 dB mean sensitivity), with n;
- severity/spatial decomposition: sev_corr, sev_shrink, sev_MAE, res_corr, res_shrink;
- **NEW, mandatory:** partial-corr(pred residual, true residual | OOF population template) and the
  incremental R² of the model over the template. *No spatial claim may be made without these.*
- **NEW, mandatory:** MAE as a multiple of the stratum Bayes floor (pooled 2.25, severe 3.16,
  moderate 2.78, mild 1.75 dB).

### 6.4 Uncertainty
- **Cluster bootstrap over the 144 `PatientID`s**, 2,000 resamples, for every headline number.
- Model-vs-model comparisons are **paired**: identical patient resamples, report Δ and its 95% CI.
  Note this SE captures *test-set* precision for two *fixed* models only; it understates the variance a
  method must survive across *retraining*. Report the per-fold paired ΔMAE (5 values) alongside it — if
  the fold-to-fold sign is unstable (as M2's was: −0.19…+0.05), the bootstrap CI is over-optimistic.
- Fold-level SDs (4 df) are the *method-level* uncertainty and set the go/no-go threshold below; the
  paired bootstrap is a finer-grained companion, not a replacement.

### 6.5 Decision rule (binding)
1. **Do not run** any method whose pre-registered expected effect is < **0.12 dB** (the method-level MDE
   from the fold spread, §2.6 — not the 0.10 dB fixed-model bootstrap MDE, which understates retraining
   variance). Record it as sub-noise instead. This kills #7, #9, #11 in §5 before they cost a GPU-hour.
2. **Promote** a method iff, on the full 5-fold OOF:
   - paired ΔMAE point estimate ≤ **−0.12 dB** *and* its 95% CI excludes 0 *and* the per-fold ΔMAE sign
     is negative in ≥ 4 of 5 folds (guards against the M2 failure mode, where one fold carried the mean); **and**
   - severe-band paired ΔMAE 95% CI upper bound < **+0.15 dB** (a pooled gain that worsens severe is
     not a win — the project's standing rule, now given a numeric definition); **and**
   - raw slope not worse than baseline.
3. **Single-fold gates are advisory only.** M2's fold-2 gate showed ΔMAE −0.19 / Δr +0.03 and the full CV
   was a wash. A fold gate may kill a method; it may never promote one.
4. **Multiplicity.** Pooled MAE is the single primary endpoint; all others are secondary and
   exploratory. If more than one method is tested against the same baseline, apply Holm across the
   primary endpoints.
5. **Claiming "MAE < 4.0."** Permitted only if the patient-bootstrap **95% upper bound < 4.0**, i.e. a
   point estimate ≤ **3.65**. A point estimate of 3.99 (CI to 4.34) may not be reported as clearing the
   threshold. The same rule applies to slope ≥ 0.60 (lower bound must exceed 0.60).
6. **Comparator claims** must be composition-adjusted, per §3.1: report our per-stratum MAEs beside
   theirs, plus our pooled MAE recomputed under their implied composition, plus theirs under ours.
7. **Stop rule.** If D1 shows flat learning curves and D2 shows OOF partial-corr ≤ 0.25 for all feature
   sets, terminate the sub-4.0 program and write up the ceiling. That is the successful outcome.

### 6.6 What we will publish either way
The negative result is stronger science than a 3.99 would have been:
1. Fundus-only 24-2 prediction **is severity estimation**. The eye-specific within-eye spatial
   contribution of a fundus photograph is **0.056 dB (CI [−0.121, +0.007])** — indistinguishable from a
   population template. Reported with the template ablation (§2.3), which no comparable paper runs.
2. Our fundus-only model **beats the 31,443-photograph SOTA in every severity stratum**, and by 1.9 dB in
   severe eyes; the pooled inversion is a case-mix artifact (§3.1).
3. The severe band is **noise-bounded, not model-bounded**: Bayes floor 3.16 dB, reliability 0.768, and
   a same-eye prior VF only reaches 4.461 dB (§2.5). Papers reporting severe-band MAE without this
   context (TDV-Net: 9.15) are reporting perimetric variability.
4. The correct slope to report under errors-in-variables is **disattenuated**; ours is 0.643 pooled.

---

## 7. Open risks (unresolved — recorded, not dissolved)

**R1 — "information-limited, not data-limited" is the weakest claim in §0, and the verdict is explicitly
conditional on it.** The supporting evidence is indirect and partly points the *other* way: (i) the fold-0
LoRA champion `lorac8_f0` reached r 0.734 / eyeCorr 0.552 / res_corr 0.495 (`improvement_log.md:207`) yet
did **not** replicate pooled — a *variance* signature, which is what you see when a real signal is
starved of data at 144 patients, i.e. it argues **data**-limited; (ii) TDV-Net's 50× data does not beat us
per stratum, but that is confounded by cohort, architecture, and target (TD vs sensitivity). **No learning
curve has ever been measured on this problem.** D1 is designed to settle it and must be run before the
verdict is published. If sev_corr is still climbing at n = 210 eyes, §0 is wrong and the correct
recommendation is "get more paired data," not "stop." This is why §0 is framed as a strong prior with a
named falsifier, not a proof.

**R2 — the severity-ceiling comparison may cut the other way.** §3.3 compares scale-free MD error and we
win (0.445 vs 0.490). But a range-restriction correction of the *correlation* transports their r = 0.755
at SD 3.96 to **r ≈ 0.864** at our SD 5.912 — above our 0.813, and essentially at the 0.875 the frontier
demands. I believe the correction over-states their model, because their r is inflated by a 40%-normal
cohort (a partly bimodal, classification-like problem) and the correction assumes homoscedastic linear
attenuation, which perimetric noise violates. **But I cannot rule it out from the abstract.** If someone
obtains their glaucoma-only subgroup r, this risk resolves. Until then, "our severity channel is at the
literature ceiling" is *supported* but not *proven*.

**R3 — the 0.056 dB spatial ablation is a statement about *this* model, not about *all* models.**
It bounds what the current decoder extracts, not what the pixels contain. Its own statistics already say
the channel is *present* (partial-corr 0.196, p < 0.001), just MAE-negligible at the operating shrinkage;
the open question is whether a *better* model could make it MAE-relevant. D2 is the correct test, and it is
gated on frozen features — a LoRA-adapted encoder could in principle beat a frozen probe. The fold-0 LoRA
champion reached eyeCorr 0.552 and res_corr 0.495 (> template ~0.488), `improvement_log.md:207`, then
failed to pool — the one datapoint arguing the eye-specific channel is real and merely unstable at 144
patients. **R3 and R1 are the same risk wearing different clothes.**

**R4 — TDV-Net's target may be total deviation, and its per-stratum n is unknown.** The stratum-wise
dominance in §3.1 assumes their MAE is on a comparable scale and their strata are HPA-defined. If they
score TD against a normal reference we cannot reproduce, the −1.90 dB severe advantage is softer than
stated. The composition argument itself is robust (it uses only their own three numbers and their own
pooled number), but the *level* comparison is not. Obtain the full text before publishing §3.1.

**R5 — `deep_cfg` may still be active in the champion recipe.** §5 #11 asserts the per-point weight is
value-agnostic under `--sector-combine sector_only`. I verified `weights = sector_w` on that branch
(`training.py:1119-1120`), but `deep_cfg`'s asymmetric `overpred_penalty` is applied *after*, to the
Huber term (`training.py:1131-1136`), independently of `sector_combine`. Whether `run_cv.py` passes a
non-`None` `deep_cfg` for the `long_global` recipe is **not verified**. If it does, the loss *is*
asymmetrically penalising deep-point over-prediction, and #11's premise is half-wrong. This changes
nothing about the verdict (the effect is still sub-noise) but must be checked before writing it up.

**R6 — 631 records are 263 eyes.** Records from the same eye at different visits share the same
anatomy and nearly the same field. Folds are eye-disjoint, so this is not leakage, but the *effective*
sample size for every metric is closer to 263 (or 144 patients) than 631, and the pooled MAE is
dominated by whichever eyes contribute the most visits. All CIs here are patient-clustered, which
handles it. Any future analysis that bootstraps *records* will understate uncertainty by ~√2.4.

**R7 — the anchor 4.256 is mildly model-selection-optimistic, which makes the verdict *safer*, not more
fragile.** Each fold's `m1sev_f{f}_best.pth` is the checkpoint minimising that fold's own val score
(`--select mae_slope`) over ~21 val evaluations (`train_lora_cached.py`; selected epochs are early —
f0 e14, f1 e6, f2 e6, f3 e4 — off the later-epoch ~4.27 plateau). So 4.256 carries ~0.05–0.10 dB of
selection optimism: a fairly-evaluated anchor would be *higher*, i.e. *further* from 4.0, so "sub-4.0 not
reachable" is conservative. The caveat that *does* bite: the external comparators (TDV-Net 3.91, MLEDL
3.90) may not be val-selected, so the *absolute* head-to-head in §3 is asymmetric in their favour by up
to ~0.1 dB — the composition argument (§3.1), which is internal to their own numbers, is unaffected.

---

## 8. Recommended sequence

1. **D1** (learning curves) — settles R1/R3. One encoder pass. If curves are rising, stop reading this
   document and go get paired data.
2. **D2** (eye-specific spatial probe) — settles the spatial channel. Hours.
3. **P3** (reductive severity + template model) — settles whether severity is variance- or
   information-limited. One 5-fold run.
4. **P1** (ROI-as-sole-view) — the only positive-EV improvement. One 5-fold run. Expect 4.05–4.15.
5. Stop. Write up §6.6.

Expected landing if all of D1/D2/P1 go the *optimistic* way: pooled MAE ≈ 4.05–4.15 raw, ≈ 4.16 at
slope 0.60. **Still short of 4.0**, on a point estimate whose 95% CI is ±0.35 dB wide.
The threshold was chosen without reference to what a fundus photograph contains.
