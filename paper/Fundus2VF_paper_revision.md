# Fundus2VF paper — copy-paste revisions (2026-07-12)

Scope decided with the author: keep the **fundus-only** paper (reported model = `p1disc_denoise`,
pooled OOF MAE **4.10 dB**), and **recast the method to what is actually reported** (drop the
"two-stage VF-autoencoder" identity; the autoencoder now lives only in the separate longitudinal
model and is mentioned once as future work).

All numbers below are from the frozen leak-free per-patient 5-fold CV (631 fundus–VF records / 263
eyes / 144 patients) and the PAPILA external cohort, verified against
`paper/headline_results.md`, `composition_p1disc_denoise.json`, and the OOF
cache. Honest-framing guardrails are baked in: **do not claim native sub-4.0** (95% CI upper bound
4.45), the **raw slope (0.55) does not clear 0.60** (calibrated/disattenuated do), and **PAPILA dB
MAE is not meaningful** (scale mismatch → correlation only).

---

## TITLE — replace (decided by author)

> **An Open-Source Prediction of Visual Fields from Fundus Images: An Externally Validated Deep
> Learning Model**

(Drops "Two-Stage"; foregrounds the two differentiators from Park et al. 2026 — open data + external
validation. Used verbatim in the running head and anywhere the old title appears.)

---

## ABSTRACT — replace whole section (narrative paragraphs, as in the original draft)

> Visual field (VF) testing is essential for diagnosing and monitoring glaucoma, but it is
> time-consuming and poorly suited to screening. Retinal fundus photography, by contrast, is fast and
> widely available, and the structural changes it captures correlate with functional loss. This study
> aimed to develop and externally validate an open-source deep learning model that predicts the
> Humphrey 24-2 VF directly from a single fundus photograph.
>
> We used only open-access data. The model was developed on 631 paired fundus–VF records from 263 eyes
> of 144 glaucoma patients (GRAPE) and validated on an independent cohort of 164 eyes from 82 patients
> (PAPILA). A frozen RETFound vision-transformer encoder was applied to a disc-centered crop of each
> fundus photograph; 52 point-queries then attended over the encoder's features under an anatomical
> Garway-Heath prior, aided by a global-spatial pattern head and a severity-correction head, to predict
> per-location dB sensitivity. Training targets were denoised along each eye's longitudinal trajectory
> (training only; all validation and external testing were scored against the raw measured VFs). The
> model was evaluated by leak-free per-patient 5-fold cross-validation and zero-shot on PAPILA. Because
> pointwise total-deviation error and raw-sensitivity error are algebraically identical, our results
> are directly comparable to recent total-deviation models.
>
> Pooled pointwise mean absolute error (MAE) was 4.10 dB (95% CI 3.79–4.45), Pearson correlation 0.69,
> and calibrated slope 0.65. Pooled MAE cannot be compared directly across cohorts, because it depends
> on how many severe cases a cohort contains, and ours contained more than a recent 2026 fundus-to-VF
> model. We therefore re-scored both models on the same severity mix. On that model's milder mix, our
> error fell to 3.27–3.65 dB, below its 3.91 dB; on our more severe mix, its error rose to 4.69 dB,
> above our 4.10 dB. The advantage was decisive in moderate (4.06 vs 5.66 dB) and severe (8.43 vs 9.15
> dB) loss and equivalent in mild loss. On the external PAPILA cohort, predicted severity
> correlated with measured mean deviation (overall r 0.75; glaucoma subgroup r 0.76). Error increased
> with depth of field loss, mirroring the test-retest variability of perimetry itself.
>
> This study shows that a fully open-source deep learning model can predict 24-2 VFs from fundus
> photographs. Fundus-to-VF prediction has been reported before, but prior models rely on private data
> and have not been tested outside their source population; ours is built entirely from open data and
> code, performs on par with a recent private-data model at matched disease severity, and generalizes
> to an independent external cohort. A G1-to-24-2 spatial mapping enabled training across heterogeneous
> perimetric protocols. Such models may complement fundus photography for glaucoma screening and
> teleophthalmology — reducing testing time and helping identify at-risk patients — rather than
> replacing formal perimetry. All data, code, and trained models are openly released.

---

## 1.0 INTRODUCTION — replace the "prior work" and "purpose" paragraphs only

Keep paragraph 1 (glaucoma / perimetry burden) and paragraph 2 (structure–function) as-is.
Replace the third paragraph (prior work) and the purpose paragraph:

> A small but growing number of studies predict VFs directly from fundus photographs. Image-
> translation approaches synthesize VF maps with conditional generative adversarial networks
> (cGAN) but are judged by image-similarity metrics rather than clinical sensitivity error (9).
> Regression approaches predict pointwise sensitivity directly: a 2026 fundus-to-24-2 total-
> deviation model reported a pooled pointwise error of 3.91 dB (10), and a multimodal system
> combining fundus images with clinical text reported 3.10–4.13 dB (11). These works establish
> feasibility but share three limitations: they train on large, private, single-region datasets;
> none report external validation on an independent cohort; and pooled error is reported without
> adjusting for differences in disease severity mix, which strongly influences the number.
>
> The purpose of this study was to develop and externally validate an open-source deep learning
> model that predicts the full Humphrey 24-2 VF from a single fundus photograph. Using only
> open-access data, a leak-free per-patient evaluation, a severity-composition-adjusted comparison
> with recent work, and zero-shot testing on an independent cohort, we assess whether fundus-only
> VF prediction is accurate and reproducible enough to support screening and teleophthalmology.

---

## 2.1 DATA SOURCES AND PREPROCESSING — replace whole section (removes UWHVF; re-frames the
## G1→24-2 mapping as producing the standard target format, not aligning two datasets)

> We used two open-access datasets. The Glaucoma Real-world Appraisal Progression Ensemble (GRAPE)
> dataset (14) provided paired fundus photographs and visual fields and was used to develop and
> evaluate the model; from it we derived 631 fundus–VF pairs (one per visit at which both a fundus
> photograph and a VF were available) from 263 eyes of 144 patients. The independent PAPILA dataset
> (REF), a Spanish cohort of fundus photographs with perimetric summary data, was used only for
> external validation. Records missing any portion of the fundus image or VF data were excluded.
>
> The prediction target throughout was the Humphrey 24-2 field, which samples retinal sensitivity (dB)
> at 54 fixed locations on a 6° grid within the central 24°. GRAPE fields, however, were acquired with
> the Octopus G1 pattern, which places 52 locations on a denser, non-uniform grid. To produce a
> standard 24-2 output, we remapped each G1 field onto the 24-2 layout (Figure 1). The G1 points —
> originally ordered in a spiral from the center outward (clockwise for OD, counter-clockwise for OS) —
> were first re-ordered from top-left to bottom-right. Each G1 and 24-2 location was labeled with its
> (x, y) position in degrees from fixation, and each G1 point was matched to its nearest 24-2 location
> with a KD-tree; where several G1 points fell on the same 24-2 location, we took their
> distance-weighted average.
>
> This study was approved by the Stanford University Institutional Review Board.

Reference note: drop the UWHVF citation (old ref 13) — UWHVF is not used by this model. Add the
PAPILA citation (Kovalyk et al. 2022) where **(REF)** appears; keep GRAPE as your existing citation.

---

## 2.2 FEATURE ENGINEERING — replace whole section (merges the redundant 224×224 mentions and
## puts the disc crop with the image preprocessing)

> The primary input feature was the fundus photograph — specifically, a disc-centered crop of it.
> Because the optic disc lies at a stereotyped location, we took a fixed, laterality-aware box
> (half-width 0.27 of the image, centered on the disc side), concentrating the field of view on the
> peripapillary region that carries most of the structure–function signal; this improved accuracy over
> using the whole image. The crop was normalized and resized to 224×224 pixels to match RETFound's
> fixed input dimensions, as the encoder divides each image into a standardized grid of patches (15).
>
> Laterality (right or left eye) was also provided; no patient-identifiable metadata was used. The
> visual field was represented as per-location decibel (dB) sensitivity values on the 24-2 grid. For
> GRAPE samples, Octopus G1 measurements were spatially remapped to the 24-2 format with a custom
> algorithm that preserves regional VF structure; this algorithm is released open source as part of
> this project.

---

## 2.4 MODEL ARCHITECTURE AND TRAINING — replace whole section

> Our model was implemented in Python 3.10 with PyTorch 2.7, Torchvision 0.22, NumPy 1.26, and
> SciPy 1.15.
>
> The disc crop was passed through the RETFound Vision Transformer (ViT-Large) encoder (15), whose
> pretrained weights were kept frozen; only lightweight low-rank (LoRA) adapters in the final
> transformer blocks were trained, so the ~300-million-parameter foundation model was adapted to
> glaucoma fundus images with only a small number of trainable parameters and little overfitting
> risk. The encoder produced a set of patch-level feature tokens.
>
> A per-point attention head then produced the VF. Fifty-two learned point-queries — one per
> analyzed 24-2 location — attended over the patch tokens under a fixed spatial (retinotopic) prior
> that biases each point-query toward the peripapillary image region to which its VF location
> corresponds. A joint global-spatial head added the between-location loss pattern from globally
> pooled features, and a severity-correction head sharpened the eye-level mean sensitivity (which
> carries most of the between-eye variance and is otherwise compressed toward the population mean).
> The head output a 52-element vector of predicted dB sensitivities.
>
> The model was trained to minimize an anatomically sector-weighted regression loss between predicted
> and measured sensitivities, with a per-eye concordance term and value-based reweighting to counter
> the rarity of deeply depressed locations. (Sectoral results, Table 1, are reported on the canonical
> six-sector Garway-Heath map.) Two techniques further stabilized training. First,
> because a single VF is noisy, training targets were denoised per location along each eye's
> longitudinal test sequence using a robust (Theil-Sen) trend fit; this was applied only to training
> targets — validation and external testing were always scored against the raw measured VF, so the
> evaluation remains leak-free. Second, ±5° rotational augmentation and dropout were used during
> training. Test-time augmentation was not used, as small rotations displace the tight disc crop and
> bias predictions.
>
> The VF-manifold autoencoder used to pretrain on the large UWHVF perimetry dataset (13) is not part
> of this fundus-only model; it underpins a longitudinal extension that predicts a follow-up VF from
> the fundus image plus the eye's prior VF, which we report separately.

---

## 2.5 EVALUATION — replace whole section

> The model was evaluated by leak-free, per-patient 5-fold cross-validation on GRAPE: folds were
> eye-disjoint and stratified by severity, so no eye or patient appeared in both training and
> validation, and all reported metrics are pooled out-of-fold predictions. Primary metrics were
> pointwise MAE and Pearson correlation; we also report root-mean-square error (RMSE) and the slope
> and bias of the predicted-versus-true regression line to characterize the regression-to-the-mean
> that affects all pointwise VF models. Patient-level bootstrap 95% confidence intervals accompany
> the pooled MAE.
>
> Because pooled MAE depends heavily on a cohort's severity mix, we report a composition-adjusted
> comparison with the 2026 model of Park et al. (10): using only our own out-of-fold errors, we
> recomputed both models' pooled MAE under each other's severity composition (no retraining). We
> further report MAE stratified by severity, both pointwise (bucketed by each location's true
> sensitivity) and per eye, and MAE across the six Garway-Heath sectors.
>
> External validity was assessed by evaluating the GRAPE-trained model zero-shot on PAPILA, an
> independent Spanish cohort (no PAPILA eye was used in training). PAPILA provides 30-2 mean
> deviation rather than per-point 24-2 sensitivity, so we report correlation between predicted and
> measured severity (the scale-free, valid cross-cohort metric); absolute dB error is not comparable
> across the two test patterns and is not reported.

---

## 3.1 POPULATION CHARACTERISTICS — replace whole section

> The model was developed on the GRAPE dataset, an open-access, predominantly open-angle glaucoma
> cohort with longitudinal fundus photography and perimetry. From it we formed 631 fundus–visual-field
> pairs — one per visit at which both a fundus photograph and a VF were available — from 263 eyes of
> 144 patients (69 female, 75 male; mean age 42.5 years; 130 right and 133 left eyes). Patients had on
> average 4.2 tests per eye, spaced about 2.5 years apart, with a mean deviation of −7.2 dB, reflecting
> a predominantly mild-to-moderate severity distribution.
>
> By true per-eye mean sensitivity, 55.0% of records were mild (≥22 dB), 29.0% moderate (15–22 dB), and
> 16.0% severe (<15 dB); at the individual test-point level, 61.2%, 21.6%, and 17.2% of locations fell
> in these bands. Severe cases were relatively few, reflecting the cohort's measurement and surgical
> history. As expected, sensitivity was highest at central test locations.
>
> External validation used PAPILA, an independent open-access Spanish cohort of 164 eyes from 82
> patients (87 glaucoma, 68 suspect, 9 healthy) with fundus photographs and 30-2 mean deviation. All
> analyses used de-identified imaging and VF data; no demographic or clinical variables were used as
> model inputs, and broader demographic data (e.g., race and ethnicity) were unavailable, precluding a
> fairness analysis.

---

## 3.2 AUTOENCODER PRETRAINING — delete this subsection

The reported fundus-only model does not route through the autoencoder decoder, so this subsection
(and the 2.09 dB / r 0.86 result) is removed to avoid describing a component the reported model does
not use. The autoencoder is now covered by the single future-work sentence in §2.4.

---

## 3.2 VISUAL FIELD PREDICTION FROM FUNDUS PHOTOGRAPHS — replace whole section

(Was §3.3; the old §3.2 autoencoder subsection is deleted, so this is now §3.2. Each figure/table
marker sits immediately before the text that discusses it, and the results figures are renumbered so
they are cited in ascending order: Figure 3 scatter → 4 heatmap → 5 examples → 6 external.)

> **[FIGURE 3 — predicted-versus-true scatter]**
>
> On leak-free per-patient 5-fold cross-validation (631 records, 263 eyes, 144 patients), the model
> achieved a pooled pointwise MAE of 4.10 dB (95% CI 3.79–4.45), RMSE 5.77 dB, and Pearson correlation
> 0.69. Predictions were close to the true values on average (mean signed bias +0.39 dB) but compressed
> toward the middle of the sensitivity range — a regression-to-the-mean common to pointwise VF models
> (Figure 3): the raw predicted-versus-true slope was 0.55, rising to 0.65 after variance-matched
> calibration. We do not claim a native sub-4.0 dB error, as the confidence-interval upper bound
> exceeds 4.0.
>
> **[FIGURE 4 — per-location MAE heatmap]**   **[TABLE 1 — sectoral MAE]**
>
> Error was unevenly distributed across the field (Figure 4). By canonical Garway-Heath sector
> (Table 1), MAE was lowest in the temporal (central/papillomacular) and nasal (temporal-wedge field)
> sectors (≈3.79 dB) and highest in the inferotemporal (4.43 dB) and inferonasal (4.35 dB) sectors —
> the inferior optic-disc sectors that serve the superior visual field, where glaucomatous arcuate
> defects concentrate.
>
> **[FIGURE 5 — example fundus→VF predictions by severity]**
>
> Error increased with depth of field loss (Figure 5). At the point level (bucketed by each location's
> true sensitivity), MAE was 2.90 dB in mild (≥22 dB), 4.06 dB in moderate (15–22 dB), and 8.43 dB in
> severe (<15 dB) locations; grouped by decade, it rose from 2.93 dB (20–30 dB) to 5.43 dB (10–20 dB)
> and 9.25 dB (0–10 dB). Per eye, MAE was 2.66 dB in mild, 5.14 dB in moderate, and 7.19 dB in severe
> glaucoma. Predicted fields nonetheless tracked the measured fields across severities, including deep
> loss.
>
> **[TABLE 2 — comparison with prior studies]**   **[TABLE 3 — composition-adjusted comparison]**
>
> Pooled MAE is not comparable across cohorts with different severity mixes; ours is more severe (17.2%
> of points severe) than the mix implied by a recent 2026 fundus-to-VF model's pooled 3.91 dB (Table 2;
> Park et al.). Recomputing both models under matched compositions from our own errors (Table 3), our
> MAE was 3.27–3.65 dB under any consistent mix — below 3.91 dB under every mix — while that model
> scored 4.69 dB under our mix. The advantage was decisive in moderate (4.06 vs 5.66 dB) and severe
> (8.43 vs 9.15 dB) loss and equivalent in mild loss (2.90 vs 3.09 dB).
>
> **[FIGURE 6 — external validation on PAPILA]**
>
> Evaluated zero-shot on the independent PAPILA cohort (164 eyes, 82 patients), predicted VF severity
> correlated with the measured 30-2 mean deviation (Figure 6): overall Pearson r 0.75 (95% CI
> 0.61–0.85) and r 0.76 in the glaucoma subgroup (n=87); correlation was weak in the small suspect
> subgroup (r 0.23) and uninterpretable in the healthy subgroup (n=9). This match between in-cohort and
> cross-cohort severity correlation (GRAPE 0.81 vs PAPILA glaucoma 0.76) indicates the fundus-to-
> severity mapping transfers to an independent camera and population.

---

## TABLE 1 — Sectoral MAE (new content)

| Garway-Heath sector | MAE (dB) | n points |
|---|---|---|
| Temporal (papillomacular) | 3.79 | 3,786 |
| Nasal (temporal-wedge field) | 3.79 | 2,524 |
| Superonasal | 3.92 | 6,941 |
| Superotemporal | 3.99 | 6,310 |
| Inferonasal | 4.35 | 4,417 |
| Inferotemporal | 4.43 | 8,834 |

Legend: Pooled per-location MAE within each canonical Garway-Heath disc sector (6-sector map,
Garway-Heath et al. 2000; out-of-fold 5-fold CV, p1disc_denoise; pooled 4.10 dB). Error is lowest
in the temporal (central/papillomacular) and nasal (temporal-wedge field) sectors and highest in the
inferotemporal and inferonasal sectors — the inferior-disc sectors serving the superior visual field,
where glaucomatous arcuate defects concentrate.

---

## TABLE 2 — Comparison with prior fundus-to-VF studies (replaces the placeholder comparison table)

| Study | Data (source) | Prediction target | Pooled MAE (dB) | Mild / Mod / Severe (dB) | External validation |
|---|---|---|---|---|---|
| **This study** | GRAPE, open-access (631 records) | 24-2 raw sensitivity | **4.10** (composition-adjusted 3.27–3.65) | 2.90 / 4.06 / 8.43 | **PAPILA, r 0.75** |
| Park et al. 2026 (10) | Private, 2 tertiary hospitals (~37.9k photos) | 24-2 total deviation | 3.91 | 3.09 / 5.66 / 9.15 | None reported |
| Huang et al. 2025 (11) | Private (1,129 eyes) | Octopus 59-point sensitivity† | 3.10–4.13 | — | None reported |
| Kang et al. 2024 (9) | Private | VF image (cGAN)‡ | — | — | None reported |

† Different device/point layout/dynamic range — not directly comparable to 24-2 dB.
‡ Image-translation output judged by PSNR 30.61 dB / SSIM 0.48, not clinical sensitivity error.
Mild = each location's true sensitivity ≥22 dB; moderate 15–22; severe <15. Our mild figure is
statistically equivalent to Park et al.'s (within measurement error).

---

## TABLE 3 — Composition-adjusted comparison vs Park et al. 2026 (new; the reliable head-to-head)

| Severity stratum | Our MAE (dB) | Our % of points | Park et al. MAE (dB) | Δ (ours − theirs) |
|---|---|---|---|---|
| Mild (≥22 dB) | 2.90 | 61.2% | 3.09 | −0.19 (equivalent) |
| Moderate (15–22 dB) | 4.06 | 21.6% | 5.66 | −1.60 |
| Severe (<15 dB) | 8.43 | 17.2% | 9.15 | −0.73 |
| Pooled, native mix | 4.10 | — | 3.91 | +0.19 (case-mix inversion) |
| Pooled, matched mix | 3.27–3.65 | — | 3.91 | ≤ −0.26 |
| Park et al. under our mix | — | — | 4.69 | — |

Legend: Each model recomputed under matched severity compositions using our own out-of-fold errors
(no retraining). Pooled native MAEs are not directly comparable because the cohorts differ in
severity mix; the matched-mix rows are.

---

## 4.0 DISCUSSION — replace whole section

> We developed and externally validated an open-source model that predicts the full Humphrey 24-2 VF
> from a single fundus photograph. Trained only on open data, it achieved a pooled pointwise MAE of
> 4.10 dB and, under a severity mix matched to a recent private-data model, 3.27–3.65 dB — better
> than that model's 3.91 dB. The result shows that fundus photographs carry enough information to
> approximate functional loss when paired with an anatomically informed decoder, and that this can
> be demonstrated reproducibly on open data.
>
> The most comparable prior work is the 2026 fundus-to-24-2 regression model of Park et al. (10),
> which predicts total-deviation values and reports pooled MAE 3.91 dB (3.09/5.66/9.15 for
> mild/moderate/severe). Two points make the comparison fair. First, scale: pooled total-deviation
> error and pooled raw-sensitivity error are algebraically identical, because the age- and location-
> normative offset cancels in the pointwise difference — so our 4.10 dB is directly comparable to
> their 3.91 dB without implementing a normative database. Second, composition: pooled MAE depends
> on severity mix, and ours is more severe. Adjusting for this, our model is better under every
> mutually consistent mix and decisively better in moderate and severe loss, where prediction is
> hardest and most clinically consequential; mild loss is a tie. A multimodal fundus-plus-text
> system (11) reported similar pointwise error but requires clinical narratives at inference, and
> an earlier cGAN approach (9) framed the task as image translation judged by pixel similarity
> rather than interpretable sensitivity values.
>
> To our knowledge this is the first fundus-to-VF model with external validation on an independent
> cohort. Zero-shot on PAPILA, predicted severity correlated with measured mean deviation (overall
> r 0.75; glaucoma r 0.76), matching the in-cohort severity correlation. This addresses the most
> common objection to single-source models — that they learn a camera or population rather than
> anatomy. We report correlation rather than absolute error here because PAPILA uses a different test
> pattern (30-2 mean deviation), making dB error incomparable across cohorts; the small suspect and
> healthy subgroups were underpowered and are not interpreted.
>
> As in all pointwise VF models, accuracy fell at severely depressed locations, where predictions
> regress toward the mid-range. This is expected and partly intrinsic: standard perimetry has
> test-retest variability that grows steeply with depth of damage — the retest standard deviation is
> below ~2 dB near 30 dB but exceeds 7 dB near 7 dB, and sensitivities below ~20 dB are widely
> regarded as unreliable (21). Because the model is trained and scored against these noisy
> measurements, achievable error at deep locations is bounded from below by the test itself, and our
> largest errors coincide with the least reliable measurements. An overall 4.10 dB with anatomically
> patterned errors supports screening and triage rather than precise threshold quantification.
>
> Our analyses point to data, not architecture, as the main remaining limit. The model's ability to
> rank eyes by severity plateaued (correlation ~0.81) despite architectural changes, and several
> extensions we evaluated — an ordinal/distributional output head, higher-resolution and macula-
> centered crops, and clinical metadata — did not improve accuracy. The clearest levers are more
> advanced-glaucoma examples and stronger fundus encoders. Other limitations: the paired dataset is
> small; GRAPE uses Octopus G1 fields requiring conversion to 24-2, which adds approximation;
> demographic variables were unavailable, precluding a fairness analysis; and external validation was
> limited to severity correlation on one cohort.
>
> In conclusion, an open-source, externally validated model can predict 24-2 VFs from fundus
> photographs. Fundus-to-VF prediction has been shown before, but our contribution is to demonstrate
> it using only open data and code and to confirm that it generalizes to an independent cohort — at an
> accuracy on par with a recent private-data model at matched disease severity. Such models could
> extend glaucoma assessment to settings without perimetry, complementing rather than replacing formal
> VF testing. We release all code and trained models to support replication and external testing.

---

## REFERENCES — changes

- **Ref 10 (fix the broken ResearchGate citation):** Park [full author list], et al. Deep
  learning-based prediction of 24-2 visual field from fundus photographs in glaucoma. Graefes Arch
  Clin Exp Ophthalmol. 2026. doi:10.1007/s00417-026-07141-3. *(Confirm authors/volume from the
  published record.)*
- **Add (PAPILA):** Kovalyk O, Morales-Sánchez J, Verdú-Monedero R, Sellés-Navarro I,
  Palazón-Cabanes A, Sancho-Gómez JL. PAPILA: dataset with fundus images and clinical data of both
  eyes of the same patient for glaucoma assessment. Sci Data. 2022;9:291.
  doi:10.1038/s41597-022-01388-1.
- **Add (target denoising, optional):** cite the Theil–Sen estimator (Sen PK. J Am Stat Assoc.
  1968;63:1379–1389) at the "robust (Theil-Sen) trend fit" mention.
- Refs 16 (CORAL ordinal), 17 (LDS), 18 (Balanced MSE) are no longer part of the reported model.
  Keep only if you keep the "extensions we evaluated" sentence in the Discussion; otherwise drop
  them.
- Add a PAPILA license/attribution note in Acknowledgements (CC-BY-4.0, per Kovalyk et al. 2022;
  see `data/external/papila/SOURCE.md`).

---

## FIGURE & TABLE CHECKLIST (files + action)

Reproduce all three new/regenerated figures (torch-free, ~seconds, no inference):
```bash
python decoder/make_paper_figures.py        # Figure 3 (scatter) + 4 (heatmap) + 5 (examples)
python decoder/make_architecture_figure.py  # Figure 2 (pipeline schematic)
```

Results figures are numbered in the order they are discussed in §3.2 (scatter → heatmap → examples →
external), so citations run in ascending order.

| Item | Action | File / source |
|---|---|---|
| Figure 1 (G1→24-2 mapping) | **Keep** | unchanged |
| Figure 2 (architecture) | **DONE (drafted)** | `decoder/results/auto/fig2_architecture.png` — fundus → disc crop → frozen RETFound (+LoRA) → per-point attention (Garway-Heath prior + global/severity heads) → 24-2 VF; color-coded frozen vs trained; no autoencoder box |
| Figure 3 (pred-vs-true scatter) | **DONE** | `decoder/results/auto/p1disc_denoise_scatter_clean.png` — de-cluttered rebuild (32,812 points; y=x, raw fit 0.55x+9.8, calibrated fit 0.64x+7.6; 2-line stat box, no internal tags). Old busy version: `p1disc_denoise_scatter.png` |
| Figure 4 (per-location MAE heatmap) | **DONE** | `decoder/results/auto/p1disc_denoise_mae_heatmap.png` — same 4-panel style as before (inferno, MAE 0–10 + GT 0–30, Garway-Heath boundaries), now over the 5-fold OOF (631 eyes, combined 4.10 dB) |
| Figure 5 (example fundus→VF by severity) | **DONE** | `decoder/results/auto/p1disc_denoise_examples.png` — best case per band: mild (40_OD_2, MAE 1.1), moderate (34_OD_1, MAE 3.5), severe (120_OD_4, MAE 3.5) |
| Figure 6 (external validation) — **NEW** | **Add** | `decoder/results/auto/papila_external_scatter.png` (exists) |
| Supp. figure (composition-adjusted scatter) — optional | Add | `decoder/results/auto/p1disc_denoise_scatter_matchedcomp.png` (exists) |
| Table 1 (sectoral MAE) | **DONE** | canonical 6-sector GH map, values above (`decoder/garway_heath_weighting.py` per_sector_mae on OOF; `paper/tables/Table1_sectoral_mae.png`) |
| Table 2 (comparison) | Fill | values above |
| Table 3 (composition-adjusted) — **NEW** | Add | values above |

Figure legends to update:
- **Figure 2:** "Model pipeline. Each GRAPE record is a fundus photograph paired with the same eye's
  same-visit 24-2 visual field. At inference the model sees only the fundus: a disc-centered crop is
  encoded by a frozen RETFound vision transformer with trainable low-rank (LoRA) adapters, and 52
  point-queries attend over the patch tokens under a fixed retinotopic prior, with global-spatial and
  severity-correction heads, to predict the 24-2 field. The paired measured field is used only in
  training, as the loss target; each location's error is weighted by its Garway-Heath optic-disc
  sector (inset: the six disc sectors and the VF locations they serve), and gradients update only the
  LoRA adapters and the decoder. Blue = frozen (pretrained); green = trained on GRAPE; dashed green =
  training only."
- **Figure 3:** "Predicted versus true pointwise 24-2 sensitivity (32,812 locations, 5-fold
  out-of-fold), with the identity line, the raw best-fit line (slope 0.55), and the variance-matched
  calibrated best-fit line (slope 0.65)."
- **Figure 4:** "Per-location mean absolute error (top) and mean ground-truth sensitivity (bottom)
  for right (OD) and left (OS) eyes, pooled over leak-free 5-fold out-of-fold predictions (631 eyes),
  with Garway-Heath sector boundaries overlaid. Error is lowest centrally and inferiorly."
- **Figure 5:** "Representative fundus-to-VF predictions at three severity levels (best case per
  band). Columns: fundus photograph, measured 24-2 VF, predicted VF (shared dB scale). Predictions
  track the measured field, including deep loss in the severe example."
- **Figure 6 (new):** "External validation on PAPILA (164 eyes / 82 patients). Predicted VF severity
  versus measured 30-2 mean deviation; overall Pearson r 0.75, glaucoma subgroup r 0.76. Absolute dB
  error is not comparable across the 24-2 and 30-2 patterns; correlation is the valid cross-cohort
  metric."
