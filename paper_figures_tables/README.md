# Paper figures & tables — Fundus2VF (assembled 2026-07-12)

All figures and tables the manuscript needs, in one place. Content matches the author-decided
revision in `../Fundus2VF_paper_revision.md` and the verified numbers in
`../decoder/results/auto/paper_headline_results.md`.

**Reported model:** `p1disc_denoise` — disc-crop-only recipe with Theil-Sen per-point trajectory
target-denoising (train-only; validation always scored against the raw HVF → leak-free).
**Headline:** pooled pointwise MAE **4.10 dB** (95% CI 3.79–4.45), Pearson r **0.69**, calibrated
slope **0.65**, over leak-free per-patient 5-fold CV (631 records / 263 eyes / 144 patients).

## Honest-framing guardrails (do not violate in the manuscript)
1. **Do not claim native sub-4.0 dB** — the 95% CI upper bound is 4.45.
2. **Raw slope (0.55) does not clear 0.60.** Only the calibrated (0.65) and disattenuated (0.65)
   slopes do. Do not imply the raw slope clears 0.60.
3. **PAPILA dB MAE (~26 dB) is not meaningful** and must not be reported — 24-2 vs 30-2 scale
   mismatch. Report correlation only (overall r 0.75; glaucoma subgroup r 0.76).
4. **Composition-adjusted superiority (3.27–3.65 vs 3.91)** is the reliable head-to-head claim, not
   a native pooled-MAE win (native 4.10 vs 3.91 is a case-mix inversion).

---

## Figures

| # | File | Legend (short) | Source of truth / how to regenerate |
|---|---|---|---|
| **1** | `figures/Figure1_g1_to_242_mapping__NOT_IN_REPO.txt` | G1→24-2 spatial mapping schematic. **KEEP — unchanged.** | ⚠️ Not a repo asset — reuse from `../Fundus2VF Paper Draft.pdf` or export from original drawing. See the .txt. |
| **2** | `figures/Figure2_architecture.png` | Model pipeline: fundus → disc crop → frozen RETFound (+LoRA) → per-point attention (Garway-Heath prior + global/severity heads) → 24-2 VF. Blue = frozen, green = trained. | `python decoder/make_architecture_figure.py` → `decoder/results/auto/fig2_architecture.png` |
| **3** | `figures/Figure3_mae_heatmap.png` | Per-location MAE (top) and mean ground-truth sensitivity (bottom), OD/OS, pooled over 5-fold OOF, Garway-Heath boundaries overlaid. | `python decoder/make_paper_figures.py` → `decoder/results/auto/p1disc_denoise_mae_heatmap.png` |
| **4** | `figures/Figure4_scatter_pred_vs_true.png` | Predicted vs true pointwise sensitivity (32,812 locations, OOF); identity line, raw fit (slope 0.55), calibrated fit (slope 0.65). | `python decoder/scatter_from_oof.py --tag p1disc_denoise` (clean rebuild) → `decoder/results/auto/p1disc_denoise_scatter_clean.png` |
| **5** | `figures/Figure5_examples_by_severity.png` | Representative fundus→VF predictions, best case per band: mild / moderate / severe. Columns: fundus, measured VF, predicted VF. | `python decoder/make_paper_figures.py` → `decoder/results/auto/p1disc_denoise_examples.png` |
| **6** | `figures/Figure6_papila_external_validation.png` | External validation on PAPILA (164 eyes / 82 patients): predicted severity vs measured 30-2 MD; overall r 0.75, glaucoma r 0.76. | `decoder/results/auto/papila_external_scatter.png` |
| **S1** | `figures/SuppFigure_composition_adjusted_scatter.png` | (Optional supplement) Composition-adjusted scatter at matched case-mix. | `decoder/results/auto/p1disc_denoise_scatter_matchedcomp.png` |

**Regenerate the three torch-free figures in seconds (no inference):**
```bash
python decoder/make_paper_figures.py        # Figure 3 (heatmap) + Figure 5 (examples)
python decoder/make_architecture_figure.py  # Figure 2 (pipeline schematic)
python decoder/scatter_from_oof.py --tag p1disc_denoise   # Figure 4 (scatter)
```

### Full figure legends (paste-ready)
- **Figure 2:** "Model pipeline. A disc-centered crop of the fundus photograph is encoded by a
  frozen RETFound vision transformer with trainable low-rank (LoRA) adapters; 52 point-queries
  attend over the patch tokens under a Garway-Heath anatomical prior, with global-spatial and
  severity-correction heads, to predict the 24-2 visual field. Blue = frozen (pretrained);
  green = trained on GRAPE."
- **Figure 3:** "Per-location mean absolute error (top) and mean ground-truth sensitivity (bottom)
  for right (OD) and left (OS) eyes, pooled over leak-free 5-fold out-of-fold predictions
  (631 eyes), with Garway-Heath sector boundaries overlaid. Error is lowest centrally and inferiorly."
- **Figure 4:** "Predicted versus true pointwise 24-2 sensitivity (32,812 locations, 5-fold
  out-of-fold), with the identity line, the raw best-fit line (slope 0.55), and the variance-matched
  calibrated best-fit line (slope 0.65)."
- **Figure 5:** "Representative fundus-to-VF predictions at three severity levels (best case per
  band). Columns: fundus photograph, measured 24-2 VF, predicted VF (shared dB scale). Predictions
  track the measured field, including deep loss in the severe example."
- **Figure 6:** "External validation on PAPILA (164 eyes / 82 patients). Predicted VF severity
  versus measured 30-2 mean deviation; overall Pearson r 0.75, glaucoma subgroup r 0.76. Absolute
  dB error is not comparable across the 24-2 and 30-2 patterns; correlation is the valid
  cross-cohort metric."

---

## Tables

| # | Files | Content |
|---|---|---|
| **1** | `tables/Table1_sectoral_mae.{md,csv}` | Sectoral MAE across the 8 Garway-Heath sectors (3.80–4.37 dB). |
| **2** | `tables/Table2_comparison_prior_studies.{md,csv}` | Comparison with prior fundus-to-VF studies (Park 2026, Huang 2025, Kang 2024). |
| **3** | `tables/Table3_composition_adjusted.{md,csv}` | Composition-adjusted head-to-head vs Park et al. 2026 (the reliable comparison). |

---

## Provenance / notes
- All figures reflect the **frozen leak-free per-patient 5-fold OOF** (`decoder/results/cv_long`)
  for the `p1disc_denoise` reported model. PAPILA external validation (Figure 6) was measured on
  the `p1disc` champion, not re-run for the denoised variant — see `paper_headline_results.md` §4.
- Figures were copied here (not moved); originals remain under `decoder/results/auto/`.
- Figure 4 uses the **de-cluttered** rebuild (`..._scatter_clean.png`). The busier version
  (`..._scatter.png`) and a 13%-severe variant (`..._scatter_sev13.png`) also exist under
  `decoder/results/auto/` if needed.
