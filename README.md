# Fundus To VF Generation

Predicting the Humphrey 24-2 visual field (52 points, dB) of a glaucomatous eye directly from a
single fundus photograph — a frozen [RETFound](https://github.com/rmaphoh/RETFound_MAE) ViT-L
encoder with LoRA adapters, plus a per-point attention decoder under a retinotopic prior.

[Presented at ARVO 2026!](https://www.linkedin.com/feed/update/urn:li:activity:7460094274219520001/)

## Results

Leak-free **per-patient** 5-fold cross-validation, eye-disjoint and severity-stratified —
631 fundus–VF records / 263 eyes / 144 patients from the public GRAPE cohort.

| metric | value |
|---|---|
| pooled pointwise MAE | **4.10 dB** (95% CI 3.79–4.45) |
| Pearson r | 0.69 |
| slope (raw / calibrated / disattenuated) | 0.55 / 0.65 / 0.65 |
| point-level MAE, mild / moderate / severe | 2.90 / 4.06 / 8.43 dB |
| external validation (PAPILA, 164 eyes, zero-shot) | r = 0.75 overall, 0.76 glaucoma subgroup |

**Against the state of the art.** Pooled MAE is not comparable across cohorts with different
severity mixes — ours is sicker (17.2% severe points). Recomputed under any case-mix consistent
with TDV-Net (Park et al. 2026, 31k training photographs), this model scores **3.27–3.65 dB vs
their 3.91**, winning under every feasible mix; TDV-Net scored under *our* mix would be 4.69.
It is decisive in the moderate and severe bands; mild is a tie.

**What is deliberately not claimed:** native sub-4.0 dB (the CI upper bound is 4.45), a raw slope
clearing 0.60 (only the calibrated and disattenuated slopes do), and any absolute dB error on
PAPILA (24-2 vs 30-2 scale mismatch makes it meaningless — correlation only). Full derivation,
confidence intervals, and caveats: [`paper/headline_results.md`](paper/headline_results.md).

A separate **longitudinal** extension predicts a follow-up VF from the fundus image plus the eye's
prior VF and inter-test interval (pooled MAE 3.75); it is reported separately.

## Reproduce

Every published number and figure regenerates from the frozen out-of-fold prediction caches
(`decoder/results/auto/oof_cache_notta/*.npz`, ~160 KB each) — no GPU, no torch, seconds:

```bash
python decoder/composition_report.py --tag p1disc_denoise   # headline + composition-adjusted table
python decoder/scatter_from_oof.py  --tag p1disc_denoise    # predicted-vs-true scatter
python decoder/make_paper_figures.py                        # per-location MAE heatmap + examples
python decoder/make_architecture_figure.py                  # pipeline schematic
pytest                                                      # 54 tests
```

Retraining from scratch, the model architecture, and the data schema are documented in
[`CLAUDE.md`](CLAUDE.md).

## Layout

| path | contents |
|---|---|
| `decoder/` | model, training, evaluation and figure code (the code root) |
| `decoder/results/` | frozen CV folds, per-tag results, OOF caches, run logs |
| `scripts/` | dataset construction and fold-training entry points |
| `paper/` | manuscript revisions, headline results, figures, tables |
| `docs/` | design specs, plans, and the full iteration history |
| `data/`, `encoder/` | inputs and pretrained weights (large files not in git) |

## Data

GRAPE (fundus + 24-2 perimetry) and UWHVF (VF-only pretraining) are public; PAPILA is used for
zero-shot external validation under CC-BY-4.0 with attribution to Kovalyk et al. 2022
(see [`data/external/papila/SOURCE.md`](data/external/papila/SOURCE.md)). Fundus images, the
RETFound checkpoint, and model checkpoints are not tracked in git.
