# Longitudinal Prior-VF Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Predict a visit's HVF 24-2 from `fundus_t + the eye's prior VF(s) + inter-test interval`, reaching honest per-patient 5-fold CV pooled MAE ~3.5 dB (sub-3.74) with calibrated slope ~0.75, leak-free and defensibly framed.

**Architecture:** Frozen RETFound encoder + Garway-Heath per-point attention (mandatory, unchanged) → predicts a per-point DELTA off a prior-VF anchor decoded through a retrained VF autoencoder manifold. Residual-on-persistence (zero-init Δ → starts at the 3.30 persistence baseline). Visit-1 records (no prior) route through a disc structure-function severity branch.

**Tech Stack:** PyTorch/MPS, existing `decoder/training.py` model + loss + CV harness, a new bottleneck VF autoencoder.

## Global Constraints
- Encoder→decoder + Garway-Heath sectoring always on. RETFound frozen. VF autoencoder pretrains on UWHVF only (never GRAPE).
- **Leak-free eval is mandatory:** per-patient folds (patient- AND eye-disjoint) + causal within-eye ordering — a target at visit t may use ONLY that eye's visits `< t` as input; never its own label, never future visits.
- Headline metric = per-patient 5-fold CV pooled pointwise MAE over ALL 631 records, reported with: the **persistence baseline (3.71)**, severity-stratified table, per-eye-averaged MAE, and the visit-1 vs prior-bearing split.
- All 52-d vectors stay in OD/OS query order (`diagnostics.vec52`) so `compute_loss`/`pooled_metrics`/CV apply unchanged.
- Run from repo root; prefix shell with `rtk`.

## Honest-framing guardrails (must hold in reporting)
- Beat **persistence (3.71)**, not just fundus-only (4.29) — beating fundus-only is trivial longitudinally.
- Disclose 263/631 (42%) are visit-1/fundus-only.
- The with-prior gain is **denoising** (regression-to-sector-mean), bounded by the 2.76 test-retest floor — NOT progression prediction (progression in GRAPE is noise-dominated: ship as an honest negative result).
- GO/NO-GO: the learned model on the prior-bearing stratum must beat persistence (3.30 photo / 3.185 any-VF); else the win is the visit-1 stratum only — frame accordingly.

---

## Task 1: Restore the interval field + (optionally) any-VF priors in the dataset
**Files:** Modify `build_longitudinal_grape.py`; regenerate `data/vf_tests/grape_longitudinal.json`; re-run `decoder/diagnostics.py split-long`. Test in `decoder/tests_session3.py`.
- The builder already parses `interval` (Follow-up col 4, cumulative years from baseline) but drops it. Add `"interval_years": r["interval"]` to each record. Also emit, for each eye, the full visit-ordered VF history (including non-photo VFs from the Follow-up sheet — these make better/closer priors: any-VF persistence 3.185 vs photo-only 3.299) as `prior_history: [{visit, interval_years, hvf}]` (causal, visits < this one).
- Test: every record has `interval_years` (float ≥0); visit-1 records have empty `prior_history`; `prior_history` is strictly earlier visits; Δt = interval(t) − interval(prior) ≥ 0.
- Regenerate folds; confirm `tests_session3.py` still passes (visit-1==baseline unaffected).

## Task 2: Retrain a proper VF autoencoder (manifold)
**Files:** Create `decoder/vf_autoencoder.py` (+ retrain script) → `decoder/pretrained_vf_ae.pth`. Do NOT touch `decoder/pretrained_vf_decoder.pth`.
- Architecture: `VFEncoder` `[v_imp(52) ‖ mask(52)] → 256 → 256 → z(64)`; `VFDecoder` `z(64) → 256 → 256 → 52`; LN+GELU+Drop(0.1). Bottleneck = manifold.
- Input fix (non-zeroed): mean-impute masked points (per-point train mean, query order) + a binary valid-mask channel; randomly drop 0–40% of measured points per step to simulate partial priors. Never feed zeros.
- Train on UWHVF standardized (~29k, query order, one AE for both eyes since query order normalizes anatomy). Masked Huber(δ=1) + 0.3·masked-MSE against the clean full field (so it learns to infill) + 1e-4·‖z‖². Reuse `pretraining.py` optimizer/loop. Save `{encoder, decoder, d_latent, point_mean(52), config}`.
- Test: round-trip recon MAE on observed points < ~1.5 dB; infill MAE on dropped points finite; `z` shape (B,64).

## Task 3: Longitudinal dataset (causal pairing)
**Files:** Create `decoder/longitudinal_dataset.py`. Test in `decoder/tests_session3.py`.
- For each (PatientID, Laterality), sort by VisitNumber; for target index j, prior = most-recent visit `< j` (from `prior_history`, any-VF allowed), `Δt = interval(t) − interval(prior)`, `has_prior = j>0`. Build 631 samples (368 with-prior + 263 visit-1).
- Each sample yields: target fundus path, target hvf (query-order 52 + mask), `prior_vec52` (+ mask, nan→mean-impute), `Δt`, `has_prior`, laterality, patient_id. Train-aug option: emit all-earlier (target, prior) pairs for richer interval supervision; score only canonical (most-recent) pairs.
- Test: no sample uses a prior with VisitNumber ≥ target; visit-1 samples have has_prior=0; severity sampler weights present.

## Task 4: LongitudinalVFModel
**Files:** Create `decoder/longitudinal_model.py` (subclass `PerPointVFModel`). Test in `decoder/tests_session3.py`.
- Forward: `latent=_encode(x)`; `attended,attn_w=attention(patches,lat)`; `z_prior = AE.enc([v_imp‖mask]) if has_prior else no_prior_token`; `ie = IntervalEmbed(Δt,has_prior)` (Fourier + Δt + log1p + has_prior → 64); `prior_field = where(mask, prior_vec, AE.dec(z_prior))`; `h = [attended ‖ cls ‖ z_prior ‖ ie]` (B,52,2176); `delta = sigmoid(alpha(ie)) · delta_head(h)` with **delta_head last layer zero-init**; `pred = refine(prior_field + delta)`; clamp 0..35; TTA average.
- AE frozen. `delta_head` = existing `PointHead` with `input_dim=2176`.
- Test: at init (zero Δ-head) `pred ≈ prior_field` for has_prior=1 (starts at persistence); visit-1 path runs; output (B,52); attention path active (GH intact).

## Task 5: Visit-1 disc structure-function severity branch
**Files:** Extend `decoder/longitudinal_model.py` (reuse `disc_crop_pil` in `training.py`).
- Second frozen-RETFound pass on the laterality-aware disc crop → disc CLS/tokens → small severity head → FiLM (per-point scale/bias) conditioning the Δ-head, so disc features set the eye-level OFFSET (probe: disc r=0.79 vs CLS 0.67). Heavy dropout (263-eye overfit risk). Active for all records; matters most for visit-1.
- Test: disc branch produces a finite per-eye offset; ablating it changes only the offset, not the spatial shape at init.

## Task 6: Training loop + losses
**Files:** Create `decoder/train_longitudinal.py` (reuse `compute_loss`, `evaluate*`, `WeightEMA`, `_update_champion`).
- Loss = `compute_loss(pred, target, lat, ...)` (GH-weighted Huber + CCC + variance + bias + dispersion, unchanged) + `λ_m·manifold_consistency` (round-trip pred through frozen AE, λ_m≈0.05) + `λ_d·mean(delta²)` (λ_d≈0.01). Prior-VF train aug: +N(0,0.5dB) on the prior. Severity-aware gate: push `α→0` at low predicted dB so severe points aren't shrunk.
- Keep encoder+AE frozen → latent-cache fast val path.

## Task 7: Leak-free CV eval + ablations
**Files:** Create `decoder/run_cv_long.py` + longitudinal path in `decoder/eval_ckpt.py`. 
- 5-fold per-patient CV on `cv_long/`; OOF pooled + `stratified_report` + per-eye-avg + visit-1/prior-bearing split.
- Ablation table: (a) persistence, (b) fundus-only [our long_global = 4.29], (c) prior-only, (d) fundus+prior full, (e) per-stratum. Confirm GO/NO-GO (model beats persistence on prior-bearing stratum).
- Scatterplot deliverable (`scatter_clean` extended): raw + variance-match calibrated, colored by stratum + severity band; report slope/floor.

## Task 8: Run + record
- Sanity fold-0 dev run → confirm starts at ~persistence and improves. Full 5-fold CV. Log to `decoder/results/auto/`. Update `iterations.md` with the honest ablation table + framing. Update champions if beaten.

## Self-Review
- Spec coverage: prior-VF (T2,T3,T4) ✓; interval (T1,T4) ✓; visit-1 disc (T5) ✓; leak-free eval + ablations + persistence baseline (T7) ✓; honest framing guardrails (Global) ✓; negatives (progression, OCT, fellow-eye) excluded ✓.
- Expected honest result: pooled ~3.45–3.55 (sub-3.74), slope ~0.75, severe floor ~5–6 on with-prior; visit-1 ~4.0. Beats persistence 3.71 and fundus-only 4.29 on the identical 631-record per-patient CV.
