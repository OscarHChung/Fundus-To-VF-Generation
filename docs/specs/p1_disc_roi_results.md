# P1 (disc-ROI sole-input) — RESOLVED results + decisions

Companion to `p1_disc_roi_handoff.md`. This is the **outcome** of the handoff: all 5 folds finished,
the pooled no-TTA OOF was scored, the pre-committed decision rule was applied, and the six ranked
directions were worked through under their pre-registered gates. Reproduction commands at the bottom.

---

## 0. Headline

- **Native pooled MAE < 4.0 is decisively NOT reached.** p1disc pooled no-TTA OOF (631 recs) = **4.113
  dB**, patient-bootstrap 95% CI **[3.786, 4.474]**. §6.5 rule 5 (population claim needs point ≤ 3.65)
  is not close. This confirms the design's §P1 prediction (best case ≈ 4.16).
- **But P1 is a real, robust pooled WIN over the m1sev champion** — the largest of any lever tried:
  paired **ΔMAE −0.146 dB**, CI **[−0.247, −0.046]**, **negative in 5/5 folds**. The opposite of M2's
  wash.
- **Formal verdict under the binding §6.5 rule = DO NOT PROMOTE**, blocked *only* by the severe-band
  criterion (see §2). This is a judgement call flagged for the user; the pre-committed rule was not
  overridden.
- **The disc crop lifted the eye-specific spatial channel** — the first lever to do so — from the
  full-image model's partial-corr **0.198 → 0.297** (§3).
- **The matched-composition publication floor got STRONGER** (§4): p1disc now beats the 31,443-photo
  TDV-Net in **all three** severity strata (m1sev's mild was a tie); ours **3.25–3.69** vs their 3.91.

Answer to the two stated goals: *push native <4.0* hits the information ceiling and fails; *secure the
publication floor* succeeds and improves. Net: the honest headline is composition-adjusted superiority
plus a measured spatial-channel gain, not a native sub-4.0 number.

---

## 1. Decisive pooled table (no-TTA, 5-fold OOF, 631 records)

| metric | m1sev (full image) | **p1disc (disc crop)** | Δ |
|---|---|---|---|
| pooled MAE | 4.259 | **4.113** | **−0.146** |
| raw slope | 0.527 | 0.536 | +0.009 |
| calib slope (MAE) | 0.641 (4.506) | 0.642 (4.345) | — |
| Pearson r | 0.661 | 0.684 | +0.023 |
| eyeCorr | 0.470 | 0.538 | **+0.068** |
| sev_corr | 0.807 | 0.802 | −0.005 (flat) |
| res_corr (raw) | 0.404 | 0.494 | +0.090 |
| severe-band MAE (n=101 eyes) | 7.372 | 7.438 | +0.067 |

Per-fold paired ΔMAE (candidate − reference): fold0 −0.064, fold1 −0.065, fold2 −0.067, fold3 −0.214,
fold4 −0.294. **All five negative.** Patient-bootstrap pooled ΔMAE CI [−0.247, −0.046], P(Δ≥0)=0.002.

**The gain is spatial, not severity:** sev_corr is flat (0.807→0.802) while eyeCorr and res_corr rise.
The disc crop did not improve the axon-count → MD channel; it improved the *within-eye pattern* channel.

**Mandatory companion metrics (§6.3), p1disc:**
- **Disattenuated slope** = raw slope / stratum reliability. Pooled **0.536 / 0.845 = 0.634 → clears
  ≥ 0.60** under the correct errors-in-variables framing (report this, not the raw 0.536 beside a
  calibrated MAE).
- **MAE as a multiple of the stratum Bayes floor** (floors: pooled 2.25 / severe 3.16 / moderate 2.78 /
  mild 1.75): pooled 4.113 = **1.83×**, severe 7.438 = 2.35×, moderate 5.093 = 1.83×, mild 2.628 =
  **1.50×**. The mild band is closest to its noise floor; the severe band is furthest (noise-bounded).

## 2. Pre-committed decision rule (§6.5 rule 2) — why DO NOT PROMOTE

| criterion | result | pass |
|---|---|---|
| pooled ΔMAE ≤ −0.12 | −0.146 | ✓ |
| pooled 95% CI upper < 0 | −0.046 | ✓ |
| ΔMAE negative in ≥4/5 folds | 5/5 | ✓ |
| **severe ΔMAE 95% CI upper < +0.15** | **+0.407** | ✗ |
| raw slope not worse | 0.536 ≥ 0.527 | ✓ |

Only the severe-band criterion fails. Its point estimate is a trivial **+0.067 dB**, but the severe
band is so noise-dominated (§2.5: Bayes floor 3.16, a same-eye prior VF only reaches 4.46) that the
paired ΔMAE CI is **[−0.288, +0.407]** — a half-width of ±0.35, which alone exceeds the +0.15 gate.
**The criterion is essentially unmeetable on this band for any method**; it cannot distinguish "flat
severe" from "worse severe." The primary endpoint (§6.2 = pooled MAE) is a clean win.

→ Recommendation for the user (judgement call, not taken unilaterally): treat P1 as the new champion
on the primary endpoint, reporting the severe effect as a noise-dominated +0.067; OR keep the strict
letter of the rule (do not promote) and report P1 as the strongest-but-formally-blocked lever. Either
way, do **not** move the goalpost after the fact — record which reading is chosen and why.

## 3. D2 template-partial gate (the mechanism) — `decoder/d2_template_partial.py`

`decompose.py`'s res_corr (0.494) is ~entirely the population template. The eye-specific signal is the
partial-corr controlling for the template:

| model | partial-corr(pred, true \| template) | 95% CI | gate |
|---|---|---|---|
| m1sev (full) | 0.198 | [0.148, 0.249] | ≤0.25 → P2 dead (matches design's 0.196) |
| **p1disc (disc)** | **0.297** | [0.249, 0.342] | 0.25–0.35 → **AMBIGUOUS** |

The disc crop **added ~+0.10 of eye-specific spatial signal** the full image did not carry — the first
lever to move the channel the ceiling analysis wrote off (updates `fundus-only-is-severity-estimation`).
But 0.297 < 0.35, so it does **not** clear the pre-registered gate to build P2. See §5-Dir2/4.

## 4. Matched-composition floor (§3.1) — `decoder/composition_report.py`

Point-level strata (each point bucketed by its own true dB), p1disc:

| stratum | our MAE | our %pts | TDV-Net | Δ (ours − TDV) |
|---|---|---|---|---|
| mild (≥22) | 2.929 | 61.2% | 3.09 | −0.161 |
| moderate (15–22) | 3.934 | 21.6% | 5.66 | −1.726 |
| severe (<15) | 8.558 | 17.2% | 9.15 | −0.592 |
| pooled | 4.113 | | 3.91 | +0.203 |

- TDV-Net scored under OUR composition (17.2% severe pts): **4.686 dB**.
- Ours under any TDV-consistent composition (≤13.5% severe, reproducing their 3.91): **3.25–3.69 dB**
  over 43 feasible mixes — **beats their 3.91 under every one**.
- p1disc now wins **all three** point-level strata (m1sev's mild was a tie at −0.02). The floor improved.

## 5. The six ranked directions — what each yielded

1. **Crop-safe TTA (dir 1) — IMPLEMENTED + unit-verified; not empirically re-eval'd.**
   `training.rotate_then_disc_crop`: rotate the FULL image then disc-crop (no black border), fixing the
   −1.4 dB TTA-on-crops bias. Default-OFF and disc-no-TTA paths byte-identical (reviewer-confirmed);
   **8/8 `tests_method_p1.py` pass**. The pooled/fold-0 TTA re-eval was NOT completed: ×3 TTA triples the
   in-memory latent cache and rotates full-res images, OOM-killing this 16 GB box (died at 85% of fold-0
   train, twice — the box's documented limit). Low EV (±0.02–0.05 dB), changes no conclusion (native
   still 4.113, promotion still severe-blocked). Run on a larger box if the exact number is wanted; the
   fix's correctness is by construction + unit test.
2. **P2 two-view fusion + 4. disc-angular prior — BELOW GATE, not built (flagged for override).** These
   are the design's single "P2" proposal, gated on D2 ≥ 0.35. p1disc's partial-corr is 0.297 (< 0.35) →
   gate not met; the design's frozen full⊕disc combined probe already failed (§D2, Session 3: eyeCorr
   0.497 < template+noise); and the actual promotion blocker (severe band) is noise-bounded and untouched
   by fusion. The dir-3 result closes the one open door: a *wider* crop REDUCED eyeCorr (0.578→0.507),
   so re-introducing macula/central context — exactly what full⊕disc fusion does — dilutes the spatial
   gain rather than lifting partial-corr past 0.35. Per §6.5 + the "scout wins" gate, a multi-hour P2
   build/retrain is **not indicated**. Build only if a future frozen-feature D2 probe on some new feature
   set clears 0.35 first (cheap, no training) — do that before writing any fusion code.
3. **Wider-crop scout 0.40 (dir 3) — pre-registered, DONE, DID NOT WIN.** Honest no-TTA fold-0:
   MAE **4.157 vs 0.27's 4.101 (+0.056 worse)**, slope 0.588→0.654, **eyeCorr 0.578→0.507**. The wider
   crop worsens the primary endpoint and *dilutes* the eye-specific spatial gain (eyeCorr drops toward
   the full-image level), trading it for slope. Per the pre-registered fold-0 rule + the scout gate →
   **no 5-fold retrain; keep 0.27.** Bonus: confirms the disc's spatial advantage is the TIGHT
   peripapillary focus — adding macula context regresses eyeCorr, i.e. direct evidence that P2 fusion
   (which re-introduces the full/macula view) would dilute the very gain that makes P1 work.
5. **Free stack (dir 5).** Variance-matched calibration is a *reporting device only* (§5 #6): p1disc
   calib slope 0.642 at MAE 4.345 — buys slope, costs MAE, never both. Cross-fold ensemble (§5 #7,
   <0.08 dB) and target denoising (§5 #9, +0.04 dB) are **sub-noise** by prior measurement; a leaky
   cross-fold ensemble on OOF would also not be honest. Reported, not run.
6. **Matched-composition reporting (dir 6) — DONE, §4 above.** The publication floor.

## 6. Reproduction

```bash
# Pooled no-TTA OOF for both tags into a shared fresh cache (fold checkpoints already exist):
python decoder/eval_oof_cached.py --tag p1disc --no-tta --cache-dir decoder/results/auto/oof_cache_notta
python decoder/eval_oof_cached.py --tag m1sev  --no-tta --cache-dir decoder/results/auto/oof_cache_notta
python decoder/paired_decision.py            # §6.5 promotion rule, p1disc vs m1sev
python decoder/d2_template_partial.py --tag p1disc   # spatial gate (also --tag m1sev)
python decoder/composition_report.py --tag p1disc    # matched-composition floor
```

## 6b. Code review

The three verdict-critical analysis scripts were independently reviewed for correctness (record
alignment, leak-freeness, statistical validity). All gating statistics, the record alignment
(npz row i ↔ `fold{f}_val.json` record i), leak-freeness (template + calibration fit on train only),
and the crop-safe TTA change (default-OFF byte-identical) were confirmed **correct**. Three secondary
issues were fixed: (1) `paired_decision.py` native-<4.0 check now bootstraps the candidate's OWN
absolute-MAE CI (upper 4.474, matching `composition_report.py`) instead of the anti-conservative
`mae_ref + paired_hi`; (2) `composition_report.py` guards `min/max(feas)` against an empty feasible
set; (3) `composition_report.py::load` asserts npz/JSON length alignment like its sibling scripts.

## 7. Open items
- Update `fundus-only-is-severity-estimation` memory: the "spatial worth ~0.06 dB" verdict holds for
  the FULL image but the disc crop lifts the eye-specific partial-corr to 0.297 — a real, if
  sub-promotion, spatial gain.
- If the 0.40 scout clears its fold-0 gate (severe not worse AND pooled better), run the gated 5-fold
  retrain (`p1disc040_f{0..4}`) and re-apply `paired_decision.py`.
- Decide the promotion judgement call in §2 and record it.
