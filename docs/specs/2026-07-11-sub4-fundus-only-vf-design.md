# Sub-4.0 Fundus-Only VF — the input-side push (design)

**Date:** 2026-07-11 · **Status:** design, approved-shape (3 scoping decisions locked) · **Next:** task-level
implementation plan (writing-plans) on go-ahead. Research/planning only — no code changed yet.

## Goal

Push the **fundus-only-inference** per-point 24-2 VF model to a **native pooled MAE < 4.0** (point estimate)
with **line-of-best-fit slope ≥ 0.6**, on the existing leak-free per-patient 5-fold CV over 631 GRAPE records,
and package it to **decisively beat the 2026 fundus→VF SOTA** for Ophthalmology Science.

- **Primary success (locked decision 3):** native fundus-only point-estimate MAE < 4.0 + slope ≥ 0.6, AND lead
  the paper with the composition-adjusted head-to-head (the reliable, already-in-hand win). Realistic stacked
  landing ≈ **3.90–4.00** (carried by the spatial/regularization stack from the already-banked 4.060 disc-jitter
  base; ~3.85 only if the uncertain severity levers also land). See the re-weighted risk section for probabilities.
- **Acceptance-critical deliverable (added 2026-07-11): EXTERNAL VALIDATION on PAPILA.** The single biggest
  rejection risk at Ophthalmology Science is "single small dataset, no external validation" — a direct consequence
  of the free-data-only constraint. PAPILA (Spanish cohort, different camera, fundus + 30-2 **MD**) is a fully
  independent cohort that can externally validate the **fundus→severity (MD)** component of the model (it can't
  validate the 52-point map — no per-point VF — but severity is the pooled-MAE-dominant channel). Reporting
  fundus→MD correlation/MAE on PAPILA, never seen in training, converts the free-data constraint from a pure
  liability into "the generalizable component is externally validated." This is likely worth more to acceptance
  than any remaining 0.1 dB of MAE.
- **Stretch (not the bar):** bulletproof CI-decisive ≤ 3.65 (bootstrap CI upper < 4.0). Likely out of reach
  fundus-only on 144 patients — do not gate the paper on it.

## Hard constraints (verbatim, immovable)

Frozen retinal foundation encoder → custom VF decoder (encoder–decoder); **2-stage pretraining**;
**Garway-Heath** anatomical sectoring in the decoder (always on); **fundus-only at inference** (no prior VF at
test time); **longitudinal data may be used during TRAINING only**; **only free-download open data — no
requests/DUAs/dbGaP, and (locked decision 1) no gated-login downloads either** (ungated-only). Never re-split /
reseed the folds; score OOF pooled over 631 vs RAW VF; per-fold quantities fit on that fold's train only.
Memory: 16 GB MPS box, **exactly one torch process at a time**; **no >100 MB checkpoints in git**; any new flag
default-OFF must be byte-identical to the current model.

## Locked scoping decisions (this session)

1. **Encoder = ungated-only.** RETFound-DINOv2 / DINOv3 (gated HF EULA) are excluded. The severity-encoder
   lever is a **fully-free frozen multi-encoder ensemble**: RETFound-MAE ⊕ RetiZero ⊕ RETFound-Green ⊕ VisionFM.
2. **External free severity data = in — PAPILA has a DUAL role (co-training + external validation).** The only
   free set with a *functional* severity label paired with fundus is **PAPILA** (~244 eyes, disc-centered, **30-2
   MD** — not all eyes have a glaucoma-range MD; CC-BY figshare). Its two uses: **(i) severity co-training** (P-A2,
   enrich the readout — uncertain, OOD) and **(ii) external validation** of the fundus→MD component (acceptance-
   critical, Phase E). These MUST be kept disjoint: if PAPILA eyes are used to train the severity head, they cannot
   also be the external-validation set — so **hold out a fixed PAPILA validation split that is never used in any
   co-training/probe**, OR run external validation with a model trained on GRAPE only (cleanest — a true external
   test). **AIROGS is BINARY (Referable/No-Referable Glaucoma), not graded** (verified 2026-07-11) → weak auxiliary
   *classification* pretrain only. Inference stays fundus-only; encoder frozen (co-training only enriches the small
   severity readout head).
3. **Success = native <4.0 point + strong framing** (see Goal).

---

## The strategic reframe (why this plan differs from all prior sessions)

Five parallel research agents (2026-07-11) converged on a single reframe: **prior sessions were optimizing the
wrong side of the model.**

- **Output-side RESCALING is dead for pooled MAE (proven).** Projecting the champion's real OOF predictions onto
  the VF archetype basis moves pooled MAE ≤ 0.005 dB and **sev_corr exactly 0.000** across K=3..20. The champion is
  already on-manifold and already de-shrunk (sev_shrink 1.02, σp/σt 0.844, res_corr 0.474). So any *post-hoc /
  rescaling* family — manifold projection, dispersion/variance loss, calibration, de-shrink — is MAE-neutral
  (explains those prior dead-ends). **Scoped caveat (an over-claim in v1, now corrected):** this proves the
  *rescaling* family is dead, NOT that *every* loss change is. A loss-FUNCTION reformulation (ordinal/CORAL, P-C1)
  changes which errors the optimizer weights and is robust to the floored/censored dB scale — it *cannot*
  manufacture severity correlation from fixed features (that's still feature-limited), but it is an unproven,
  low-prior *pooled-MAE* lever, kept as a cheap probe rather than dismissed as scatterplot-only.
- **The binding constraint is fundus→severity correlation, an input/feature/data property.** Reliability
  decomposition (Agent 2, validated against the 2.76 dB reference): severity reliability(K=1) = 0.936, spatial
  reliability(K=1) = 0.693. Disattenuated **true** correlations: **severity r_true ≈ 0.834, spatial r_true ≈
  0.493** — both below the sub-4.0 frontier (0.875 / 0.60). Severity is **eye-count-limited, not
  noise-limited** (why D1's learning curve keeps rising with n but label-denoising barely helps severity).
- **Consequence:** the winning levers must attack the **input** (better/more features) or the **training
  signal** (cleaner labels / more distinct severity eyes) — never the decoder head. Stack levers on *different*
  error terms (severity vs spatial); they are roughly additive on pooled MAE.

### Evidence base (this session's 5-agent research)

| Finding | Source | Consequence for the plan |
|---|---|---|
| Output-side archetype projection: pooled Δ≤0.005, sev_corr Δ=0.000 | Agent 5 offline probe on champion OOF | Decoder/manifold levers → scatterplot only, **not** pooled MAE |
| true corr severity 0.834 / spatial 0.493, both < frontier | Agent 2 disattenuation | Fundus-only decisive ≤3.65 is very hard; native <4.0 point is the realistic bar |
| Every photo-eye has ≥3 VFs (mean 4.24) → all 631 records get a denoised target | Agent 2 visit-count audit | Trajectory denoising covers 100% incl. the visit-1 bottleneck; **distillation ⊂ denoising** |
| Trajectory target-denoising: spatial +0.031 (tuned)…+0.078 (under-reg), pooled −0.05..−0.12 | Agent 2 empirical OOF probe | Real, modest, spatial-channel lever |
| RETFound-MAE uses FIXED sin-cos pos-embeds + analytic regenerator | Agent 4 code read | Higher-res (384/448) has **no OOD pos-embed artifact** — the flagship input lever |
| Generic DINOv2 worse on both channels; retinal FMs beat generic; FusionFM: frozen retinal-FM ensemble +4% AUC glaucoma | Agent 3 web + prior bake-off | Ensemble of ungated retinal FMs is the severity-encoder play |
| Severity eye-count-limited; free per-point VF data = none beyond GRAPE; only free *functional*-severity set = PAPILA (~244 OOD MD eyes); AIROGS is binary | Agents 1/2/3 | **PAPILA MD transfer** is the one free severity lever — real but uncertain (domain shift), probe-gated; no strong severity lever exists under the constraints |
| Beat TDV-Net decisively in moderate+severe (mild TIE) + composition-adjusted 3.25–3.69 vs 3.91 [corrected — the 3.37–3.62 figure this row originally cited was the prior champion m1sev's, not p1disc's; see paper/headline_results.md §1.3] ; split unconfirmed; MLEDL is Octopus not 24-2 | Agent 1 SOTA | Composition-adjusted win is reliable/in-hand; native push makes it unarguable |

Base numbers to beat (no-TTA pooled OOF): **p1disc 4.113** (champion), m1sev 4.259; frozen linear-probe
sev_corr@505 = 0.724 (in-run) / trained sev_corr 0.807; spatial partial-corr: m1sev 0.198, p1disc 0.297,
P2-build gate 0.35; sub-4.0 frontier sev_corr ≈ 0.875.

---

## Lever ladder — cheap probe → pre-committed gate → train only survivors

**Discipline:** every lever has a probe that needs NO training (numpy/ridge or one frozen cache pass), reusing
existing machinery (`diag_encoder_bakeoff.py`, `diag_d1_learning_curve.py`, `d2_template_partial.py`,
`paired_decision.py`). Run ALL probes first; keep only gate-passers; then stack survivors into ONE model and run
the §6.5 5-fold CV once. One torch process at a time throughout.

### Phase 0 — probes (no training; the decision gates)

**P-A2 · PAPILA MD severity transfer (SEVERITY — the one free eye-count lever, but uncertain).**
- Rationale + risk: severity is eye-count-limited and D1's curve is still rising, so *more distinct eyes* should
  help the readout — **iff** the external eyes are on-distribution. PAPILA is a different camera/population
  (Spanish) with **30-2 MD** vs GRAPE's 24-2, so adding its ~244 eyes could equally *hurt* a ridge (distribution
  shift). The probe is exactly the arbiter; prior is genuinely ~50/50.
- Method: cache frozen RETFound-MAE features for PAPILA (disc-centered fundus → 30-2 MD; z-score MD per dataset to
  neutralize the 24-2/30-2 scale gap; consider glaucoma+suspect only). Fit a ridge severity readout on GRAPE-train
  **+ PAPILA eyes**; evaluate OOF **against GRAPE single-visit MD** (leak-free, fold-disjoint). Compare sev_corr
  vs GRAPE-train-only. Optionally add AIROGS as a *binary* auxiliary head (weak).
- Gate: OOF sev_corr@631 rises by **≥ +0.02** with external eyes added. If it drops, try (a) disc-crop-only
  features, (b) per-dataset feature centering / CORICA-style domain alignment; else record "external severity data
  does not transfer" and drop.
- Files: extend `diag_encoder_bakeoff.py` cache path for external images; new `diag_papila_severity.py`.
- Expected if pass: **0 to −0.10 pooled** (tempered from v1's −0.05..−0.15: only ~244 OOD eyes, not the mistaken
  "100k graded").

**P-A1/A3 · Ungated frozen-encoder ensemble (SEVERITY channel only).**
- Method: add ungated loaders (RetiZero, RETFound-Green, VisionFM) to `encoders.py`; cache each once; run the
  D1 severity learning curve + spatial partial-corr per encoder AND for hstacked concatenations
  (MAE⊕RetiZero, MAE⊕RetiZero⊕Green).
- **Scope caveat:** the members have *different grids/dims* (RetiZero ViT-L/14→16×16; Green ViT-S/14@392→28×28,
  384-d; VisionFM ViT-B→768-d). Concatenating **pooled** features (CLS⊕mean) for the SEVERITY head is clean and
  grid-agnostic. Feeding them into the *spatial* decoder (14×14 GH prior) is a multi-grid integration mess → the
  realistic use is a **severity-head ensemble**, not the spatial channel. Scope this lever to severity.
- **EV honesty:** all published wins for these are classification AUC, not severity regression, and the prior
  bake-off already showed generic DINOv2 *worse* — so the retinal-ensemble gain is speculative. Cheap to probe.
- Gate: best ensemble sev_corr@505 exceeds RETFound-MAE by **≥ +0.03**. Verify each is a genuine ungated download
  (and a research-OK license) before wiring.
- Files: `encoders.py`, `diag_encoder_bakeoff.py` (add `probe()` hstack of cached npz).
- Expected if pass: −0.03 to −0.10 pooled; complementary SSL objectives, zero new labels.

**P-B1 · High-res frozen RETFound @384/448 (SPATIAL+severity — flagship input lever).**
- Method: extend `diag_encoder_bakeoff.py::cache_encoder` with `input_size`; bypass timm's `PatchEmbed` size
  assert via `patch_embed.proj` directly; regenerate pos-embed with the repo's `get_2d_sincos_pos_embed(1024,
  H//16, cls_token=True)`. Cache `{full@224 (baseline), full@384, full@448, disc@224 (=P1), disc@384, disc@448}`.
  The bake-off descriptors are already grid-agnostic → `--probe` runs unchanged.
- **OOD honesty:** the sin-cos regenerator removes the *pos-embed* artifact, but two residual OOD axes remain —
  patch **magnification** (16-px patches now cover finer physical detail than pretraining saw) and **sequence
  length** (576/784 tokens vs the 196 the ViT was pretrained on). These can degrade features; the probe reveals
  it. Treat >512 as likely OOD; 384/448 is the sweet spot.
- **Cheap refinement to co-probe (addresses the disc-jitter wash):** a **detected/registered disc center** (green-
  channel brightness/vessel-convergence centroid within the laterality quadrant — dependency-free) vs the current
  FIXED box (OD cx0.78/OS cx0.22). Agent 4's read of the disc-jitter dilution (fold-0 −0.109 → −0.053 pooled) is
  that the fixed box is slightly mis-centered per-eye; a detected center captures that benefit *deterministically*
  without jitter's variance. Cache disc@{fixed, detected} × {224, 384} and compare in the same probe.
- Gate: any config beats the 0.724 sev / 0.176 spatial anchors by the +0.03/+0.05 advance rule; flag whether the
  **high-res disc** pushes trained-scale spatial toward/past the 0.35 P2 gate.
- Expected if pass: spatial +0.03–0.08, pooled a few hundredths; the most likely single config to clear the P2 gate.

**P-B3 · Multi-crop feature-fusion probe (SPATIAL, conditioned on P-B1).**
- Method: hstack the `spat` vectors of the best {disc, full} caches (ideally B1's high-res disc) and run the
  leak-free `d2_template_partial` spatial partial-corr with patient-bootstrap CI.
- Gate (pre-registered): concat spatial partial-corr ≥ 0.35 → build feature-fusion "P2" head; ≤ 0.25 → dead;
  0.25–0.35 → hold. Run *after* P-B1 (fusion on 224 features nearly replays a prior null).

**P-C1 · Ordinal per-point head probe (cheap, uncertain).**
- Method: offline, ridge vs CORAL/SORD ordinal probe of frozen CLS→eye-mean and per-point→dB-bins on the 5 OOF
  folds; compare sev_corr / Spearman / pointwise-R. VF-HM (MICCAI 2023) reports +16.6% MAE from ordinal on the
  same task; Agent 5 is skeptical it survives on already-de-shrunk features — the probe decides.
- Gate: ordinal beats regression by ≥ +0.02 corr → carry into the stacked model; else drop.

### Phase 1 — build + stack survivors, then the honest CV (train)

For each gate-passer, add a **default-OFF byte-identical flag + unit test** (mirror `--disc-only`/`--disc-jitter`),
fold-0 scout vs `p1disc_f0` (4.101) — single fold can KILL not promote — then stack all survivors and train the
full 5-fold CV. Trajectory denoising (P-B2 below) is a training-target change applied throughout.

**P-B2 · Trajectory target-denoising (SPATIAL; all 631 records; already probed positive).**
- Method: precompute `hvf_denoised` per record in `build_longitudinal_grape.py` — per-eye per-point robust
  (Theil-Sen/Huber) trajectory fit over the eye's ≥3 VFs, evaluated at the image date. In
  `longitudinal_dataset.py:__getitem__`, load `hvf_denoised` as the target for the **train split only** (val/eval
  keep raw `hvf`). Loss untouched. Applies to the visit-1 fundus branch too.
- No probe gate (Agent 2 already probed −0.05..−0.12 pooled); confirm on fold-0 no-TTA vs p1disc_f0.
- **Sub-MDE caveat:** −0.05..−0.12 straddles the 0.12 method-MDE, so B2 *alone* likely fails §6.5 (like
  disc-jitter's −0.053 wash). Its value is as a **stack component**, not a standalone promotion.

**Final CV + promotion:** `eval_oof_cached.py --tag <stack> --no-tta` → `paired_decision.py --new <stack> --ref
p1disc`. Promote on §6.5. Re-run `composition_report.py`, `d2_template_partial.py`, `make_scatterplot.py`.

### Phase 2 — paper deliverables (framing; partly independent of Phase 1)

- **Head-to-head, stated honestly (the reliable win):** the robust claim is the **composition-adjusted** one —
  ours **3.25–3.69 dB under any TDV-consistent severity mix** vs their 3.91, and TDV-Net **4.686 under our
  17.2%-severe mix** (from OUR pooled OOF, no training; `composition_report.py` / `fundus_only_ceiling_design.md`
  §3.1). [Corrected — the 3.37–3.62 / 4.69–4.81 figures this bullet originally cited were the prior champion
  m1sev's, not p1disc's; re-run fresh on p1disc these are 3.250–3.689 vs 3.91 and TDV-under-our-mix a single
  value 4.686 (not a range). See `paper/headline_results.md` §1.3 for the current numbers, including the
  reported model `p1disc_denoise`'s own (3.274–3.649, TDV-under-our-mix unchanged at 4.686).] Stratum-wise:
  **decisive in moderate + severe under both eye-level and point-level conventions; mild is a TIE — do not
  headline "every band."** The eye-level convention (2.76/5.44/7.25) and point-level (3.07/4.29/8.44) numbers
  originally listed here were also m1sev's, not p1disc's. p1disc's own point-level strata (mild/moderate/severe)
  are **2.929/3.934/8.558**; its eye-level strata are 2.628/5.093/7.438. (Do NOT reuse the single-fold
  p1disc_f0 numbers — a v1 error.)
- **Total-Deviation reporting (C2) — NOT just cosmetic:** TDV-Net predicts TD and stratifies by eye stage, so
  computing OUR numbers in TD space (subtract the 24-2 age-normal; GRAPE Age from the Baseline sheet) is what
  makes the stratified head-to-head and the scatter *airtight* (pooled TD-MAE ≡ sensitivity-MAE; r/slope
  invariant). Elevate from "reporting only" to a required comparability deliverable.
- **Honest ceiling + protocol contrast:** disattenuated slope (0.834 true severity, 0.493 spatial — both
  sub-frontier) alongside raw; per-patient/causal 5-fold vs TDV-Net's unconfirmed split; MLEDL is Octopus 59-pt
  (its fundus-only ROI 3.903 is not dB-comparable). Archetype/latent-basis head (C3) optional for a cleaner
  scatter figure.

### Phase E — External validation on PAPILA (acceptance-critical; answers the #1 reviewer objection)

**Objective:** externally validate the model's generalizable **fundus→severity (MD)** component on a fully
independent cohort (PAPILA — Spanish, different camera), converting the free-data constraint from a liability into
a strength. Not a per-point-VF validation (PAPILA has no 52-point field), but severity is the pooled-MAE-dominant
channel, so this is the meaningful external test.

- **Data hygiene (non-negotiable):** the external-validation PAPILA eyes must be **disjoint from any PAPILA eyes
  used in P-A2 co-training**. Cleanest design: evaluate a **GRAPE-only-trained** model on **all** of PAPILA (a true
  external test, no PAPILA in training) — report this as the headline external number. If P-A2 co-training is
  promoted, additionally hold out a fixed PAPILA test split (patient-disjoint) never seen in co-training and report
  on it too. Never let PAPILA touch the GRAPE cv_long folds.
- **Method:** run the final GRAPE-trained fundus model's eye-mean (MD proxy) on PAPILA fundus images; align scales
  (PAPILA 30-2 MD vs our 24-2 eye-mean — report Pearson/Spearman r, which are scale-free, plus MAE after a
  train-fit linear calibration). Stratify by PAPILA's healthy/suspect/glaucoma labels. Patient-bootstrap CI.
- **Deliverable:** "fundus→MD generalizes to an independent cohort: r = X (95% CI), MAE = Y dB" — a first-class
  results paragraph + a PAPILA scatter. New script `decoder/eval_external_papila.py`.
- **Gate/interpretation:** this is a *reporting* deliverable, not a promotion gate — report whatever it shows
  honestly (even a modest external r strengthens the paper by demonstrating generalization; a collapse would be an
  important, publishable limitation about cross-camera fundus→VF transfer).
- **Sequencing note:** PAPILA download + harmonization (fundus + `Age`/`MD` from its clinical CSVs → our schema) is
  a Phase-0 data task (no torch); the eval itself runs once the final GRAPE model exists (after Phase 1).

---

## Expected outcome & honest risk (re-weighted after adversarial review)

**Where the reliable gains actually are (corrected from v1's severity-optimism):** the SEVERITY channel has NO
strong lever under the locked constraints — the encoder options are ungated retinal FMs with only classification
evidence (speculative), and the only free functional-severity data is ~244 OOD PAPILA eyes (~50/50). Disattenuation
says the fundus's *information* ceiling on severity may sit near true-corr ~0.83 (the frontier needs ~0.905 true) —
so betting the target on severity is fragile. **The reliable gains are SPATIAL + regularization**, and we are
*already* close: the disc-jitter point estimate is **4.060** (0.06 above 4.0, blocked from promotion only by the
severe-band CI rule, not by its point estimate). Stacking on that: high-res disc (−0.03..−0.08) + trajectory
denoising (−0.05..−0.12) plausibly reaches a **native point ≈ 3.90–4.00** without needing the uncertain severity
levers; the severity levers, if they land, push toward ~3.85 and toward Tier-2.

**Resolving the "spatial is only worth 0.056 dB" objection (important):** the `fundus-only-is-severity-estimation`
memory measured the within-eye spatial residual at **0.056 dB pooled — but that was on the FULL-image model**, where
the disc is destroyed at 224 px. The disc crop already **broke** that (−0.146 pooled, 5/5 folds; spatial partial-corr
0.196→0.297) — so the spatial channel is NOT capped at 0.056. BUT the disc crop already banked the *big* jump, so
high-res disc + denoising are **incremental over p1disc**, subject to diminishing returns — hence the tempered
per-lever numbers (high-res disc −0.03..−0.08; denoising −0.05..−0.12), not a fresh −0.146.

Budget caveat: pooled MAE is not Pythagorean, so the per-channel budget is a heuristic, and some
disc-jitter/early-stop gains are partly **selection/regularization artifacts** that may not fully stack or
generalize (the disc-jitter −0.109 fold-0 → −0.053 pooled dilution is the cautionary precedent).

- **Tier-1 (native point <4.0 + slope ≥0.6): reasonably likely (~55–70%)**, carried mainly by the spatial/
  regularization stack from the 4.060 base — NOT "high probability," and NOT dependent on the shaky severity levers.
  **Slope, stated precisely:** the RAW slope is only 0.54–0.57; **≥0.6 is met by the calibrated line-of-best-fit
  (p1disc 0.642) and the disattenuated slope (0.643)** — report all three and lead with the calibrated value (a
  train-fit post-hoc calibration, standard in VF-prediction). Do not imply the raw slope clears 0.6.
- **Tier-2 (≤3.65 CI-decisive): unlikely (~15–25%)**, needs the severity levers to beat an uncertain information
  ceiling. Not the paper's gate.
- **The genuinely reliable deliverable is the composition-adjusted superiority** (~0.3–0.55 dB better than the
  31k-photo SOTA at matched mix) + the honest ceiling + the clean protocol — **already in hand, promotion-independent**,
  and it survives the §6.5 severe-band CI wall that blocked p1disc/disc-jitter.
- **Main risks:** (a) PAPILA domain shift makes P-A2 null/negative (~50%); (b) ungated encoders individually weak →
  ensemble marginal; (c) high-res OOD (magnification/seq-length) past 448; (d) severe-band CI wall blocks *formal*
  promotion even if the point estimate improves — mitigate by leading with the composition-adjusted framing.

## Self-review

- **Coverage:** severity channel → P-A1/A3 (ensemble, scoped to severity) + P-A2 (PAPILA); spatial channel → P-B1
  (high-res) + P-B2 (denoising) + P-B3 (fusion); loss-function → P-C1 (ordinal); comparability/scatter → Phase 2.
  Output-side *rescaling* correctly excluded (proven dead); output-side *loss reformulation* kept as a cheap probe.
- **Decisions honored:** ungated-only (no DINOv2); external severity data in (PAPILA; AIROGS only as binary aux);
  native<4.0+framing as the bar.
- **Gates:** every lever has a no-train probe + numeric gate before any GPU; §6.5 governs promotion; native-<4.0
  claim uses the candidate's own patient-bootstrap.
- **Placeholders:** none. **Ambiguity:** "sub-4.0" is point-estimate (Tier-1); CI-decisive is out of scope as a gate.

## Adversarial review log

**Round 1 (2026-07-11) — issues found and fixed in this doc:**
1. **Framing over-claim (major).** v1 claimed we "dominate every band" with numbers 2.46/4.84/7.53 — those were
   **single-fold p1disc_f0**, not pooled OOF, and the repo's own §3.1 says the **mild claim is a TIE** under the
   point-level convention. Fixed: use the pooled dual-convention numbers + the composition-adjusted headline; state
   "moderate+severe decisive, mild tie."
2. **AIROGS mischaracterized (major).** Verified AIROGS is **binary RG/NRG**, not "100k graded." The only free
   *functional*-severity data is PAPILA (~244 OOD 30-2-MD eyes). Fixed: tempered A2 to a small/uncertain,
   probe-gated PAPILA transfer + binary auxiliary; EV cut from −0.05..−0.15 to 0..−0.10.
3. **"Output side is dead for pooled MAE" over-generalized (major).** The proof covers *rescaling* (manifold/
   dispersion/calibration), not loss-function reformulation. Fixed: scoped the claim; ordinal (P-C1) restored as a
   genuine-if-uncertain pooled-MAE probe instead of "scatterplot-only" (removing the internal contradiction with C1).
4. **Probability over-optimism (major).** v1 called Tier-1 "high probability" leaning on shaky severity levers.
   Re-weighted: reliable gains are spatial+regularization from the already-banked 4.060 base → Tier-1 ~55–70%,
   Tier-2 ~15–25%; the composition-adjusted superiority is the reliable, promotion-independent deliverable.
5. **TDV-Net comparability (medium).** Confirmed it predicts **total deviation** (pooled TD-MAE ≡ sensitivity-MAE,
   valid) and stratifies by **eye stage**. Fixed: elevated C2 (TD reporting) to a required comparability task; report
   eye-level strata to match.
6. **Encoder-ensemble grid mismatch (medium).** Members have different grids/dims → scoped A1/A3 to the *severity*
   (pooled-feature) head, not the spatial decoder.
7. **High-res OOD (medium).** Noted the residual magnification + sequence-length OOD (only the pos-embed artifact
   is eliminated). **B2 sub-MDE (medium):** flagged denoising alone likely fails §6.5; value is as a stack component.

**Round 2–3 (2026-07-11) — consistency + two deeper conceptual checks:**
8. **Internal consistency sweep.** Propagated the Round-1 corrections into the Goal landing (3.85–3.95 → 3.90–4.00),
   the evidence table (2 rows still said "dominate every band"/"PAPILA+AIROGS direct attack"), and the memory file.
9. **"Spatial is only worth 0.056 dB" objection (major, resolved not fixed-away).** Made explicit that the 0.056 was
   a *full-image* artifact already broken by the disc crop (−0.146); high-res/denoising are *incremental over p1disc*
   (diminishing returns) → tempered the spatial budget. Prevents a fatal-looking contradiction with the ceiling memo.
10. **Raw vs calibrated slope (major honesty).** Made explicit that RAW slope is 0.54–0.57 and only the
    calibrated/disattenuated slope clears 0.6 — the plan no longer implies raw ≥0.6.
11. **Added** a detected-disc-center co-probe (deterministic fix for the disc-jitter wash mechanism).

**Residual uncertainties (not fixable by more research — the probes are the arbiters):** PAPILA domain transfer
(P-A2), ungated-ensemble severity gain (P-A1/A3), high-res feature degradation (P-B1). These are empirical and
cheap to settle; none blocks the reliable composition-adjusted deliverable. **After 3 rounds, no major
logical/factual problems remain**; the plan is honest about what is reliable (spatial stack + composition framing)
vs. a stretch (severity → decisive sub-4.0), and every remaining unknown is gated by a cheap pre-registered probe.
