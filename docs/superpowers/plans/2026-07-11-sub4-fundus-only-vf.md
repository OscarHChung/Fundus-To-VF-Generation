# Sub-4.0 Fundus-Only VF — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> This is a **gated research plan**: Phase-0 probe tasks each end in a pre-committed GATE; a failed gate SKIPS its
> dependent Phase-1 task (it does not block the plan). Code-infrastructure tasks use full TDD; experiment/probe
> tasks state an **objective, method, and binding gate**. Design spec:
> `docs/superpowers/specs/2026-07-11-sub4-fundus-only-vf-design.md`.

**Goal:** Push the fundus-only-inference 24-2 VF model toward native pooled MAE < 4.0 (point) with calibrated
slope ≥ 0.6, add PAPILA external validation of the fundus→severity component, and package the composition-adjusted
head-to-head vs the 2026 SOTA — for Ophthalmology Science.

**Architecture:** Keep the frozen-encoder → `PerPointAttention`(Garway-Heath prior) → per-point head pipeline
(+ M1 severity head + P1 disc-only crop = champion `p1disc`, 4.113). Attack the *input* side (higher-res disc
frozen features, ungated retinal-FM ensemble for severity, PAPILA severity co-training) and the *training-signal*
side (trajectory target-denoising), never the decoder-rescaling side (proven MAE-neutral). Every lever is a
default-OFF byte-identical flag decided by a no-train frozen-feature probe before any GPU spend.

**Tech Stack:** PyTorch (MPS), `decoder/encoders.py` swappable frozen backbones, NumPy ridge probes, the existing
`decoder/` analysis scripts (`diag_encoder_bakeoff.py`, `diag_d1_learning_curve.py`, `d2_template_partial.py`,
`paired_decision.py`, `composition_report.py`), `build_longitudinal_grape.py`, `longitudinal_dataset.py`.

## Global Constraints

- **Fundus-only at inference; longitudinal + external data at TRAIN only.** Frozen encoder → custom decoder;
  2-stage pretraining; Garway-Heath sectoring always on.
- **Free-download, ungated-only data** — no requests/DUAs/dbGaP and no gated-login HF repos (RETFound-DINOv2/DINOv3
  excluded).
- **Never re-split / reseed the GRAPE folds.** Score OOF pooled over 631 records vs RAW VF; per-fold quantities
  (calibration, template, probe) fit on that fold's TRAIN split only. **PAPILA never touches the cv_long folds.**
- **Memory: 16 GB MPS box, exactly ONE torch process at a time.** Serialize all cache/train/eval runs. Eval
  no-TTA for crops (TTA OOMs). Long trainings launch detached: `nohup … & disown` (NOT harness background, NOT
  `setsid`); verify exactly one `train_lora_cached`/cache process before each launch.
- **No >100 MB checkpoints in git.** Any new flag default-OFF must be **byte-identical** to the current model.
- **Do not run** any method whose pre-registered expected effect < 0.12 dB standalone (method-level MDE) except as
  a stack component.
- **Promotion = §6.5 rule 2** (`paired_decision.py`): pooled ΔMAE ≤ −0.12 AND pooled 95% CI upper < 0 AND ≥4/5
  folds negative AND severe-band ΔMAE 95% CI upper < +0.15 AND raw slope not worse. Native "<4.0" claim needs the
  candidate's own patient-bootstrap 95% upper < 4.0.

---

## Phase 0 — Data acquisition + cheap frozen-feature probes (no training; the decision gates)

### Task 0: Acquire + harmonize PAPILA (and AIROGS subset) to our schema

**Files:**
- Create: `data/external/papila/` (raw download), `data/external/papila_records.json` (harmonized)
- Create: `decoder/build_external_papila.py`
- Test: `decoder/tests_external_papila.py`

**Interfaces:**
- Produces: `data/external/papila_records.json` = list of
  `{"eye_id": str, "image": "abs/path.jpg", "Laterality": "OD"|"OS", "md": float, "label": "healthy"|"suspect"|"glaucoma", "source": "papila"}`.
  Consumed by Task 3 (co-training probe) and Task 13 (external validation).

- [ ] **Step 1: Write the failing test** (`decoder/tests_external_papila.py`)

```python
import json, os
def test_papila_records_schema():
    recs = json.load(open("data/external/papila_records.json"))
    assert len(recs) >= 200
    r = recs[0]
    assert set(r) >= {"eye_id","image","Laterality","md","label","source"}
    assert r["Laterality"] in ("OD","OS") and isinstance(r["md"], float)
    assert os.path.exists(r["image"])
    # glaucoma-range MD present (not all healthy) so severity transfer is meaningful
    assert any(x["md"] < -6 for x in recs)
```

- [ ] **Step 2: Run it, verify it fails** — `python -m pytest decoder/tests_external_papila.py -q` → FAIL (file missing).

- [ ] **Step 3: Download PAPILA** — figshare record 14798004 (CC-BY): disc-centered fundus images + the clinical
  `patient_data_od.xlsx`/`patient_data_os.xlsx` (age, MD, diagnosis). Save under `data/external/papila/`.
  Document the exact URL + license in a `data/external/papila/SOURCE.md`.

- [ ] **Step 4: Implement `decoder/build_external_papila.py`** — read the two clinical sheets (OD/OS), join to image
  files, map diagnosis→{healthy,suspect,glaucoma}, keep eyes with a numeric MD, write `papila_records.json`. Log
  the count and MD range.

- [ ] **Step 5: Run tests, verify pass** — `python -m pytest decoder/tests_external_papila.py -q` → PASS.

- [ ] **Step 6 (optional AIROGS, weak aux):** note in `SOURCE.md` that AIROGS is **binary RG/NRG** (not graded); only
  wire it if Task 3 wants a binary auxiliary head. Do NOT block on it.

- [ ] **Step 7: Commit** — `git add decoder/build_external_papila.py decoder/tests_external_papila.py data/external/papila/SOURCE.md && git commit -m "feat: harmonize PAPILA (fundus+MD) for severity transfer + external validation"`
  (do NOT commit the raw images — add `data/external/papila/*.jpg` to `.gitignore`).

### Task 1: High-res / disc-crop / detected-center cache capability in the bake-off

**Files:**
- Modify: `decoder/diag_encoder_bakeoff.py` (`cache_encoder`, `npz_path`, `main`) — anchors: `cache_encoder` L35,
  `encode_prefix` call L52, RETFound sin-cos regen via `encoder/RETFound_MAE/util/pos_embed.py:get_2d_sincos_pos_embed`,
  disc crop `training.disc_crop_pil` L277.
- Modify: `decoder/encoders.py` — let the RETFound-MAE `encode_prefix` accept a non-224 input by bypassing timm's
  `PatchEmbed` size assert (`h = enc.patch_embed.proj(x).flatten(2).transpose(1,2)`) and regenerating the pos-embed.
- Test: `decoder/tests_bakeoff_highres.py`

**Interfaces:**
- Produces: `cache_encoder(name, view="full", input_size=224, disc_center="fixed")` writing
  `results/auto/bakeoff_{name}__{view}{input_size}[_det].npz` with the SAME keys as today (`sev, spat, md, vf52,
  lat, pid, fold, grid`), so `probe()` runs unchanged (its descriptors are grid-agnostic).

- [ ] **Step 1: Write the failing test** (`decoder/tests_bakeoff_highres.py`)

```python
import torch, decoder.encoders as EN
def test_retfound_encode_prefix_highres_grid():
    enc = EN.load_encoder("retfound_mae")
    for sz, g in [(224,14),(384,24),(448,28)]:
        x = torch.randn(1,3,sz,sz)
        h = enc.encode_prefix(x, input_size=sz)   # new kwarg; default 224 byte-identical
        assert h.shape == (1, 1+g*g, enc.dim)
def test_default_224_byte_identical():
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(1,3,224,224)
    assert torch.allclose(enc.encode_prefix(x), enc.encode_prefix(x, input_size=224), atol=1e-6)
```

- [ ] **Step 2: Run it, verify it fails** — `python -m pytest decoder/tests_bakeoff_highres.py -q` → FAIL (`encode_prefix` has no `input_size`).

- [ ] **Step 3: Implement** the `input_size` path in `encoders.py` RETFound-MAE prefix (Conv patch-embed bypass +
  `get_2d_sincos_pos_embed(1024, sz//16, cls_token=True)` pos-embed, prepend CLS, run blocks, `enc.norm`), and add
  `view`/`input_size`/`disc_center` params to `cache_encoder` (disc view via `training.disc_crop_pil`; `disc_center
  ="detected"` = green-channel brightness/vessel centroid in the laterality quadrant, else the fixed box).

- [ ] **Step 4: Run tests, verify pass** — `python -m pytest decoder/tests_bakeoff_highres.py -q` → PASS.

- [ ] **Step 5: Commit** — `git add decoder/encoders.py decoder/diag_encoder_bakeoff.py decoder/tests_bakeoff_highres.py && git commit -m "feat: high-res + disc + detected-center frozen-feature caching (default 224 identical)"`

### Task 2 (EXPERIMENT · gate): High-res disc probe — P-B1

**Objective:** does a higher-resolution disc (or full) frozen view beat RETFound-MAE@224 on severity and/or the
within-eye spatial channel, and does the high-res disc push spatial partial-corr toward the 0.35 P2 gate?

**Method (ONE torch process at a time):** cache the set, then probe (numpy):
```bash
for v in "full 224 fixed" "full 384 fixed" "full 448 fixed" "disc 224 fixed" "disc 384 fixed" \
         "disc 448 fixed" "disc 224 detected" "disc 384 detected"; do \
  set -- $v; python decoder/diag_encoder_bakeoff.py --cache --only retfound_mae --view $1 --input-size $2 --disc-center $3; done
python decoder/diag_encoder_bakeoff.py --probe   # prints sev_corr@505, sev_oof, spatial pcorr+CI per cache
```
Anchors: RETFound-MAE@224 full sev_corr@505 ≈ 0.724, spatial pcorr ≈ 0.176; disc@224 spatial ≈ 0.297 (trained).

**GATE (pre-committed):** advance a config to Phase 1 (Task 8) iff sev_corr@505 ≥ +0.03 OR spatial pcorr ≥ +0.05
over the @224 counterpart, AND the config's own spatial pcorr does not *drop* vs disc@224. Record the winning
`(view,input_size,disc_center)`. If nothing clears → high-res dead; skip Task 8.

- [ ] Run cache set (serial). - [ ] Run probe. - [ ] Record winner + gate verdict in the design doc's review log.

### Task 3 (CODE + EXPERIMENT · gate): PAPILA severity-transfer probe — P-A2

**Files:** Create `decoder/diag_papila_severity.py` (reuse `diag_encoder_bakeoff.oof_sev_corr` + a PAPILA cache).

**Objective:** does adding PAPILA eyes to the frozen-feature→MD ridge raise GRAPE OOF sev_corr (evaluated only on
GRAPE, fold-disjoint)?

- [ ] **Step 1:** cache RETFound-MAE features for PAPILA images (disc view to match the champion), z-score MD within
  each dataset to neutralize the 24-2/30-2 scale gap.
- [ ] **Step 2:** for each GRAPE fold, fit ridge on GRAPE-train **+ all PAPILA** vs GRAPE-train-only; predict GRAPE
  val; pool OOF; compare sev_corr and sev_MAE. Patient-bootstrap the Δ.
- [ ] **GATE:** GRAPE OOF sev_corr rises by ≥ +0.02 with PAPILA added → carry to Task 10. Else try disc-only feats /
  per-dataset feature centering; if still <+0.02, record "PAPILA severity transfer does not hold (domain shift)"
  and skip Task 10 (PAPILA still used for external validation, Task 13).
- [ ] Commit the probe script.

### Task 4 (CODE + EXPERIMENT · gate): Ungated retinal-FM ensemble probe — P-A1/A3

**Files:** Modify `decoder/encoders.py` (add `retizero`, `retfound_green`, `visionfm` loaders — ungated GitHub/
release weights only; verify each downloads without a login and has a research-OK license before wiring).

**Objective:** does concatenating pooled frozen features from RETFound-MAE ⊕ {RetiZero, Green, VisionFM} raise
severity sev_corr@505 (severity head only — members have different grids, so NOT the spatial decoder)?

- [ ] **Step 1 (test):** `tests_encoders.py`-style shape test per new loader (skip-if-weights-absent).
- [ ] **Step 2:** cache each encoder once (serial); add an hstack option to `diag_encoder_bakeoff.probe()` that
  concatenates the `sev` arrays of two/three caches and runs `oof_sev_corr` / `sev_curve_505` on the union.
- [ ] **GATE:** best ensemble sev_corr@505 ≥ RETFound-MAE + 0.03 → carry to Task 9. Else record "ungated ensemble
  marginal" and skip Task 9.
- [ ] Commit loaders + probe change (not weights).

### Task 5 (EXPERIMENT · gate, conditioned on Task 2): Multi-crop feature-fusion probe — P-B3

**Objective:** does concatenating the best disc + full spatial descriptors clear the 0.35 P2 build gate?
**Method:** hstack the `spat` vectors of the winning disc cache (ideally Task-2 high-res disc) + full cache; run
`diag_encoder_bakeoff.spatial_pcorr` on the union.
**GATE:** concat spatial pcorr ≥ 0.35 → build feature-fusion head (Task 8b); ≤ 0.25 → dead; 0.25–0.35 → hold. Run
only after Task 2 (fusion on 224 features nearly replays a prior null).

- [ ] Run fusion probe. - [ ] Record verdict.

### Task 6 (EXPERIMENT · gate): Ordinal per-point head probe — P-C1

**Objective:** does an ordinal/CORAL readout beat plain ridge on the frozen features (pointwise-R / sev_corr)?
**Method:** offline on the 5 OOF folds, compare ridge vs a CORAL/SORD ordinal probe (CLS→eye-mean bins;
per-point→dB bins). **GATE:** ordinal beats regression by ≥ +0.02 corr → carry to Task 11; else drop (Agent 5's
prior: likely hits the same feature ceiling).

- [ ] Run probe. - [ ] Record verdict.

---

## Phase 1 — Build + stack survivors, then the honest 5-fold CV (train; one torch process at a time)

### Task 7: Trajectory target-denoising (applies to ALL 631 records; the one non-gated train lever)

**Files:**
- Modify: `build_longitudinal_grape.py` (add per-record `hvf_denoised`), `decoder/longitudinal_dataset.py`
  (`__getitem__` loads `hvf_denoised` for the TRAIN split only; val/eval keep raw `hvf`), add a `--denoise-target`
  flag threaded through `decoder/train_lora_cached.py` (default OFF byte-identical).
- Test: `decoder/tests_denoise_target.py`

**Interfaces:** Produces `hvf_denoised: [8×9]` per record = per-(PatientID,Laterality) per-point robust trajectory
fit (Theil-Sen slope + median intercept over the eye's ≥3 VFs by visit date) evaluated at the record's date;
masked points stay masked. Eval targets are UNCHANGED.

- [ ] **Step 1: Write the failing test**

```python
import json, numpy as np
def test_denoised_target_present_and_reasonable():
    recs = json.load(open("data/vf_tests/grape_longitudinal.json"))
    with_d = [r for r in recs if r.get("hvf_denoised") is not None]
    assert len(with_d) == len(recs)                       # all records get a denoised target
    r = with_d[0]
    a = np.array(r["hvf"]); b = np.array(r["hvf_denoised"])
    valid = a < 99.0
    assert (b[valid] < 99.0).all()                        # masking preserved
    assert np.abs(a[valid]-b[valid]).mean() < 6.0         # denoise ≠ wild extrapolation
def test_eval_uses_raw_hvf():
    # dataset in val mode returns the raw field, not the denoised one
    import decoder.longitudinal_dataset as L
    # (construct a tiny val dataset and assert target == raw hvf) — see Step 3
```

- [ ] **Step 2: Run it, verify it fails** — `python -m pytest decoder/tests_denoise_target.py -q` → FAIL (`hvf_denoised` absent).
- [ ] **Step 3: Implement** the Theil-Sen per-point trajectory fit in `build_longitudinal_grape.py`; regenerate
  `data/vf_tests/grape_longitudinal.json`; gate the target swap behind `--denoise-target` in the dataset/trainer
  (val/eval untouched). Verify visit-1==baseline invariant still holds (maxdiff 0 on raw `hvf`).
- [ ] **Step 4: Run tests, verify pass** → PASS.
- [ ] **Step 5: Fold-0 scout** — train p1disc recipe + `--denoise-target` on fold 0 (`nohup … & disown`), eval
  no-TTA vs `p1disc_f0` (4.101). Record Δ (expect ~−0.05..−0.12; sub-MDE alone — keep as a stack component).
- [ ] **Step 6: Commit** code (not the regenerated JSON if >100 MB; else commit).

### Task 8 (GATED on Task 2): High-res disc training + eval flag

**Files:** Modify `decoder/training.py` (thread the winning `input_size`/`disc_center` into `_encode` L907 and the
disc-crop path L277; make `build_vf_to_patch_prior` L213 and `PerPointAttention.patch_pos` L436 a function of the
encoder grid `(gh,gw)` instead of hard-coded 196/13), `decoder/train_lora_cached.py` + `decoder/eval_ckpt.py`
(add `--input-size`/`--disc-center`, store in checkpoint, reload on eval). Test: grid-parameterized prior equals
the current prior at 224 (default-identity guard).

- [ ] Test (default-identity) → implement grid-parameterized prior + high-res path → test PASS.
- [ ] Fold-0 scout vs `p1disc_f0`; single fold can KILL not promote. Commit code.
- [ ] **Task 8b (only if Task 5 cleared 0.35):** feature-level disc+full fusion head into `PerPointAttention`
  (route disc tokens→peripheral GH sectors, full→central); test default-OFF identity; fold-0 scout.

### Task 9 (GATED on Task 4): Ungated-ensemble severity head

**Files:** Modify `decoder/training.py` (severity/M1 head consumes concatenated pooled features from the winning
encoder set via `encoders.load_encoder`; frozen; spatial decoder stays on RETFound-MAE grid), `train_lora_cached.py`
/`eval_ckpt.py` (`--encoder-set`, stored in ckpt). Test: default single-encoder byte-identical. Fold-0 scout. Commit.

### Task 10 (GATED on Task 3): PAPILA severity co-training

**Files:** Modify `decoder/train_lora_cached.py` to add PAPILA (fundus→MD) as an auxiliary severity-head batch
(z-scored MD, weight tuned on fold-0), keeping the per-point VF loss GRAPE-only. **Hold out any PAPILA eyes reserved
for Task 13 external validation** (or train GRAPE-only and keep ALL PAPILA for Task 13 — cleaner; decide per Task 3
result). Test: PAPILA batch never enters the VF/per-point loss or the cv_long eval. Fold-0 scout. Commit.

### Task 11 (GATED on Task 6): Ordinal per-point head

**Files:** Modify `decoder/training.py` loss/head to a CORAL/SORD ordinal formulation (keep GH-weighting + CCC as
aux), behind `--ordinal-head` default-OFF. Test default-OFF identity. Fold-0 scout. Commit.

### Task 12: Stack survivors + full 5-fold CV + §6.5 decision

- [ ] Combine all gate-passers into one recipe (denoise-target + winning high-res disc + [ensemble sev head] +
  [PAPILA co-train] + [ordinal]). Fold-0 scout of the STACK vs `p1disc_f0`.
- [ ] Train folds 1–4 serially (`nohup … --warm-start long_global_f{f} … & disown`, ONE at a time; verify a single
  torch proc before each launch).
- [ ] Eval OOF no-TTA: `python decoder/eval_oof_cached.py --tag <stack> --no-tta --cache-dir results/auto/oof_cache_notta`.
- [ ] Decide: `python decoder/paired_decision.py --new <stack> --ref p1disc`. Promote iff §6.5 passes; record the
  native point estimate + patient-bootstrap CI (native "<4.0" needs CI upper < 4.0). Re-run `composition_report.py`,
  `d2_template_partial.py`, `make_scatterplot.py`, calibration.

---

## Phase E — External validation on PAPILA (acceptance-critical)

### Task 13: `eval_external_papila.py` — fundus→MD generalization on an independent cohort

**Files:** Create `decoder/eval_external_papila.py`. Test: `decoder/tests_external_eval.py` (no PAPILA eye is in
cv_long; scale-free r is computed on held-out PAPILA).

- [ ] **Step 1 (test):** assert the evaluated PAPILA eye-ids are disjoint from every cv_long fold record, and that
  the report contains `r`, `r_ci`, `mae_calibrated`, stratified by PAPILA label.
- [ ] **Step 2:** load the final **GRAPE-trained** stacked model (the cleanest external test — no PAPILA in training,
  so ALL PAPILA is external); run its eye-mean (MD proxy) on `papila_records.json`; report Pearson/Spearman r
  (scale-free) + MAE after a train-fit linear calibration; patient-bootstrap CI; stratify healthy/suspect/glaucoma;
  save a PAPILA scatter. If Task 10 (co-training) was promoted, additionally evaluate on the reserved held-out
  PAPILA split only.
- [ ] **Step 3:** run tests → PASS. Report the number honestly (a modest external r still strengthens the paper; a
  collapse is a publishable cross-camera-transfer limitation).
- [ ] **Step 4: Commit** — `git commit -m "feat: PAPILA external validation of fundus->MD (independent cohort)"`

---

## Phase 2 — Paper deliverables (framing; partly independent of Phase 1)

### Task 14: Total-Deviation reporting (C2) — comparability to the TD-based SOTA

**Files:** Create `decoder/td_report.py` (subtract the 24-2 age-normal using GRAPE `Age` from the Baseline sheet;
pooled TD-MAE ≡ sensitivity-MAE, r/slope invariant — a comparability + scatter deliverable, not a perf change).

- [ ] Join Age → records; compute TD; emit TD-space pooled + eye-stage-stratified MAE + a TD scatter to match
  TDV-Net's convention. Commit.

### Task 15: Head-to-head + ceiling refresh

- [ ] Re-run `composition_report.py` on the final tag; assemble the honest table: composition-adjusted (ours
  3.37–3.62 vs 3.91; theirs 4.69–4.81 under our mix), eye-level strata (moderate+severe decisive, **mild a tie**),
  disattenuated slope + raw + calibrated (raw 0.54–0.57; calibrated/disattenuated ≥ 0.6 — report all three),
  protocol contrast, and the PAPILA external-validation paragraph (Task 13). Commit the assembled results doc.

---

## Self-Review

- **Spec coverage:** severity → Tasks 3 (PAPILA), 4 (ensemble); spatial → Tasks 2 (high-res), 5 (fusion), 7
  (denoise); loss → Task 6/11; external validation → Tasks 0+13; comparability/framing → Tasks 14–15; stack + CV →
  Task 12. Output-side *rescaling* correctly excluded. All spec phases (0/1/E/2) mapped.
- **Gating honored:** every Phase-1 build task (8/9/10/11) is gated on its Phase-0 probe (2/4/3/6); a failed gate
  skips the task, never blocks the plan. Task 7 (denoise) + Task 13 (external val) + Tasks 14–15 are non-gated.
- **Placeholders:** infra tasks (0,1,7,13) carry test + implementation + commands; experiment tasks carry
  objective/method/binding gate. Conditional build tasks (8–11) state files + method + gate + a default-identity
  test; their exact head/loss code is deliberately fixed at execution time by the probe outcome (noted, not a TBD).
- **Type consistency:** `cache_encoder(name, view, input_size, disc_center)` and the `bakeoff_{name}__{view}{size}.npz`
  keys are reused identically in Tasks 1/2/3/4/5; `papila_records.json` schema is fixed in Task 0 and consumed
  unchanged in Tasks 3/13; `--denoise-target`/`--input-size`/`--disc-center`/`--encoder-set`/`--ordinal-head` are
  default-OFF byte-identical flags threaded through `train_lora_cached.py` + `eval_ckpt.py` consistently.
- **Constraints:** one-torch-process serialization, `nohup … & disown` launches, no >100 MB checkpoints, PAPILA
  never in cv_long — all stated in Global Constraints and repeated at each risky task.
