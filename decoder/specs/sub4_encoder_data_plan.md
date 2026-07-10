# Sub-4.0 via Encoder + Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> This is a **research plan with pre-committed decision gates**, not a pure feature build: code tasks use
> full TDD; experiment tasks state an objective, a method, and a binding gate whose failure stops the phase.

**Goal:** Break the fundus-only VF-prediction ceiling toward native pooled MAE < 4.0 (and lift the 3.75
longitudinal SOTA) by attacking the *data limitation* that diagnostic D1 revealed — primarily a stronger,
more data-efficient frozen encoder; secondarily data expansion — under the existing leak-free 5-fold §6.5
protocol.

**Architecture:** Keep the current pipeline (frozen retinal backbone → `PerPointAttention` with a
Garway-Heath prior → per-point head, + M1 severity head + P1 disc-only crop). Make the **frozen backbone
swappable** behind a `--encoder` flag; decide the winner by a cheap grid-agnostic frozen-feature bake-off
BEFORE paying for any full retrain; only then integrate + retrain + gate.

**Tech Stack:** PyTorch (MPS), `rmaphoh/RETFound` loader (ships RETFound-MAE / RETFound-DINOv2 / DINOv2 /
DINOv3 via HuggingFace), NumPy/scikit-style ridge probes, the existing `decoder/` analysis scripts.

## Global Constraints (copy verbatim into every task's context)

- **Never re-split / reseed** the folds. Score OOF, pooled over 631 records, against the **RAW** VF only.
  Any per-fold quantity (calibration, template, probe) is fit on that fold's **train** split only.
- **Memory: 16 GB box, ~a few hundred MB free under load. Exactly ONE torch process at a time** — no
  concurrent torch/heavy-numpy (it OOM-kills). Serialize every training/eval run. TTA eval OOMs; keep eval
  no-TTA for crops.
- **Primary endpoint = pooled OOF MAE vs raw VF.** Promotion rule = §6.5 rule 2 (already coded in
  `decoder/paired_decision.py`): pooled ΔMAE ≤ −0.12 AND pooled 95% CI upper < 0 AND ≥4/5 folds negative
  AND severe-band ΔMAE 95% CI upper < +0.15 AND raw slope not worse. Native "<4.0" claim needs the
  candidate's own patient-bootstrap 95% upper < 4.0 (§6.5 rule 5).
- **Do not run** any method whose pre-registered expected effect < 0.12 dB (method-level MDE).
- **No 500 MB checkpoints in git.** Default-OFF of any new flag must be byte-identical to the current model.

---

## Evidence base (this session's diagnostics — why this plan)

| Lever | Diagnostic | Result | Verdict |
|---|---|---|---|
| **Data/encoder ceiling** | D1 learning curve (`diag_d1_learning_curve.py`) | frozen-feature sev_corr **still rising +0.087 over n=210→505** (≫ the 0.02/doubling falsification bar) | **DATA-limited, not info-limited** — the old §0 ceiling verdict is falsified; more data / better encoder can lift sev_corr toward the 0.875 sub-4.0 frontier |
| **Metadata** | incremental-value probe (`diag_metadata_value.py`) | RNFL/age/CCT add partial-corr 0.23 but **0 dB** over the fundus for severity | **Deprioritize** — the fundus already extracts the structure; not MLEDL's −0.3 dB path for us |
| **Better encoder** | feasibility (`diag_metadata_inventory` + web) | RETFound-DINOv2 > RETFound-MAE on retinal benchmarks; DINOv2 > RETFound on glaucoma; all public, drop-in via one loader | **Top pick** — feasible + data-efficient (what a data-limited regime needs) |
| **Longitudinal** | headroom (`diag_longitudinal_headroom.py`) | 42% first-visit = fundus-only; fundus ΔMAE → longitudinal ×0.42 | **Free beneficiary** — improving the fundus branch lifts the 3.75 SOTA too |

Base numbers to beat (no-TTA pooled OOF): **p1disc 4.113** / m1sev 4.259; frozen-feature linear-probe
sev_corr at n=505 = **0.738**; trained-decoder sev_corr = **0.807**; sub-4.0 frontier ≈ **0.875**.

---

## File structure

- **Create** `decoder/encoders.py` — the swappable frozen-backbone loader (one responsibility: given an
  `--encoder` name, return a frozen module exposing `encode_prefix(imgs)->(B,1+P,D)` plus its grid `(H,W)`
  and dim `D`). Isolates all backbone-specific loading/preprocessing from `training.py`.
- **Create** `decoder/tests_encoders.py` — encoder-loader unit tests (shape, determinism, default-identity).
- **Create** `decoder/diag_encoder_bakeoff.py` — extends `diag_d1_learning_curve` across encoders.
- **Modify** `decoder/training.py` — route the frozen backbone through `encoders.py` behind `--encoder`
  (default `retfound_mae` byte-identical); make `PerPointAttention`'s patch grid + GH prior a function of
  `(H,W)` instead of the hard-coded 196/13 (Task A3 only).
- **Modify** `decoder/train_lora_cached.py`, `decoder/eval_ckpt.py` — thread `--encoder` + store it in the
  checkpoint (so eval reloads the right backbone), mirroring how `disc_only`/`disc_half` are handled.
- **Reuse** `decoder/paired_decision.py`, `d2_template_partial.py`, `composition_report.py` for gating.

---

## Phase A — Encoder bake-off (top lever; cheap decisive experiment first, retrain only if it wins)

> **STATUS (Session 5, 2026-07-10):** A1 **DONE** (`decoder/encoders.py` + 4/4 tests). A2 **PARTIAL —
> BLOCKED on access.** The only *public* candidate, **DINOv2-large, FAILED the gate** (sev_corr@505
> 0.653 vs RETFound-MAE 0.724 = −0.072; spatial pcorr 0.128 vs 0.176 = −0.048 — worse on BOTH channels;
> general-domain SSL loses to the retinal-domain MAE, as theory predicts). But the plan's **actual top
> pick RETFound-DINOv2 and DINOv3-L are GATED HuggingFace repos** (401 — need the user's HF login +
> license acceptance) and **VisionFM has no local weights** — all three UNTESTED. So we may **not** yet
> conclude "encoder swap does not help"; only "the *general* DINOv2 does not." **USER DECISION (2026-07-10):
> "Use RETFound, not the dino one" → keep RETFound-MAE; do NOT pursue the gated retinal-DINOv2/DINOv3.
> Phase A is CLOSED (encoder = RETFound-MAE).** A3 not started (no winning encoder to integrate). Champion
> unchanged: **p1disc 4.113**. → Proceed to **Phase B** (data-efficiency, keeps RETFound-MAE, no access).

### Task A1: Swappable frozen-encoder loader

**Files:**
- Create: `decoder/encoders.py`
- Create: `decoder/tests_encoders.py`

**Interfaces:**
- Produces: `load_encoder(name: str) -> FrozenEncoder` where `FrozenEncoder` has
  `.encode_prefix(imgs: Tensor[B,3,H,W]) -> Tensor[B, 1+P, D]`, `.grid -> (gh, gw)` (P = gh*gw),
  `.dim -> int`, `.input_size -> int`. `name ∈ {retfound_mae, retfound_dinov2, dinov2_l, dinov3_l, visionfm}`.
- Consumes (in later tasks): `training.py` calls `load_encoder(args.encoder)` in place of `base_model`.

- [ ] **Step 1: Write the failing test** (`decoder/tests_encoders.py`)

```python
import torch, decoder.encoders as EN

def test_retfound_mae_default_shapes():
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(2, 3, enc.input_size, enc.input_size)
    h = enc.encode_prefix(x)
    assert h.shape == (2, 1 + enc.grid[0] * enc.grid[1], enc.dim)
    assert enc.grid == (14, 14) and enc.dim == 1024      # ViT-L/16 @224

def test_encode_prefix_is_deterministic_and_nograd():
    enc = EN.load_encoder("retfound_mae")
    x = torch.randn(1, 3, enc.input_size, enc.input_size)
    a = enc.encode_prefix(x); b = enc.encode_prefix(x)
    assert torch.allclose(a, b, atol=1e-5)               # eval mode, no MAE shuffle
    assert not a.requires_grad
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest decoder/tests_encoders.py -q` — Expected: FAIL (`encoders` missing).

- [ ] **Step 3: Implement `decoder/encoders.py`**

Wrap the existing RETFound-MAE path first (must reproduce `training._encode` exactly — no MAE random
patch shuffle, `enc.norm` applied). Add DINOv2/v3 + RETFound-DINOv2 via the `rmaphoh/RETFound` HF loader,
and VisionFM via its repo. Each returns the `(B,1+P,D)` prefix and reports its own `grid`/`dim`/`input_size`
(DINOv2 ViT-L/14 @224 → grid (16,16), D=1024; register tokens dropped). Keep weights frozen + `.eval()`.

```python
# decoder/encoders.py  (skeleton — fill each backbone; RETFound-MAE MUST match training._encode)
import torch, torch.nn as nn

class FrozenEncoder(nn.Module):
    def __init__(self, backbone, grid, dim, input_size, prefix_fn):
        super().__init__(); self.backbone = backbone.eval(); self.grid = grid
        self.dim = dim; self.input_size = input_size; self._prefix_fn = prefix_fn
        for p in self.backbone.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def encode_prefix(self, imgs):
        return self._prefix_fn(self.backbone, imgs)

def load_encoder(name):
    if name == "retfound_mae":
        from models_mae import mae_vit_large_patch16_dec512d8b
        m = mae_vit_large_patch16_dec512d8b(); _load_retfound_weights(m)
        return FrozenEncoder(m, (14, 14), 1024, 224, _mae_prefix)   # _mae_prefix mirrors training._encode
    if name in ("dinov2_l", "dinov3_l", "retfound_dinov2"):
        return _load_dinov2_family(name)     # rmaphoh/RETFound HF loader; grid (16,16), drop register tokens
    if name == "visionfm":
        return _load_visionfm()
    raise ValueError(name)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest decoder/tests_encoders.py -q` — Expected: PASS (the two RETFound-MAE tests; add
per-encoder shape tests as each backbone is wired, `@pytest.mark.skipif` when weights absent).

- [ ] **Step 5: Commit** — `git add decoder/encoders.py decoder/tests_encoders.py && git commit -m "feat: swappable frozen-encoder loader (default retfound_mae identical)"`

### Task A2: Frozen-feature bake-off across encoders (the decision, no retrain)

**Files:** Create `decoder/diag_encoder_bakeoff.py` (import `encoders.load_encoder` + reuse
`diag_d1_learning_curve.ridge_fit_predict` / `learning_curve`).

**Objective:** For each obtainable encoder, cache pooled frozen features (CLS ⊕ mean-patch) over the 631
images and run the **same** severity learning curve + a spatial partial-corr probe as D1, on our folds.

- [ ] **Step 1:** For each `name`, cache `{name}` features to `results/auto/bakeoff_{name}.npz` (reuse
  `diag_d1_learning_curve.cache_features`, parameterized by encoder). One torch process at a time.
- [ ] **Step 2:** Print, per encoder: OOF sev_corr at n=505, the full learning curve, and spatial
  partial-corr (reuse `d2_template_partial` logic on the probe residuals).
- [ ] **Step 3 — GATE (pre-committed):** Advance an encoder to Task A3 **iff** its linear-probe OOF
  sev_corr at n=505 exceeds RETFound-MAE's **0.738 by ≥ +0.03** (above the curve's ~0.02 resample noise)
  **OR** its spatial partial-corr exceeds RETFound-MAE's D1 value by ≥ +0.05. If no encoder clears the
  gate → **stop Phase A**, record "encoder swap does not help our data," proceed to Phase B.

### Task A3 (GATED on A2): Integrate the winning encoder + retrain + §6.5 gate

**Files:** Modify `decoder/training.py` (route backbone via `encoders.load_encoder`; make
`build_vf_to_patch_prior` and `PerPointAttention.patch_pos`/`prior_bias` a function of the encoder
`grid=(gh,gw)` — replace the hard-coded `196` and `13`/`patch_r = 13-int(r*13/7)` with `gh,gw`-derived
math), `decoder/train_lora_cached.py` + `decoder/eval_ckpt.py` (add `--encoder`, store in checkpoint,
reload on eval).

- [ ] **Step 1 (test):** Add `tests_method_p1`-style test: with `--encoder retfound_mae`, the re-gridded
  prior tensor equals the current hard-coded prior (default-identity guard).
- [ ] **Step 2:** Implement grid-parameterized prior + backbone routing; run the test → PASS.
- [ ] **Step 3 (fold-0 scout):** Retrain P1 disc fold-0 with the winning encoder (recipe = the p1disc
  recipe + `--encoder <winner>`). Eval no-TTA vs `p1disc_f0` (4.101). **Gate:** proceed to full CV only if
  fold-0 MAE is clearly better AND severe not worse (single fold can kill, never promote).
- [ ] **Step 4 (full CV, gated):** Retrain folds 1–4 serially; `python decoder/eval_oof_cached.py --tag
  <winner> --no-tta --cache-dir results/auto/oof_cache_notta`; then `python decoder/paired_decision.py
  --new <winner> --ref p1disc`. **Promote iff §6.5 passes.** Re-run `composition_report.py` +
  `d2_template_partial.py` on the winner.
- [ ] **Step 5:** Commit code (not checkpoints).

---

## Phase B — Data expansion / efficiency (justified by D1; pursue if Phase A under-delivers or in parallel research)

### Task B1: Additional paired fundus + 24-2 VF data (search + harmonize)
**Objective:** D1 says more paired eyes lift sev_corr. Web search + registries for open paired fundus+24-2
VF beyond GRAPE (candidates to check: private-but-requestable cohorts, any 2025-26 releases; UWHVF is
VF-only so it can only pretrain the VF-manifold, not the encoder→VF map).
- [ ] **Gate:** Only integrate a dataset that can be harmonized to `MultiImageDataset`'s schema AND scored
  patient-disjoint. If found, add its records to a *pretraining* pool (not the frozen eval folds — never
  re-split cv_long), pretrain the decoder, then fine-tune + eval on cv_long. Re-run `diag_d1` to confirm
  the curve extends.

### Task B2: Data-efficiency without new data
**Objective:** Squeeze more from 144 patients (what a data-limited regime rewards).
- [ ] Decoder warm-start from the VF-manifold: already have `pretrained_vf_ae.pth`; test initializing the
  per-point head from it. Gate: fold-0 sev_corr up ≥ +0.02.
- [ ] Stronger train-time augmentation on the disc crop (scale/shift jitter, not photometric — §5 #14
  showed ColorJitter hurts). Gate: fold-0 MAE not worse; keep only if the full-CV §6.5 passes.

---

## Phase C — Metadata spatial residual (cheap, low priority; the one untested metadata angle)

### Task C1: Does RNFL-sector predict the within-eye VF *pattern* beyond the fundus?
**Files:** extend `decoder/diag_metadata_value.py` with a spatial variant.
**Objective:** Diag B killed metadata for *severity*; structure-function is fundamentally *sectoral*, so
test the residual channel: partial-corr(RNFL 4-sector features, true within-eye residual | fundus residual).
- [ ] **Gate:** Only build a small metadata-fusion head if this partial-corr ≥ 0.15 (above the disc's own
  spatial contribution). Prior is low (RNFL is 4 coarse sectors; fundus already sees them) — one probe, no
  training, decide in minutes.

---

## Phase D — Longitudinal re-eval (free once the fundus branch improves)

### Task D1: Propagate the fundus gain into the 3.75 SOTA
- [ ] After any promoted fundus-branch improvement, re-run `python decoder/train_longitudinal.py` +
  `python decoder/eval_strata_longitudinal.py`. Expect the first-visit stratum (42%, `diag_longitudinal_headroom`)
  to drop by ~the fundus ΔMAE, lifting pooled by ≈0.42×ΔMAE. Report with patient-bootstrap CI.

---

## Self-review

- **Spec coverage:** encoder lever → Phase A; data lever → Phase B; metadata → Phase C (gated low); longitudinal
  → Phase D. All four diagnostic verdicts have a task.
- **Placeholders:** none — each task has files, method, exact commands/scripts, and a numeric gate.
- **Type consistency:** `load_encoder(name) -> FrozenEncoder` with `.encode_prefix/.grid/.dim/.input_size`
  used identically in A1/A2/A3; gates reference the concrete baselines (0.738 probe / 4.101 fold-0 / 4.113 pooled).
- **Order of EV:** A2 (cheap, no retrain, decisive) precedes A3 (expensive). Metadata is last because it was
  diagnosed weak. This matches "diagnose → gate → only then spend GPU."
```

---

## Progress log — what we tried and why (append-only)

### Session 5 (2026-07-10) — Phase A executed to a gated blocker

**A1 (DONE).** Built `decoder/encoders.py`: `load_encoder(name) -> FrozenEncoder` with a uniform
`encode_prefix/grid/dim/input_size` interface. `retfound_mae` reuses the already-loaded
`training.base_model` (no 2nd 3.7 GB read) and is **byte-identical to `training._encode`** (unit test
`test_retfound_mae_matches_training_encode`, atol 1e-5). DINOv2/v3/RETFound-DINOv2 load on demand via HF
transformers (run at OUR 224 pipeline resolution — grid derived from 224, not the model's native 518 —
for an apples-to-apples swap; register tokens dropped). VisionFM from local weights. 4/4 tests pass.
Commit `1dc787b` (+ dinov2 fix in `dbce9a4`).

**A2 (PARTIAL, BLOCKED).** `decoder/diag_encoder_bakeoff.py` caches CLS⊕mean-patch (severity) and a
grid-agnostic 4×4-pooled patch descriptor (spatial) over the 631 imgs per encoder, then runs — identically
for all — the D1 severity learning curve and a **leak-free per-point spatial partial-corr probe** (PCA→
multi-output ridge → `d2_template_partial` template-partial, template from each fold's train only). The
gate uses an **in-run** RETFound-MAE baseline (same post-norm `encode_prefix`) rather than the historical
pre-norm 0.738, to remove the norm confound — in-run baseline sev_corr@505 = **0.724**, spatial = **0.176**.

| encoder | source | sev_corr@505 | Δ | spatial pcorr | Δ | verdict |
|---|---|---|---|---|---|---|
| retfound_mae | baseline | 0.724 | — | 0.176 | — | — |
| **dinov2_l** | facebook/dinov2-large (public) | **0.653** | −0.072 | **0.128** | −0.048 | **sub-gate (worse on both)** |
| retfound_dinov2 | YukunZhou/RETFound_dinov2_meh | — | — | — | — | **BLOCKED: gated HF (401)** |
| dinov3_l | facebook/dinov3-vitl16 | — | — | — | — | **BLOCKED: gated HF (401)** |
| visionfm | local weights | — | — | — | — | **UNAVAILABLE: no weights on box** |

**Interpretation.** General-purpose DINOv2 is a *worse* frozen encoder than the retinal-domain
RETFound-MAE on our task — decisively, on both the severity and spatial channels. This is the theoretically
expected direction (domain-specific pretraining > scale here at 224). It does **not** falsify the plan's
encoder lever, because the lever was the *retinal* RETFound-DINOv2, which we could not obtain (gated). The
box has **no HF token** and the RETFound-DINOv2 / DINOv3 repos require accepting a license while logged in.

**Falsified/updated:** "swap DINOv2 for RETFound-MAE" as a *drop-in generic* win is **dead** (−0.072). The
narrower "retinal RETFound-DINOv2 beats RETFound-MAE" remains **untested** (needs HF access). Note the
in-run post-norm RETFound-MAE probe is 0.724 (vs the historical pre-norm 0.738) — a −0.014 norm artifact,
not a regression; gate on the in-run number.

**Next open task (decision-gated):** EITHER (a) user grants HF access → cache `retfound_dinov2` (+`dinov3_l`)
via `python decoder/diag_encoder_bakeoff.py --cache --only retfound_dinov2` then `--probe`, apply the same
gate; OR (b) accept the partial negative and start **Phase B / Task B2** (data-efficiency without new data:
VF-manifold decoder warm-start, disc-crop geometric augmentation — both cheap fold-0 scouts).

**RESOLVED (2026-07-10):** user chose to **keep RETFound-MAE** ("use retfound not the dino one"). Phase A
closed with no encoder swap. The generic-DINOv2 negative stands; the retinal-DINOv2 lever is left untested
by choice (re-openable later if HF access is granted — the loader + bake-off are ready, just `--cache
--only retfound_dinov2 && --probe`). Moving to Phase B. Cheapest decisive remaining probe is actually
**Task C1** (RNFL spatial-metadata partial-corr — numpy, minutes, no training) — run it before the B2
training scouts per "cheapest decisive first," then B2.
