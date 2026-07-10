# Kickoff prompt — implement the sub-4.0 encoder + data plan (paste into a fresh session)

You are continuing a glaucoma ML research project: predict the Humphrey 24-2 visual field (52 dB values)
from a fundus photo. A prior session ran diagnostics that **overturned the old "fundus-only is at its
ceiling" verdict** — it is DATA-limited, not information-limited — and wrote a plan to push native
fundus-only pooled MAE toward < 4.0 (which also lifts the 3.75 longitudinal SOTA). Your job is to execute
that plan rigorously, and to leave this prompt, the plan, and the project memory **better than you found
them** (see "Ratchet" below). This is real research: most experiments will fail; your value is running the
cheapest decisive one next, gating honestly, and never fooling us.

## Read these first, in this order (do not skip)
1. `decoder/specs/sub4_encoder_data_plan.md` — the plan you are executing (phases, tasks, pre-committed gates).
2. `decoder/specs/p1_disc_roi_results.md` — the current champion (P1 disc-only) + how it was evaluated.
3. `decoder/specs/fundus_only_ceiling_design.md` §6 — the binding eval protocol (§6.5 decision rule).
4. Project memory: `~/.claude/.../memory/MEMORY.md` and the files it indexes, especially
   `fundus-only-noise-floor-and-ceilings` (the D1=data-limited finding) and `p1-disc-roi-result`.
5. `CLAUDE.md` (repo root) — pipeline + architecture conventions.
Then skim the analysis scripts you will reuse: `paired_decision.py`, `d2_template_partial.py`,
`composition_report.py`, `diag_d1_learning_curve.py`, `eval_oof_cached.py`, `eval_ckpt.py`.

## Non-negotiable constraints (violating these invalidates results or crashes the box)
- **One torch process at a time.** 16 GB box, a few hundred MB free under load — it OOM-kills any second
  torch/heavy process. Serialize every train/eval. Do NOT run a numpy analysis alongside a torch job.
  (Last session's TTA eval and a scout were both OOM-killed this way.)
- **Never re-split or reseed** the folds (`decoder/results/cv_long/`). Score **out-of-fold, pooled over all
  631 records, against the RAW VF only.** Fit any per-fold quantity (calibration, template, probe) on that
  fold's TRAIN split only.
- **Eval crops with NO TTA.** TTA rotates crops → border/latent-cache blowup (broken + OOMs).
- **No 500 MB checkpoints in git.** A new flag's default-OFF must be byte-identical to the current model
  (add a unit test proving it).

## How to decide anything (the §6.5 protocol — already coded)
- **Primary endpoint = pooled OOF MAE vs raw VF.** Everything else is secondary.
- **Promote a change** iff `python decoder/paired_decision.py --new <tag> --ref <champion>` passes ALL of:
  pooled ΔMAE ≤ −0.12, pooled 95% CI upper < 0, ≥4/5 folds negative, severe ΔMAE 95% CI upper < +0.15,
  raw slope not worse. **A single fold can KILL but never PROMOTE** (the M2 lesson — one fold washed out).
- **Do not run** any method whose pre-registered expected effect < 0.12 dB (method-level MDE). Record it as
  sub-noise instead.
- **"native < 4.0" claim** needs the candidate's OWN patient-bootstrap 95% upper < 4.0 (point est ≤ 3.65).
- Report every headline with the companion metrics (§6.3): disattenuated slope, severity-stratified table,
  Bayes-floor multiples, patient-bootstrap CI. Use `composition_report.py` + `d2_template_partial.py`.

## Hard-won lessons (do not relearn these the expensive way)
- **The lever is a better/more data-efficient ENCODER, then more DATA. NOT metadata** — Diag B showed
  GRAPE's IOP/CCT/age/OCT-RNFL add partial-corr 0.23 but ~0 dB over the fundus (the disc crop already
  extracts the RNFL). Don't build a metadata model expecting MLEDL's −0.3 dB.
- **Don't chase the severe band or slope ≥ 0.60.** Severe is noise-bounded (Bayes floor 3.16; a same-eye
  prior VF only reaches 4.46). Report it; do not optimize it. Slope ≥ 0.60 is free via disattenuation/calibration.
- **The current champion is P1 disc-only** (`p1disc_f{0..4}_best.pth`, pooled no-TTA 4.113). Its gain is
  SPATIAL (eyeCorr/partial-corr 0.198→0.297), not severity. A *wider* crop dilutes it (0.40 lost eyeCorr).
- **Cheap experiment before expensive.** The plan's bake-off (Task A2) is a frozen-feature probe with NO
  retrain — run it before any 5-fold retrain (~4 hr serialized, 40 min/fold).
- **Domain > scale at 224 (measured):** a generic DINOv2-large frozen encoder is WORSE than retinal
  RETFound-MAE on our data (−0.072 sev_corr, −0.048 spatial). Only *retinal-domain* encoders are worth
  chasing — and those (RETFound-DINOv2, DINOv3) are **gated HF repos needing a logged-in token with the
  license accepted**; this box has none, so plan for that access wall before promising an encoder swap.
- **Gate the bake-off on the IN-RUN RETFound-MAE baseline (0.724), not the historical 0.738.** D1 used
  pre-norm `_encode_prefix`; `encoders.encode_prefix` applies `enc.norm` (post-norm, = `training._encode`).
  Same-pipeline comparison removes the −0.014 norm confound. Don't reintroduce it by mixing the two numbers.

## Your work loop (repeat until the plan's phases are exhausted or a gate stops you)
1. Pick the **cheapest decisive** open task from the plan (start: Task A1 encoder loader → A2 bake-off).
2. TDD for code (write failing test → run → implement → pass → commit); default-OFF byte-identical.
3. Run the experiment **alone** (one torch process). Cache features/preds so reruns are cheap.
4. Apply the pre-committed gate. **If it passes**, go deeper (integrate → fold-0 scout → gated 5-fold →
   §6.5). **If it fails**, record the number + why, and pivot to the next lever. Never move the goalpost.
5. **Update the trail** (Ratchet, below) before moving on.

## Ratchet — leave it better than you found it (this is the "always improving" part)
After every milestone (a gate decision, a promoted change, a killed idea):
- Append the result + interpretation to `decoder/specs/sub4_encoder_data_plan.md` (mark tasks done/dead)
  and to a running "what we tried and why" log.
- Update project memory: the champion number, the live frontier, any falsified assumption. Update the
  `MEMORY.md` index line. Delete memories that turn out wrong.
- **Improve THIS prompt** for the next session: refresh the numbers-to-beat, add any new hard-won lesson,
  and repoint "start here" to the next open task. The next session should start smarter than you did.

## Numbers to beat right now (update these as you go)
- Champion pooled no-TTA OOF: **p1disc 4.113** (m1sev 4.259). Sub-4.0 needs 95% upper < 4.0. UNCHANGED this session.
- Encoder bake-off IN-RUN baseline (post-norm `encode_prefix`, `diag_encoder_bakeoff.py`): RETFound-MAE
  sev_corr@505 = **0.724**, spatial partial-corr = **0.176**. (Historical pre-norm D1 anchor was 0.738; the
  −0.014 is a norm artifact — **gate on the in-run 0.724**, not 0.738.) Trained decoder sev_corr **0.807**;
  sub-4.0 frontier ≈ **0.875**.
- **Bake-off status (Session 5): DINOv2-large FAILED** (sev 0.653 = −0.072, spatial 0.128 = −0.048; generic
  SSL < retinal MAE). **RETFound-DINOv2 + DINOv3 are GATED HF repos** (401, box has no HF token) → the top
  pick is UNTESTED. VisionFM has no local weights.

## START HERE (next open task — decision-gated)
The encoder lever is unresolved, not dead. Pick one:
- **(a) Test the top pick** (if the user grants HF access / provides a token with the RETFound-DINOv2 +
  DINOv3 license accepted): `python decoder/diag_encoder_bakeoff.py --cache --only retfound_dinov2` then
  `--probe`; apply the same +0.03 sev / +0.05 spatial gate vs the in-run 0.724 / 0.176. If it clears →
  Task A3 (integrate, fold-0 scout vs p1disc_f0 4.101, gated 5-fold + `paired_decision.py`).
- **(b) Pivot to Phase B / Task B2** (data-efficiency without new data, no access needed): VF-manifold
  decoder warm-start (gate: fold-0 sev_corr +≥0.02) and disc-crop geometric augmentation (gate: fold-0 MAE
  not worse → full-CV §6.5). Both are cheap fold-0 scouts on the memory-tight box.

## Honesty mandate
Report failures with the actual numbers. Distinguish "provably sub-noise/dead" from "not yet tested."
State when something is a judgment call and leave it to the human. Verify before claiming done (run the
command, show the output). The strongest result this project can publish is an honest one — a rigorous
negative (the ceiling) is worth more than an inflated 3.99.
