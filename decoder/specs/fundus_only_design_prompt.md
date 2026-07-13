Design (do not implement) a plan for improving our fundus->24-2 VF model under a hard single-fundus-image inference constraint.

## Starting point

These are not the same number. Do not conflate them.

- Our headline 3.75 dB MAE / 0.709 calibrated slope is the LONGITUDINAL model, which consumes the eye's prior VF at inference. It violates the constraint below and is NOT a valid baseline for this task.
- Under fundus-only inference the honest number is MAE 4.29 dB (first-visit stratum: 4.545).
- Persistence (copy the prior VF) = 3.71 dB, i.e. it BEATS the longitudinal model. 3.71, not 3.75, is the bar any prior-using result must clear.
- Test-retest noise floor = 2.76 dB.
- Severe band (n=101): MAE 6.06, slope 0.482 -- both thresholds already fail here.

Thresholds we consider clinically meaningful: MAE < 4.0 dB and slope >= 0.6. Report both pooled AND on the severe band.

The question is whether sub-4.0 fundus-only MAE is reachable at all, not how to reach it. If your honest conclusion is that no proposal within the constraints has a good chance of clearing 4.0 at slope >= 0.6 on 631 records, say so plainly and identify the binding limit: data volume, label noise, or the structure-function information content of a disc photo. A well-argued "not reachable, here's the evidence" is a successful outcome of this task, not a failure.

Comparator: "Deep learning-based prediction of 24-2 visual field from fundus photographs in glaucoma." I want to know where we genuinely stand against it, including whether its reported metrics are computed on a comparable stratum.

## Hard constraints

- Keep the two-stage architecture: RETFound encoder + custom VF decoder with Garway-Heath sectoring. Both freely editable internally.
- Inference takes a single fundus image. Training may use additional signal (longitudinal priors, auxiliary targets, etc.).

## Method

1. Read `decoder/results/auto/iterations.md`, `decoder/results/champion/longitudinal_champion.json`, and the M1/M2 history. The M2 RNFL auxiliary head was a wash on full 5-fold OOF, and the longitudinal delta head learned ~nothing above persistence. Use those results to rule things out; do not re-propose them without stating what is different this time.

2. Propose candidate methods. Be creative -- anything within the constraints is fair game.

3. For each proposal state, up front: what would falsify it, and whether a plausible gain would survive both the 2.76 dB noise floor and fold-to-fold variance on 631 records (263 independent baseline eyes). Label any proposal whose expected gain is smaller than the fold standard error as sub-noise / unfalsifiable at this sample size.

4. Critique pass. One reviewer agent argues against each proposal on methodological grounds: leakage, eval validity, the noise floor, and statistical power at 631 records. A second agent independently checks each objection against the actual code and data before the objection is accepted, so the design does not get contorted around objections that turn out to be wrong. Fixer agents address confirmed objections. Up to 3 rounds. Record unresolved material objections in the design file as open risks rather than resolving them cosmetically.

You may write and run throwaway scripts to check assumptions: data statistics, noise-floor estimates, fold-level variance, ablation feasibility. Do NOT implement or train the proposed method.

## Deliverable

One design file containing:

(a) an honest assessment of where fundus-only stands vs the comparator;
(b) surviving proposals, each with falsification criteria and expected effect size relative to fold noise;
(c) proposals ruled out, and why;
(d) a pre-committed evaluation protocol: folds, metrics, comparisons, and the decision rule -- fixed before any training.
