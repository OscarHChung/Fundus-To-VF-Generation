"""Task 5 (P-B3) + Task 6 (P-C1) -- two cheap, independent frozen-feature probes on already-cached
GRAPE bake-off features (no torch, no training).

PROBE A (fusion, Task 5 / P-B3): does concatenating the disc@224 + full@224 SPATIAL descriptors
clear the P2 build gate (>=0.35 leak-free spatial partial-corr, see decoder/d2_template_partial.py)?
High-res disc-alone already FAILED (Task 2); this checks whether the orthogonal full-image view adds
enough eye-specific spatial signal when concatenated with the disc view at the resolution that
already works (224).

PROBE B (ordinal readout, Task 6 / P-C1): does a CORAL-style ordinal severity readout beat plain
ridge regression on the SAME frozen disc@224 `sev` features, for the per-eye MD (mean 24-2
sensitivity, dB) target? CORAL (Cao et al. 2020, simplified, no heavy deps beyond sklearn): bin the
continuous target into K ordinal bins on a FIXED dB grid (not fitted from data -> no leakage), then
fit K-1 binary "is bin >= k" logistic classifiers sharing the same standardized features; decode
back to a continuous estimate as the bin-probability-weighted mean of bin midpoints. Both methods
are evaluated identically: pooled leak-free OOF over the same 5 GRAPE folds, correlated (Pearson +
Spearman) against the same continuous md target.

Reuses `diag_encoder_bakeoff.spatial_pcorr` (leak-free per-point spatial partial-corr + patient
bootstrap CI) and `diag_d1_learning_curve.ridge_fit_predict` (the plain-ridge readout).

  python decoder/diag_fusion_ordinal_probe.py     # runs both probes, prints verdicts
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_encoder_bakeoff as BO
import diag_d1_learning_curve as D1

AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
DISC224 = os.path.join(AUTO, "bakeoff_retfound_mae__disc224.npz")
FULL224 = os.path.join(AUTO, "bakeoff_retfound_mae.npz")

FUSION_GATE = 0.35
FUSION_DEAD = 0.25
ORDINAL_GATE = 0.02


# --------------------------------------------------------------------------- PROBE A (P-B3 fusion)
def fusion_spatial_probe(disc_path=DISC224, full_path=FULL224, k=100, lam=30.0, boot=3000, seed=42):
    """Concatenate disc@224 + full@224 `spat` descriptors -> leak-free spatial partial-corr.

    Both caches are meant to be the SAME 631 GRAPE records in the SAME fold order (same build loop
    in diag_encoder_bakeoff.cache_encoder); verified here by pid/lat/fold equality rather than
    assumed -- if a future cache regeneration ever reorders rows, this raises instead of silently
    comparing misaligned eyes.
    """
    d_disc = np.load(disc_path)
    d_full = np.load(full_path)
    if not (np.array_equal(d_disc['pid'], d_full['pid']) and
            np.array_equal(d_disc['lat'], d_full['lat']) and
            np.array_equal(d_disc['fold'], d_full['fold'])):
        raise ValueError(
            "disc224 and full224 caches are not row-aligned (pid/lat/fold differ) -- refusing to "
            "hstack mismatched eyes. Regenerate one of the caches or add an explicit pid+lat join."
        )
    fused = np.hstack([d_disc['spat'], d_full['spat']])
    pc_fused, rc_fused, lo_fused, hi_fused = BO.spatial_pcorr(
        fused, d_disc['vf52'], d_disc['lat'], d_disc['pid'], d_disc['fold'],
        k=k, lam=lam, boot=boot, seed=seed)
    pc_disc, rc_disc, lo_disc, hi_disc = BO.spatial_pcorr(
        d_disc['spat'], d_disc['vf52'], d_disc['lat'], d_disc['pid'], d_disc['fold'],
        k=k, lam=lam, boot=boot, seed=seed)
    verdict = ("PASS (>=0.35) -> build fusion head (Task 8b)" if pc_fused >= FUSION_GATE else
               "DEAD (<=0.25)" if pc_fused <= FUSION_DEAD else
               "HOLD (0.25-0.35)")
    return dict(fused_pcorr=pc_fused, fused_ci=[lo_fused, hi_fused], fused_res_corr=rc_fused,
                disc_alone_pcorr=pc_disc, disc_alone_ci=[lo_disc, hi_disc],
                disc_alone_res_corr=rc_disc, delta_vs_disc_alone=pc_fused - pc_disc, verdict=verdict)


# --------------------------------------------------------------------------- PROBE B (P-C1 ordinal)
def coral_fit_predict(Xtr, ytr, Xte, edges, C=0.05, max_iter=2000):
    """Simplified CORAL ordinal readout: K-1 shared-feature binary "y_bin >= k" logistic classifiers
    (Cao et al. 2020 style, no rank-consistency weight-sharing -- just independent thresholds, which
    is enough to test whether an ordinal decode beats plain ridge here), decoded to a continuous
    estimate as the bin-probability-weighted mean of bin midpoints. `edges` is a FIXED grid (not
    fitted from data) so there is no leakage from choosing bin boundaries."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    K = len(edges) - 1
    bin_tr = np.clip(np.digitize(ytr, edges[1:-1]), 0, K - 1)
    probs_ge = np.ones((Xte_s.shape[0], K))         # probs_ge[:, 0] == P(bin >= 0) == always 1
    for k_ in range(1, K):
        labels = (bin_tr >= k_).astype(int)
        if labels.min() == labels.max():            # degenerate fold: no eyes on one side of this cut
            probs_ge[:, k_] = float(labels[0])
            continue
        clf = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')
        clf.fit(Xtr_s, labels)
        col = list(clf.classes_).index(1)
        probs_ge[:, k_] = clf.predict_proba(Xte_s)[:, col]
    bin_probs = np.zeros((Xte_s.shape[0], K))
    bin_probs[:, 0] = 1 - probs_ge[:, 1]
    for j in range(1, K - 1):
        bin_probs[:, j] = probs_ge[:, j] - probs_ge[:, j + 1]
    bin_probs[:, K - 1] = probs_ge[:, K - 1]
    bin_probs = np.clip(bin_probs, 0, None)          # independent thresholds aren't guaranteed monotone
    row_sums = bin_probs.sum(1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    bin_probs /= row_sums
    mids = (edges[:-1] + edges[1:]) / 2
    return bin_probs @ mids


def ordinal_severity_probe(disc_path=DISC224, n_bins=10, lo=0.0, hi=30.0, lam=50.0, C=0.05):
    """Pooled leak-free OOF comparison, plain ridge vs CORAL ordinal, on the disc@224 `sev` ->
    per-eye md severity target, over the same 5 GRAPE folds. `lo`/`hi` are a FIXED dB grid spanning
    the plausible 24-2 mean-sensitivity range (not fitted per-fold), so bin boundaries never leak."""
    from scipy.stats import spearmanr
    d = np.load(disc_path)
    sev, md, fold = d['sev'], d['md'], d['fold']
    edges = np.linspace(lo, hi, n_bins + 1)
    pred_ridge = np.empty_like(md, dtype=np.float64)
    pred_coral = np.empty_like(md, dtype=np.float64)
    for f in range(int(fold.max()) + 1):
        te = fold == f
        pred_ridge[te] = D1.ridge_fit_predict(sev[~te], md[~te], sev[te], lam)
        pred_coral[te] = coral_fit_predict(sev[~te], md[~te], sev[te], edges, C=C)
    corr_ridge = float(np.corrcoef(pred_ridge, md)[0, 1])
    corr_coral = float(np.corrcoef(pred_coral, md)[0, 1])
    spear_ridge = float(spearmanr(pred_ridge, md).statistic)
    spear_coral = float(spearmanr(pred_coral, md).statistic)
    mae_ridge = float(np.abs(pred_ridge - md).mean())
    mae_coral = float(np.abs(pred_coral - md).mean())
    delta_corr = corr_coral - corr_ridge
    delta_spear = spear_coral - spear_ridge
    best_delta = max(delta_corr, delta_spear)
    verdict = "PASS (>=+0.02) -> carry to Task 11" if best_delta >= ORDINAL_GATE else "DROP"
    return dict(corr_ridge=corr_ridge, corr_coral=corr_coral, delta_corr=delta_corr,
                spear_ridge=spear_ridge, spear_coral=spear_coral, delta_spear=delta_spear,
                mae_ridge=mae_ridge, mae_coral=mae_coral, verdict=verdict)


def main():
    print("=" * 92)
    print("PROBE A (Task 5 / P-B3) -- disc@224 + full@224 spatial-descriptor fusion")
    print("-" * 92)
    a = fusion_spatial_probe()
    print(f"  disc@224 alone    spatial pcorr = {a['disc_alone_pcorr']:.3f}  95% CI {a['disc_alone_ci']}")
    print(f"  fused disc+full   spatial pcorr = {a['fused_pcorr']:.3f}  95% CI {a['fused_ci']}"
          f"  (delta vs disc-alone {a['delta_vs_disc_alone']:+.3f})")
    print(f"  GATE (>=0.35 PASS / <=0.25 dead / else hold): {a['verdict']}")
    print("=" * 92)
    print("PROBE B (Task 6 / P-C1) -- ordinal (CORAL) vs plain ridge severity readout")
    print("-" * 92)
    b = ordinal_severity_probe()
    print(f"  ridge : sev_corr={b['corr_ridge']:.3f}  spearman={b['spear_ridge']:.3f}  MAE={b['mae_ridge']:.3f}")
    print(f"  CORAL : sev_corr={b['corr_coral']:.3f}  spearman={b['spear_coral']:.3f}  MAE={b['mae_coral']:.3f}")
    print(f"  delta corr={b['delta_corr']:+.3f}  delta spearman={b['delta_spear']:+.3f}")
    print(f"  GATE (best delta >= +0.02 PASS else DROP): {b['verdict']}")
    print("=" * 92)


if __name__ == "__main__":
    main()
