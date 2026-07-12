"""Task 3 (P-A2) -- PAPILA severity-transfer probe (sub4 design, Phase B lever).

Does adding PAPILA eyes to the frozen-feature -> per-eye severity (MD) ridge raise GRAPE pooled-OOF
sev_corr, evaluated ONLY on GRAPE (fold-disjoint; PAPILA is TRAIN-ONLY, never inserted into a
held-out GRAPE-val fold)?

Severity-target harmonization: GRAPE `md` (decoder/results/auto/bakeoff_retfound_mae__disc224.npz)
is per-eye 24-2 MEAN SENSITIVITY (~28 dB healthy, LOWER = worse). PAPILA `md`
(data/external/papila_records.json) is per-eye 30-2 MEAN DEVIATION (~0 dB healthy, MORE NEGATIVE =
worse). Different scale AND zero-point, but BOTH increase with eye health -- no sign flip needed,
only a per-dataset z-score to put the two targets on one shared severity axis (sev_corr is
scale-free, so the OOF gate is robust once both sides are z-scored; z-scoring a ridge target is
also an affine transform, which per-condition leaves the GRAPE-only baseline's sev_corr unchanged
-- it only matters once PAPILA rows are mixed into the same fit).

Two passes, ONE torch process at a time (16 GB box):
  python decoder/diag_papila_severity.py --cache   # torch: cache 164 PAPILA 2048-d sev features
  python decoder/diag_papila_severity.py --probe   # numpy: fold-wise ridge + gate + fallbacks

GATE (task-3 brief, pre-committed): GRAPE pooled-OOF sev_corr rises by >= +0.02 when PAPILA (all 164
eyes, z-scored MD) is added to every fold's GRAPE-train ridge fit (GRAPE disc224 features -- matches
the champion's disc-crop view) -> PASS, carry to Task 10. Else try fallback (a) GRAPE FULL features
(bakeoff_retfound_mae.npz, whole-fundus view) + PAPILA, fallback (b) per-dataset feature-mean
centering before the ridge (removes any encoder-domain-shift offset between the GRAPE and PAPILA
feature clouds that a single joint standardization inside ridge_fit_predict would not remove). If
still < +0.02 after both fallbacks -> "PAPILA severity transfer does not hold (domain shift)" ->
skip Task 10 (PAPILA remains usable for external validation, Task 13).
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GRAPE_DISC224 = os.path.join(AUTO, "bakeoff_retfound_mae__disc224.npz")   # primary: disc view (champion match)
GRAPE_FULL = os.path.join(AUTO, "bakeoff_retfound_mae.npz")               # fallback (a): whole-fundus view
PAPILA_JSON = os.path.join(REPO_ROOT, "data", "external", "papila_records.json")
PAPILA_FEAT = os.path.join(AUTO, "papila_sev_features.npz")               # gitignored (see .gitignore)

GATE_THRESHOLD = 0.02


# --------------------------------------------------------------------------- PASS 1: cache (torch)
def cache_papila_features():
    """Frozen RETFound-MAE CLS+mean-patch descriptor (2048-d) over all 164 harmonized PAPILA eyes
    -- mirrors diag_encoder_bakeoff.cache_encoder's `sev` construction exactly (cat([CLS,
    mean(patch_tokens)], dim=1)). PAPILA images are ALREADY disc-centered, so we use the FULL
    PAPILA image at 224 via the same T.val_transform (Resize 224 + ImageNet norm) as every other
    frozen-feature cache in this repo -- NOT GRAPE's fixed macula-relative disc-crop, which would
    crop the wrong region on an already-disc-centered photo. This is the closest FOV match to
    GRAPE's disc224 champion view."""
    import torch
    import encoders as EN, training as T
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    records = json.load(open(PAPILA_JSON))

    class _PapilaDataset(Dataset):
        def __init__(self, records, tfm):
            self.records = records
            self.tfm = tfm

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            img = Image.open(r["image"]).convert("RGB")
            return self.tfm(img), float(r["md"]), r["eye_id"], r["Laterality"]

    enc = EN.load_encoder("retfound_mae").to(T.DEVICE)
    ds = _PapilaDataset(records, T.val_transform)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    SEV, MD, EYEID, LAT = [], [], [], []
    with torch.no_grad():
        for imgs, mds, eids, lats in loader:
            pre = enc.encode_prefix(imgs.to(T.DEVICE), input_size=224)   # (B, 1+196, 1024)
            cls = pre[:, 0, :]
            meanp = pre[:, 1:, :].mean(1)
            sev = torch.cat([cls, meanp], 1).float().cpu().numpy()       # (B, 2048)
            for b in range(sev.shape[0]):
                SEV.append(sev[b]); MD.append(float(mds[b]))
                EYEID.append(eids[b]); LAT.append(lats[b])
    np.savez_compressed(PAPILA_FEAT, sev=np.array(SEV), md=np.array(MD),
                        eye_id=np.array(EYEID), lat=np.array(LAT))
    print(f"cached {len(SEV)} PAPILA features (2048-d) -> {os.path.basename(PAPILA_FEAT)}")


# --------------------------------------------------------------------------- PASS 2: probe (numpy)
def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


def pooled_oof(sev_g, md_g, fold_g, extra_X=None, extra_y=None, lam=50.0, center_separately=False):
    """Pooled OOF predictions for GRAPE: for each fold, fit ridge on GRAPE-train (+ optional extra
    rows from PAPILA -- TRAIN-ONLY, never in a held-out GRAPE-val fold), predict that fold's
    GRAPE-val. center_separately (fallback b): subtract each dataset's OWN feature mean (GRAPE-train's,
    PAPILA's) before concatenating -- guards against an encoder-domain-shift offset between the two
    feature clouds that ridge_fit_predict's single joint standardization would not remove."""
    import diag_d1_learning_curve as D1
    pred = np.empty_like(md_g, dtype=np.float64)
    for f in range(5):
        te = fold_g == f
        Xtr, ytr, Xte = sev_g[~te], md_g[~te], sev_g[te]
        if extra_X is not None:
            if center_separately:
                mu_g, mu_p = Xtr.mean(0), extra_X.mean(0)
                Xtr, Xte = Xtr - mu_g, Xte - mu_g
                Xp = extra_X - mu_p
            else:
                Xp = extra_X
            Xtr = np.concatenate([Xtr, Xp], axis=0)
            ytr = np.concatenate([ytr, extra_y], axis=0)
        pred[te] = D1.ridge_fit_predict(Xtr, ytr, Xte, lam)
    return pred


def sev_corr_mae(pred, true):
    return float(np.corrcoef(pred, true)[0, 1]), float(np.abs(pred - true).mean())


def bootstrap_delta(pred_base, pred_plus, md_g, pid_g, boot=2000, seed=42):
    """Patient-clustered bootstrap of Delta(sev_corr) = corr(+PAPILA) - corr(baseline), resampling
    whole GRAPE patients (both fellow eyes move together) with replacement."""
    uniq = np.unique(pid_g)
    by = {p: np.where(pid_g == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    deltas = np.empty(boot)
    for b in range(boot):
        rows = np.concatenate([by[p] for p in rng.choice(uniq, uniq.size, replace=True)])
        c_base = np.corrcoef(pred_base[rows], md_g[rows])[0, 1]
        c_plus = np.corrcoef(pred_plus[rows], md_g[rows])[0, 1]
        deltas[b] = c_plus - c_base
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def run_condition(label, grape_npz, sev_p, md_p, lam=50.0, boot=2000, center_separately=False):
    """One full (baseline vs +PAPILA) comparison for a given GRAPE feature cache."""
    d = np.load(grape_npz)
    sev_g, pid_g, fold_g = d["sev"], d["pid"], d["fold"]
    md_g = zscore(d["md"])
    pred_base = pooled_oof(sev_g, md_g, fold_g, lam=lam)
    pred_plus = pooled_oof(sev_g, md_g, fold_g, extra_X=sev_p, extra_y=md_p, lam=lam,
                           center_separately=center_separately)
    corr_base, mae_base = sev_corr_mae(pred_base, md_g)
    corr_plus, mae_plus = sev_corr_mae(pred_plus, md_g)
    delta = corr_plus - corr_base
    d_mean, d_lo, d_hi = bootstrap_delta(pred_base, pred_plus, md_g, pid_g, boot=boot)
    return dict(label=label, n_grape=int(sev_g.shape[0]), n_papila=int(sev_p.shape[0]),
               corr_base=corr_base, mae_base=mae_base, corr_plus=corr_plus, mae_plus=mae_plus,
               delta=delta, delta_boot_mean=d_mean, delta_ci_lo=d_lo, delta_ci_hi=d_hi)


def _print_condition(r):
    print(f"\n[{r['label']}]  (n_grape={r['n_grape']}, n_papila={r['n_papila']})")
    print(f"    baseline (GRAPE-only) : sev_corr={r['corr_base']:.3f}  sev_MAE(z)={r['mae_base']:.3f}")
    print(f"    +PAPILA               : sev_corr={r['corr_plus']:.3f}  sev_MAE(z)={r['mae_plus']:.3f}")
    print(f"    Delta sev_corr = {r['delta']:+.3f}   patient-bootstrap mean {r['delta_boot_mean']:+.3f}, "
          f"95% CI [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}]")


def probe(lam=50.0, boot=2000):
    if not os.path.exists(PAPILA_FEAT):
        raise SystemExit("cache PAPILA features first: python decoder/diag_papila_severity.py --cache")
    dp = np.load(PAPILA_FEAT, allow_pickle=True)
    sev_p, md_p = dp["sev"], zscore(dp["md"])

    print("=" * 96)
    print("P-A2 -- PAPILA severity-transfer probe (164 PAPILA eyes, z-scored MD, TRAIN-ONLY)")
    print("=" * 96)

    results = {}
    primary = run_condition("primary: GRAPE disc224 + PAPILA", GRAPE_DISC224, sev_p, md_p, lam, boot)
    results["primary"] = primary
    _print_condition(primary)

    if primary["delta"] >= GATE_THRESHOLD:
        verdict = f"PASS ({primary['label']}) -- carry to Task 10"
    else:
        fb_a = run_condition("fallback (a): GRAPE FULL + PAPILA", GRAPE_FULL, sev_p, md_p, lam, boot)
        results["fallback_a"] = fb_a
        _print_condition(fb_a)

        fb_b = run_condition("fallback (b): disc224 + PAPILA, per-dataset centering", GRAPE_DISC224,
                             sev_p, md_p, lam, boot, center_separately=True)
        results["fallback_b"] = fb_b
        _print_condition(fb_b)

        best = max([primary, fb_a, fb_b], key=lambda r: r["delta"])
        if best["delta"] >= GATE_THRESHOLD:
            verdict = f"PASS via {best['label']} -- carry to Task 10"
        else:
            verdict = ("FAIL -- PAPILA severity transfer does not hold (domain shift); skip Task 10 "
                      "(PAPILA remains usable for external validation, Task 13)")

    print("-" * 96)
    print(f"GATE (>= +{GATE_THRESHOLD:.2f} Delta sev_corr vs GRAPE-only baseline): {verdict}")
    print("=" * 96)
    results["gate_threshold"] = GATE_THRESHOLD
    results["verdict"] = verdict
    json.dump(results, open(os.path.join(AUTO, "diag_papila_severity.json"), "w"), indent=2, default=float)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="pass 1: cache PAPILA frozen features (torch, alone)")
    ap.add_argument("--probe", action="store_true", help="pass 2: gate + fallbacks (numpy)")
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.cache:
        cache_papila_features()
    if a.probe:
        probe(lam=a.lam, boot=a.boot)
    if not a.cache and not a.probe:
        ap.error("choose --cache and/or --probe")


if __name__ == "__main__":
    main()
