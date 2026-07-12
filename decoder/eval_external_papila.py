"""Task 13 -- external validation of the champion fundus->VF model's SEVERITY (mean-deviation)
prediction on PAPILA, a fully independent cohort (Spanish, different camera, never in training).

This answers the "single small dataset / no external validation" reviewer objection. It is NOT a
per-point-VF validation -- PAPILA has no 52-point field, only a scalar clinical MD -- so we collapse
the champion's 52-point prediction to its per-eye MEAN (the model's implicit severity estimate) and
compare that against PAPILA's true mean deviation.

Champion = p1disc: 5 GRAPE-only fold checkpoints (decoder/results/auto/p1disc_f{0..4}_best.pth),
each bundling the frozen RETFound encoder + the small disc-ROI decoder. GRAPE-only training means
ALL of PAPILA is a valid external test (never seen in training, in any fold).

FOV choice (documented per the task brief -- this is a real methodological decision, not an
oversight): the champion's disc-only view is a FIXED laterality-mirrored crop of a MACULA-centered
GRAPE photo (disc_crop_pil in training.py; DISC_CX_OD=0.78/DISC_CX_OS=0.22, DISC_CY=0.49,
DISC_HALF=0.27). PAPILA's FundusImages are already DISC-CENTERED (Kovalyk et al. 2022) -- applying
GRAPE's fixed macula-relative crop coordinates to a disc-centered photo would crop an arbitrary
off-center region, not the disc. So for PAPILA we bypass disc_crop_pil entirely and feed the FULL
PAPILA image through the model's decoder at 224x224 (T.val_transform / T.get_tta_transforms, the
SAME resize+normalize pipeline used everywhere else in this repo) -- this is the closest FOV match
to the champion's intended disc view, and matches the precedent already established for PAPILA in
diag_papila_severity.py. This FOV/domain gap (crop framing, camera, population) is part of what is
being tested, not something to paper over.

Severity-scale harmonization (see diag_papila_severity.py's docstring for the same note): GRAPE's
ground truth ("md" throughout this codebase, e.g. bakeoff_retfound_mae__disc224.npz) is per-eye
24-2 MEAN SENSITIVITY (~28 dB healthy, LOWER = worse -- GRAPE has no age-normative database, so
there is no literal clinical MD to fit against). PAPILA's `md` is genuine Humphrey 30-2 MEAN
DEVIATION (~0 dB healthy, MORE NEGATIVE = worse). Both increase with eye health, so no sign flip is
needed, but the scales differ. Pearson/Spearman r are scale-free (affine/monotonic invariant) so
they are unaffected by this and are the PRIMARY externally-meaningful numbers here. MAE is not
scale-free, so we fit ONE FIXED linear map (slope, intercept) from the champion's raw predicted
per-eye mean sensitivity to GRAPE's true per-eye mean sensitivity, using p1disc's own out-of-fold
(OOF) predictions over all 631 GRAPE eyes (5-fold, each eye scored only by the model that never saw
it in training) -- then apply that SAME fixed map to PAPILA's ensemble predictions. Because GRAPE's
target scale (~0-30 dB sensitivity) and PAPILA's target scale (~+2 to -30 dB deviation) are
genuinely different quantities, the resulting "mae_calibrated" is NOT expected to be a small,
flattering number -- it is reported honestly as the secondary, scale-sensitive metric it is; r is
the number that actually answers "does the fundus->severity map transfer to an unseen cohort."

Usage (ONE torch process, foreground; ~10-20 min on this box -- 5 checkpoint loads x
(164 PAPILA eyes + ~126 GRAPE-fold eyes) x 3 TTA views through frozen RETFound-ViT-L):
  python decoder/eval_external_papila.py
  python decoder/eval_external_papila.py --no-tta      # faster ablation, no rotation TTA
  python decoder/eval_external_papila.py --n-boot 500   # fewer bootstrap resamples (faster)

Outputs:
  decoder/results/auto/papila_external_eval.json   -- full numeric report (read by
                                                       tests_external_eval.py::test_report_schema)
  decoder/results/auto/papila_external_scatter.png  -- calibrated-pred vs true-MD scatter, by label
"""
import argparse
import gc
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D
import eval_ckpt as EC
import training as T

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
AUTO_DIR    = os.path.join(CURRENT_DIR, "results", "auto")
CV_LONG_DIR = os.path.join(CURRENT_DIR, "results", "cv_long")
PAPILA_JSON = os.path.join(REPO_ROOT, "data", "external", "papila_records.json")
CKPT_TMPL   = os.path.join(AUTO_DIR, "p1disc_f{f}_best.pth")

OUT_JSON = os.path.join(AUTO_DIR, "papila_external_eval.json")
OUT_PNG  = os.path.join(AUTO_DIR, "papila_external_scatter.png")

N_FOLDS  = 5
N_BOOT   = 2000
SEED     = 42
LABELS   = ("healthy", "suspect", "glaucoma")


# ============================================================== data
class PapilaFullImageDataset(Dataset):
    """One PAPILA eye -> a stack of TTA-augmented FULL-image (no crop) views.

    PAPILA images are already disc-centered (see module docstring for why the champion's
    macula-relative disc crop is NOT applied here); this mirrors T.MultiImageDataset's val-mode
    'full' view path exactly (T.get_tta_transforms() / T.val_transform), just against an absolute
    PAPILA image path instead of a GRAPE FUNDUS_DIR-relative one.
    """

    def __init__(self, records, use_tta=True):
        self.records = records
        self.use_tta = use_tta

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image"]).convert("RGB")
        if self.use_tta:
            views = [tfm(img) for tfm in T.get_tta_transforms()]
        else:
            views = [T.val_transform(img)]
        return torch.stack(views), r["Laterality"], r["eye_id"]


def _identity_collate(batch):
    return batch[0]


def papila_patient_id(eye_id):
    """'papila_RET002OD' -> 'RET002' (both eyes of a patient cluster together for bootstrap)."""
    m = re.match(r"papila_(RET\d+)", eye_id)
    assert m, f"unexpected PAPILA eye_id format: {eye_id!r}"
    return m.group(1)


# ============================================================== model pass (one fold)
@torch.no_grad()
def run_fold(f, papila_records, papila_pred_sum, use_tta):
    """Load fold f's p1disc checkpoint once and do BOTH jobs it's needed for:
      (a) GRAPE OOF eval on this fold's held-out val split (feeds the GRAPE calibration fit --
          reuses eval_ckpt.per_eye_preds unmodified, so it exercises the champion's real disc-ROI
          eval path exactly as already reported in p1disc_cv.json);
      (b) PAPILA full-image inference on all 164 eyes, accumulated into the running 5-fold sum
          (the ensemble average IS "the model" per the task brief).
    Returns this fold's (grape_preds, grape_trues) lists (per-eye 52-vectors, nan at masked pts).
    """
    ckpt_path = CKPT_TMPL.format(f=f)
    print(f"[fold {f}] loading {os.path.basename(ckpt_path)} ...")
    model = EC.load_model(ckpt_path)
    model.eval()

    val_json = os.path.join(CV_LONG_DIR, f"fold{f}_val.json")
    grape_preds, grape_trues = EC.per_eye_preds(model, val_json, use_tta=use_tta)
    print(f"[fold {f}] GRAPE OOF: {len(grape_preds)} held-out eyes scored")

    ds = PapilaFullImageDataset(papila_records, use_tta=use_tta)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_identity_collate)
    for imgs, lat, eye_id in loader:
        imgs = imgs.to(T.DEVICE)
        latent = model._encode(imgs)
        pred = model.decode_latent(latent, lat, average_multi=True).cpu().numpy()[0]   # (52,)
        papila_pred_sum[eye_id] += pred
    print(f"[fold {f}] PAPILA: {len(papila_records)} eyes scored")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return grape_preds, grape_trues


# ============================================================== calibration
def fit_grape_calibration(grape_preds, grape_trues):
    """slope, intercept such that intercept + slope*pred_mean_sensitivity ~= true GRAPE per-eye
    mean sensitivity ('GRAPE MD' by this codebase's convention), fit OLS over all 631 GRAPE eyes'
    OOF predictions pooled across the 5 folds. Also returns a sanity-check comparison against the
    previously-reported champion pooled stats (p1disc_cv.json: raw MAE 4.113, corr 0.684)."""
    pred_mean = np.array([np.nanmean(p) for p in grape_preds])
    true_mean = np.array([np.nanmean(t) for t in grape_trues])
    slope, intercept = np.polyfit(pred_mean, true_mean, 1)
    sev_corr = float(np.corrcoef(pred_mean, true_mean)[0, 1])
    sev_mae  = float(np.mean(np.abs(pred_mean - true_mean)))

    pooled = D.pooled_metrics(grape_preds, grape_trues)
    print(f"GRAPE OOF sanity check (should ~match p1disc_cv.json raw): "
          f"MAE {pooled['mae']:.3f} corr {pooled['corr']:.3f} slope {pooled['slope']:.3f} "
          f"(p1disc_cv.json: MAE 4.113 corr 0.684 slope 0.536)")
    # Fix (final code review): this must ASSERT, not just print -- a silent encoder/timm/eval-path
    # regression would otherwise still produce a "successful" external-validation report built on a
    # champion that no longer reproduces its own reported CV number.
    _p1disc_cv_path = os.path.join(AUTO_DIR, "p1disc_cv.json")
    with open(_p1disc_cv_path) as _f:
        _stored_raw = json.load(_f)["raw"]
    _mae_tol = 0.01
    _mae_delta = abs(pooled["mae"] - _stored_raw["mae"])
    assert _mae_delta < _mae_tol, (
        f"GRAPE OOF reproduction check FAILED: live pooled MAE {pooled['mae']:.4f} vs stored "
        f"{_p1disc_cv_path} raw MAE {_stored_raw['mae']:.4f} (|Δ|={_mae_delta:.4f} >= tol "
        f"{_mae_tol}) -- the champion checkpoints no longer reproduce their reported CV number "
        f"(encoder/timm/eval-path regression?); fix that before trusting this external-validation "
        f"run.")
    print(f"GRAPE per-eye severity calibration: slope={slope:.4f} intercept={intercept:.3f} "
          f"| sev_corr={sev_corr:.3f} sev_mae={sev_mae:.3f} (n={len(pred_mean)})")
    return dict(slope=float(slope), intercept=float(intercept), n_eyes=int(len(pred_mean)),
                sev_corr=sev_corr, sev_mae=sev_mae,
                sanity_pooled_mae=pooled["mae"], sanity_pooled_corr=pooled["corr"],
                sanity_pooled_slope=pooled["slope"])


# ============================================================== metrics
def cluster_bootstrap(idx, patient_ids, pred_raw, true_md, slope, intercept,
                       n_boot=N_BOOT, seed=SEED):
    """Patient-clustered bootstrap over a subset `idx` of eyes. Resamples unique patients (both of
    a patient's eyes move together) with replacement `n_boot` times; returns (mean, lo, hi) 95% CI
    for Pearson r, Spearman r, and calibrated MAE."""
    idx = np.asarray(idx)
    pid_sub = patient_ids[idx]
    uniq = np.unique(pid_sub)
    by = {p: idx[pid_sub == p] for p in uniq}
    rng = np.random.default_rng(seed)

    rs, rhos, maes = [], [], []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([by[p] for p in chosen])
        x, y = pred_raw[rows], true_md[rows]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        rs.append(pearsonr(x, y)[0])
        rhos.append(spearmanr(x, y).statistic)
        cal = intercept + slope * x
        maes.append(float(np.mean(np.abs(cal - y))))

    def ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(np.mean(vals)), float(lo), float(hi)

    r_mean, r_lo, r_hi = ci(rs)
    rho_mean, rho_lo, rho_hi = ci(rhos)
    mae_mean, mae_lo, mae_hi = ci(maes)
    return dict(r_pearson_boot=r_mean, r_pearson_ci=[r_lo, r_hi],
                r_spearman_boot=rho_mean, r_spearman_ci=[rho_lo, rho_hi],
                mae_calibrated_boot=mae_mean, mae_calibrated_ci=[mae_lo, mae_hi])


def subset_report(idx, patient_ids, pred_raw, true_md, slope, intercept, n_boot=N_BOOT):
    idx = np.asarray(idx)
    x, y = pred_raw[idx], true_md[idx]
    r_p = float(pearsonr(x, y)[0]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else float("nan")
    r_s = float(spearmanr(x, y).statistic) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else float("nan")
    cal = intercept + slope * x
    mae_raw = float(np.mean(np.abs(x - y)))
    mae_cal = float(np.mean(np.abs(cal - y)))
    boot = cluster_bootstrap(idx, patient_ids, pred_raw, true_md, slope, intercept, n_boot=n_boot)
    return dict(n_eyes=int(len(idx)), n_patients=int(len(np.unique(patient_ids[idx]))),
                r_pearson=r_p, r_spearman=r_s, mae_raw=mae_raw, mae_calibrated=mae_cal,
                r_pearson_ci=boot["r_pearson_ci"], r_spearman_ci=boot["r_spearman_ci"],
                mae_calibrated_ci=boot["mae_calibrated_ci"])


# ============================================================== plotting
def save_scatter(pred_raw, true_md, labels, slope, intercept, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = intercept + slope * pred_raw
    colors = {"healthy": "#2e7d32", "suspect": "#ed6c02", "glaucoma": "#c62828"}
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for lab in LABELS:
        m = labels == lab
        if m.any():
            ax.scatter(cal[m], true_md[m], s=22, alpha=0.75, color=colors[lab],
                       edgecolors="white", linewidths=0.4, label=f"{lab} (n={m.sum()})")
    lo = min(cal.min(), true_md.min()) - 1
    hi = max(cal.max(), true_md.max()) + 1
    sl2, ic2 = np.polyfit(cal, true_md, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, sl2 * xs + ic2, "--", color="#444", lw=1.4,
            label=f"PAPILA best fit: y={sl2:.2f}x+{ic2:.1f}")
    r_p = pearsonr(cal, true_md)[0]
    r_s = spearmanr(cal, true_md).statistic
    ax.set_xlabel("Champion predicted severity, GRAPE-calibrated (dB, sensitivity scale)", fontsize=10)
    ax.set_ylabel("PAPILA true mean deviation (dB)", fontsize=10)
    ax.set_title("External validation: fundus→severity on PAPILA (independent cohort)\n"
                 f"Pearson r={r_p:.3f}  Spearman ρ={r_s:.3f}  (n={len(true_md)} eyes)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.15)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    use_tta = not a.no_tta

    papila_records = json.load(open(PAPILA_JSON))
    papila_pred_sum = {r["eye_id"]: np.zeros(T.NUM_VALID_POINTS, dtype=np.float64) for r in papila_records}

    all_grape_preds, all_grape_trues = [], []
    for f in range(N_FOLDS):
        gp, gt = run_fold(f, papila_records, papila_pred_sum, use_tta)
        all_grape_preds += gp
        all_grape_trues += gt

    calib = fit_grape_calibration(all_grape_preds, all_grape_trues)
    slope, intercept = calib["slope"], calib["intercept"]

    eye_ids   = [r["eye_id"] for r in papila_records]
    true_md   = np.array([r["md"] for r in papila_records], dtype=np.float64)
    labels    = np.array([r["label"] for r in papila_records])
    patient_ids = np.array([papila_patient_id(e) for e in eye_ids])
    # ensemble = mean over the 5 folds' 52-point predictions, per eye (task-13 brief step 1);
    # eye's predicted severity = mean of its (ensembled) 52 predicted sensitivities.
    pred_raw = np.array([papila_pred_sum[e].mean() / N_FOLDS for e in eye_ids], dtype=np.float64)

    overall = subset_report(np.arange(len(eye_ids)), patient_ids, pred_raw, true_md,
                            slope, intercept, n_boot=a.n_boot)
    by_label = {}
    for lab in LABELS:
        idx = np.where(labels == lab)[0]
        by_label[lab] = subset_report(idx, patient_ids, pred_raw, true_md, slope, intercept,
                                      n_boot=a.n_boot) if len(idx) else dict(n_eyes=0)

    report = dict(
        method=dict(
            champion="p1disc (5 GRAPE-only fold checkpoints, ensembled by averaging 52-pt preds)",
            fov="FULL PAPILA image (already disc-centered) through the model's non-crop decode "
                "path -- GRAPE's fixed macula-relative disc-crop coords are NOT applied (see "
                "module docstring)",
            tta=use_tta, n_folds=N_FOLDS, n_boot=a.n_boot,
            grape_severity_convention="GRAPE 'md' = per-eye 24-2 mean sensitivity (higher=healthier); "
                                      "PAPILA 'md' = true Humphrey 30-2 mean deviation (higher=healthier, "
                                      "different scale/zero-point) -- Pearson/Spearman r are scale-free "
                                      "and are the primary numbers; MAE is calibrated but not scale-matched.",
        ),
        grape_calibration=calib,
        overall=overall,
        by_label=by_label,
    )
    json.dump(report, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}")

    save_scatter(pred_raw, true_md, labels, slope, intercept, OUT_PNG)

    print("\n" + "=" * 90)
    print("PAPILA EXTERNAL VALIDATION -- fundus->severity (p1disc 5-fold ensemble)")
    print("=" * 90)
    o = overall
    print(f"OVERALL   n={o['n_eyes']} ({o['n_patients']} patients) | "
          f"Pearson r={o['r_pearson']:.3f} CI{np.round(o['r_pearson_ci'],3).tolist()} | "
          f"Spearman r={o['r_spearman']:.3f} CI{np.round(o['r_spearman_ci'],3).tolist()} | "
          f"MAE raw={o['mae_raw']:.2f} calibrated={o['mae_calibrated']:.2f} "
          f"CI{np.round(o['mae_calibrated_ci'],2).tolist()}")
    for lab in LABELS:
        r = by_label[lab]
        if r.get("n_eyes"):
            print(f"  {lab:9s} n={r['n_eyes']:3d} | Pearson r={r['r_pearson']:.3f} | "
                  f"Spearman r={r['r_spearman']:.3f} | MAE cal={r['mae_calibrated']:.2f}")
        else:
            print(f"  {lab:9s} n=0")
    print("=" * 90)


if __name__ == "__main__":
    main()
