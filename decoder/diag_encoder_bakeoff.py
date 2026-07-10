"""Task A2 — frozen-feature encoder bake-off (the decision; NO retrain).

Extends diag_d1_learning_curve across swappable backbones (decoder/encoders.py). For each obtainable
encoder we cache pooled frozen features over the 631 GRAPE images (full view, the same records/folds
as D1), then run — identically for every encoder — the D1 SEVERITY learning curve and a leak-free
per-point SPATIAL partial-corr probe on our folds.

Two passes, ONE torch process at a time (16 GB box):
  python decoder/diag_encoder_bakeoff.py --cache                 # torch: cache every available encoder
  python decoder/diag_encoder_bakeoff.py --cache --only dinov2_l # torch: cache just one
  python decoder/diag_encoder_bakeoff.py --probe                 # numpy: curves + spatial + GATE

Gate (pre-committed, §A2 step 3): advance an encoder to Task A3 iff, vs the RETFound-MAE baseline
measured IN THIS RUN by the identical pipeline,
  sev_corr@505 exceeds it by >= +0.03   OR   spatial partial-corr exceeds it by >= +0.05.
If no encoder clears the gate -> stop Phase A, record "encoder swap does not help our data" -> Phase B.
(Historical anchor: D1's pre-norm RETFound-MAE probe = 0.738 sev_corr@505; the in-run baseline uses the
post-norm encode_prefix, so we gate on the in-run number and report both.)
"""
import os, sys, json, argparse, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")
ALL_ENCODERS = ["retfound_mae", "retfound_dinov2", "dinov2_l", "dinov3_l", "visionfm"]
POOL = 4   # patch grid adaptively pooled to POOLxPOOL for the (grid-agnostic) spatial descriptor


def npz_path(name):
    return os.path.join(AUTO, f"bakeoff_{name}.npz")


# --------------------------------------------------------------------------- PASS 1: cache (torch)
def cache_encoder(name):
    import torch, torch.nn.functional as F
    import encoders as EN, training as T
    from torch.utils.data import DataLoader
    enc = EN.load_encoder(name)
    dev = T.DEVICE
    enc = enc.to(dev)
    gh, gw = enc.grid
    SEV, SPAT, MD, VF52, LAT, PID, FOLD = [], [], [], [], [], [], []
    for f in range(5):
        ds = T.MultiImageDataset(os.path.join(CV_DIR, f"fold{f}_val.json"), T.FUNDUS_DIR,
                                 T.val_transform, mode='train')          # 1 full-view img/record
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        items = json.load(open(os.path.join(CV_DIR, f"fold{f}_val.json")))
        si = 0
        with torch.no_grad():
            for imgs, hvf, lat in loader:
                pre = enc.encode_prefix(imgs.to(dev))                      # (B, 1+gh*gw, D)
                cls = pre[:, 0, :]                                         # (B, D)
                patch = pre[:, 1:, :]                                      # (B, gh*gw, D)
                meanp = patch.mean(1)                                      # (B, D)
                grid = patch.transpose(1, 2).reshape(pre.shape[0], enc.dim, gh, gw)
                # adaptive_avg_pool2d on MPS requires divisible sizes (14->4 fails); pool on CPU (tiny).
                pooled = F.adaptive_avg_pool2d(grid.cpu(), POOL)           # (B, D, POOL, POOL)
                pooled = pooled.reshape(pre.shape[0], -1).to(dev)          # (B, D*POOL*POOL)
                sev = torch.cat([cls, meanp], 1).float().cpu().numpy()     # (B, 2D)
                spat = torch.cat([cls, pooled], 1).float().cpu().numpy()   # (B, D*(1+POOL^2))
                for b in range(sev.shape[0]):
                    is_od = str(lat[b]).startswith('OD')
                    vi = T.valid_indices_od if is_od else T.valid_indices_os
                    v = np.asarray(hvf[b]).reshape(-1)[vi].astype(np.float64)  # 52-vector (fixed order)
                    v = np.where(v < T.MASKED_VALUE_THRESHOLD, v, np.nan)      # per-record extra masks -> nan
                    SEV.append(sev[b]); SPAT.append(spat[b])
                    MD.append(np.nanmean(v)); VF52.append(v)
                    LAT.append('OD' if is_od else 'OS')
                    PID.append(int(items[si]['PatientID'])); FOLD.append(f)
                    si += 1
    np.savez_compressed(npz_path(name), sev=np.array(SEV), spat=np.array(SPAT), md=np.array(MD),
                        vf52=np.array(VF52), lat=np.array(LAT), pid=np.array(PID),
                        fold=np.array(FOLD), grid=np.array(enc.grid))
    print(f"[{name}] cached {len(SEV)} recs  sev={np.array(SEV).shape}  spat={np.array(SPAT).shape}"
          f"  grid={enc.grid} -> {os.path.basename(npz_path(name))}")
    del enc
    gc.collect()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cache_all(only=None):
    names = [only] if only else ALL_ENCODERS
    for name in names:
        if only is None and os.path.exists(npz_path(name)):
            print(f"[{name}] cached already ({os.path.basename(npz_path(name))}); skip")
            continue
        try:
            cache_encoder(name)
        except Exception as e:
            print(f"[{name}] UNAVAILABLE — skipped: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- PASS 2: probe (numpy)
def _pca_fit(Xtr, k):
    mu = Xtr.mean(0)
    U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    comp = Vt[:min(k, Vt.shape[0])]
    return mu, comp


def oof_sev_corr(X, md, fold, lam=50.0):
    """Pooled OOF: train ridge on the other 4 folds, predict each fold's val, pool, correlate."""
    import diag_d1_learning_curve as D1
    pred = np.empty_like(md, dtype=np.float64)
    for f in range(5):
        te = fold == f
        pred[te] = D1.ridge_fit_predict(X[~te], md[~te], X[te], lam)
    return float(np.corrcoef(pred, md)[0, 1]), float(np.abs(pred - md).mean())


def sev_curve_505(X, md, pid, fold, seed=0, lam=50.0):
    """D1-style resampled curve endpoint at n=505 (same estimator that produced the 0.738 anchor)."""
    import diag_d1_learning_curve as D1
    rng = np.random.default_rng(seed)
    corrs = []
    n = 505
    for f in range(5):
        te = fold == f
        pool_idx = np.where(~te)[0]
        if n > len(pool_idx):
            n_use = len(pool_idx)
        else:
            n_use = n
        pool_pids = np.unique(pid[pool_idx])
        for _ in range(20):
            perm = rng.permutation(pool_pids); chosen, cnt = [], 0
            for p in perm:
                rows = pool_idx[pid[pool_idx] == p]
                chosen.extend(rows.tolist()); cnt += len(rows)
                if cnt >= n_use:
                    break
            tr = np.array(chosen)
            pr = D1.ridge_fit_predict(X[tr], md[tr], X[te], lam)
            if md[te].std() > 1e-6:
                corrs.append(np.corrcoef(pr, md[te])[0, 1])
    return float(np.mean(corrs))


def spatial_pcorr(spat, vf52, lat, pid, fold, k=100, lam=30.0, boot=3000, seed=42):
    """Leak-free per-point spatial probe -> D2 template-partial. For each fold, PCA(spat|train)+multi-
    output ridge predicts the 52-vector; a per-laterality template is built from train targets only;
    partial-corr(pred_resid, true_resid | template) is pooled OOF with a patient-clustered bootstrap."""
    import d2_template_partial as D2
    eyes = []   # (pid, pr, tr, zr)
    for f in range(5):
        te = fold == f; trm = ~te
        mu, comp = _pca_fit(spat[trm], k)
        Ztr = (spat[trm] - mu) @ comp.T
        Zte = (spat[te] - mu) @ comp.T
        # standardize probe inputs, ridge (multi-output) on mean-imputed targets (fixed 24-2 mask)
        s = Ztr.std(0) + 1e-8
        Ztr = Ztr / s; Zte = Zte / s
        Ytr = vf52[trm].copy()
        colmean = np.nanmean(Ytr, axis=0)
        Ytr = np.where(np.isnan(Ytr), colmean, Ytr)
        A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
        ymu = Ytr.mean(0)
        B = np.linalg.solve(A, Ztr.T @ (Ytr - ymu))
        Pte = Zte @ B + ymu                                   # (n_val, 52) predicted
        # template from this fold's train targets (nan-preserving), per laterality
        templ = D2.build_template(list(vf52[trm]), list(lat[trm]))
        vp = list(Pte); vt = list(vf52[te])
        val_lat = list(lat[te]); val_pid = list(pid[te])
        for (pr, tr, zr), p in zip(D2.eye_triples(vp, vt, val_lat, templ), val_pid):
            eyes.append((int(p), pr, tr, zr))
    PR = np.concatenate([e[1] for e in eyes])
    TR = np.concatenate([e[2] for e in eyes])
    ZR = np.concatenate([e[3] for e in eyes])
    pcorr = D2.partial_corr(PR, TR, ZR)
    res_corr = float(np.corrcoef(PR, TR)[0, 1])
    pids = np.array([e[0] for e in eyes]); uniq = np.unique(pids)
    by = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed); bs = np.empty(boot)
    for b in range(boot):
        rows = np.concatenate([by[p] for p in rng.choice(uniq, uniq.size, replace=True)])
        bs[b] = D2.partial_corr(np.concatenate([eyes[i][1] for i in rows]),
                                np.concatenate([eyes[i][2] for i in rows]),
                                np.concatenate([eyes[i][3] for i in rows]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return pcorr, res_corr, float(lo), float(hi)


def probe():
    avail = [n for n in ALL_ENCODERS if os.path.exists(npz_path(n))]
    if "retfound_mae" not in avail:
        raise SystemExit("cache retfound_mae first (the baseline): --cache --only retfound_mae")
    print("=" * 96)
    print(f"ENCODER BAKE-OFF — frozen-feature probes over 631 GRAPE recs, our folds; encoders: {avail}")
    print("=" * 96)
    rows = {}
    for name in avail:
        d = np.load(npz_path(name))
        sev, spat, md = d['sev'], d['spat'], d['md']
        vf52, lat, pid, fold = d['vf52'], d['lat'], d['pid'], d['fold']
        sc_oof, mae_oof = oof_sev_corr(sev, md, fold)
        sc_505 = sev_curve_505(sev, md, pid, fold)
        pc, rc, plo, phi = spatial_pcorr(spat, vf52, lat, pid, fold)
        rows[name] = dict(sev_corr_oof=sc_oof, sev_mae_oof=mae_oof, sev_corr_505=sc_505,
                          spatial_pcorr=pc, spatial_res_corr=rc, spatial_ci=[plo, phi],
                          grid=[int(x) for x in d['grid']])
        print(f"\n[{name}]  grid={tuple(int(x) for x in d['grid'])}")
        print(f"    severity  : sev_corr(pooled OOF)={sc_oof:.3f}  sev_MAE={mae_oof:.3f}  "
              f"sev_corr@505(D1-style)={sc_505:.3f}")
        print(f"    spatial   : partial-corr(pred,true|template)={pc:.3f}  95% CI [{plo:.3f}, {phi:.3f}]"
              f"  (raw res_corr {rc:.3f})")

    base = rows["retfound_mae"]
    print("\n" + "=" * 96)
    print(f"GATE vs in-run RETFound-MAE baseline (sev_corr@505={base['sev_corr_505']:.3f}, "
          f"pooled OOF={base['sev_corr_oof']:.3f}, spatial pcorr={base['spatial_pcorr']:.3f}; "
          f"historical D1 anchor 0.738)")
    print("-" * 96)
    print(f"{'encoder':>16}{'Δsev@505':>11}{'Δsev_oof':>11}{'Δspatial':>11}   verdict")
    winners = []
    for name in avail:
        if name == "retfound_mae":
            continue
        dsev = rows[name]['sev_corr_505'] - base['sev_corr_505']
        dsev_oof = rows[name]['sev_corr_oof'] - base['sev_corr_oof']
        dspat = rows[name]['spatial_pcorr'] - base['spatial_pcorr']
        passed = (dsev >= 0.03) or (dspat >= 0.05)
        verdict = "ADVANCE to A3" if passed else "sub-gate"
        if passed:
            winners.append(name)
        print(f"{name:>16}{dsev:>+11.3f}{dsev_oof:>+11.3f}{dspat:>+11.3f}   {verdict}")
    print("-" * 96)
    if winners:
        print(f"GATE PASSED: {winners} -> Task A3 (integrate winner, fold-0 scout, gated 5-fold).")
    else:
        print("GATE FAILED for every encoder -> STOP Phase A ('encoder swap does not help our data');"
              " proceed to Phase B (data).")
    print("=" * 96)
    json.dump(rows, open(os.path.join(AUTO, "encoder_bakeoff.json"), "w"), indent=2, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true', help="pass 1: cache frozen features (torch, alone)")
    ap.add_argument('--only', default=None, help="restrict --cache to one encoder name")
    ap.add_argument('--probe', action='store_true', help="pass 2: curves + spatial + gate (numpy)")
    a = ap.parse_args()
    if a.cache:
        cache_all(a.only)
    if a.probe:
        probe()
    if not a.cache and not a.probe:
        ap.error("choose --cache and/or --probe")


if __name__ == "__main__":
    main()
