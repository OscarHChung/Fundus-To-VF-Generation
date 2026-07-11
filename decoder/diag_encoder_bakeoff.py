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
import os, sys, json, argparse, gc, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")
ALL_ENCODERS = ["retfound_mae", "retfound_dinov2", "dinov2_l", "dinov3_l", "visionfm"]
POOL = 4   # patch grid adaptively pooled to POOLxPOOL for the (grid-agnostic) spatial descriptor


def npz_path(name, view="full", input_size=224, disc_center="fixed"):
    """Default config (full@224, fixed box) resolves to the EXACT existing filename — this is the
    already-cached baseline anchor (bakeoff_{name}.npz) and must never be renamed/shadowed. Any
    other config (high-res, disc view, detected center) gets its own distinct file so it can sit
    alongside the baseline without ever overwriting it."""
    if view == "full" and input_size == 224 and disc_center == "fixed":
        return os.path.join(AUTO, f"bakeoff_{name}.npz")
    det = "_det" if disc_center == "detected" else ""
    return os.path.join(AUTO, f"bakeoff_{name}__{view}{input_size}{det}.npz")


def _detect_disc_center(img, laterality):
    """Dependency-free optic-disc localization: the green-channel brightness centroid within the
    eye's laterality quadrant (right half of the image for OD, left half for OS; top half, since
    the fixed nominal DISC_CY≈0.49 sits in the image's upper half). No learned detector — plain
    numpy on the PIL image, mirroring how disc_crop_pil's FIXED box is itself laterality-aware.

    Returns (cx, cy) as FRACTIONS of (W, H), meant to be diffed against the fixed nominal center
    and fed to training.disc_crop_pil as cx_off/cy_off. Degenerate case (all-dark quadrant) falls
    back to the fixed nominal center (cx_off=cy_off=0 -> identical to the fixed-box crop)."""
    import training as T
    is_od = str(laterality).startswith('OD')
    fixed_cx = T.DISC_CX_OD if is_od else T.DISC_CX_OS
    fixed_cy = T.DISC_CY
    arr = np.asarray(img.convert('RGB'))
    h, w = arr.shape[0], arr.shape[1]
    x0, x1 = (w // 2, w) if is_od else (0, w // 2)
    y0, y1 = 0, h // 2                                    # DISC_CY ~0.49 -> top-half quadrant
    region = arr[y0:y1, x0:x1, 1].astype(np.float64)      # green channel (brightest for the disc)
    wsum = region.sum()
    if wsum <= 0:
        return fixed_cx, fixed_cy
    ys, xs = np.mgrid[0:region.shape[0], 0:region.shape[1]]
    cy_local = float((ys * region).sum() / wsum)
    cx_local = float((xs * region).sum() / wsum)
    return (x0 + cx_local) / w, (y0 + cy_local) / h


def _discover_configs():
    """Extra caches beyond the plain per-encoder baseline files (which have no '__' in the name) —
    any bakeoff_*__*.npz, labeled by its filename stem. Lets probe() include high-res/disc/detected
    caches produced by cache_encoder(view=..., input_size=..., disc_center=...) without touching
    the ALL_ENCODERS baseline scan at all."""
    paths = sorted(glob.glob(os.path.join(AUTO, "bakeoff_*__*.npz")))
    return [(os.path.basename(p)[len("bakeoff_"):-len(".npz")], p) for p in paths]


# --------------------------------------------------------------------------- PASS 1: cache (torch)
def cache_encoder(name, view="full", input_size=224, disc_center="fixed"):
    """Cache pooled frozen features for one (encoder, view, input_size, disc_center) config.

    Defaults reproduce today's behavior byte-for-byte: view='full' -> T.val_transform (Resize
    224) via the unmodified T.MultiImageDataset(mode='train') path, encode_prefix(input_size=224)
    takes the untouched 224 branch. Non-default configs: view='disc' applies the laterality-aware
    disc crop (training.disc_crop_pil) BEFORE resizing to input_size; disc_center='detected' crops
    around _detect_disc_center's green-channel centroid instead of the fixed box; input_size != 224
    resizes to that size and drives encode_prefix's high-res (RETFound-MAE) path.
    """
    if view not in ("full", "disc"):
        raise ValueError(f"unknown view {view!r}; choose 'full' or 'disc'")
    if disc_center not in ("fixed", "detected"):
        raise ValueError(f"unknown disc_center {disc_center!r}; choose 'fixed' or 'detected'")
    import torch, torch.nn.functional as F
    import encoders as EN, training as T
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    enc = EN.load_encoder(name)
    dev = T.DEVICE
    enc = enc.to(dev)

    is_default = (view == "full" and input_size == 224 and disc_center == "fixed")
    tfm = T.val_transform if is_default else transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class _DetectedDiscDataset(Dataset):
        """Disc crop centered on the DETECTED optic-nerve-head rather than the fixed box. One
        full-view image per record (matches the CV fold jsons: FundusImage is a 1-element list),
        so samples line up 1:1 with the fold json's record order — the same assumption the
        'full' and fixed-disc paths already rely on (via T.MultiImageDataset's mode='train')."""
        def __init__(self, json_path, fundus_dir):
            with open(json_path, 'r') as jf:
                self.data = json.load(jf)
            self.fundus_dir = fundus_dir

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            img_path = item['FundusImage']
            if isinstance(img_path, list):
                img_path = img_path[0]
            laterality = item.get('Laterality', 'OD').strip().upper()
            img = Image.open(os.path.join(self.fundus_dir, img_path)).convert('RGB')
            cx, cy = _detect_disc_center(img, laterality)
            fixed_cx = T.DISC_CX_OD if laterality.startswith('OD') else T.DISC_CX_OS
            img = T.disc_crop_pil(img, laterality, cx_off=cx - fixed_cx, cy_off=cy - T.DISC_CY)
            hvf = np.array(item['hvf'], dtype=np.float32).flatten()
            hvf_tensor = torch.tensor(hvf)
            # Match T.MultiImageDataset(mode='train')'s label-noise injection exactly (training.py
            # ~L400) so the 'detected' config's cached targets are noise-matched to the 'full' and
            # fixed-disc configs (both go through T.MultiImageDataset) — otherwise 'detected' would
            # get artificially noise-free targets and look spuriously better/worse in probe().
            if T.LABEL_NOISE_STD > 0:
                noise = torch.randn_like(hvf_tensor) * T.LABEL_NOISE_STD
                valid_mask = hvf_tensor < T.MASKED_VALUE_THRESHOLD
                hvf_tensor = hvf_tensor + noise * valid_mask.float()
                hvf_tensor = torch.clamp(hvf_tensor, 0.0, 35.0) * valid_mask.float() + \
                             hvf_tensor * (~valid_mask).float()
            return tfm(img), hvf_tensor, laterality

    SEV, SPAT, MD, VF52, LAT, PID, FOLD = [], [], [], [], [], [], []
    gh = gw = None
    for f in range(5):
        fold_json = os.path.join(CV_DIR, f"fold{f}_val.json")
        if view == "full":
            ds = T.MultiImageDataset(fold_json, T.FUNDUS_DIR, tfm, mode='train')       # 1 img/record
        elif disc_center == "fixed":
            ds = T.MultiImageDataset(fold_json, T.FUNDUS_DIR, tfm, mode='train', disc_only=True)
        else:
            ds = _DetectedDiscDataset(fold_json, T.FUNDUS_DIR)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        items = json.load(open(fold_json))
        si = 0
        with torch.no_grad():
            for imgs, hvf, lat in loader:
                pre = enc.encode_prefix(imgs.to(dev), input_size=input_size)  # (B, 1+gh*gw, D)
                n_tok = pre.shape[1] - 1
                gh = gw = int(round(n_tok ** 0.5))          # recomputed from the actual output —
                cls = pre[:, 0, :]                          # robust to input_size, not enc.grid
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
    out_path = npz_path(name, view, input_size, disc_center)
    np.savez_compressed(out_path, sev=np.array(SEV), spat=np.array(SPAT), md=np.array(MD),
                        vf52=np.array(VF52), lat=np.array(LAT), pid=np.array(PID),
                        fold=np.array(FOLD), grid=np.array([gh, gw]))
    print(f"[{name}] view={view} input_size={input_size} disc_center={disc_center} cached "
          f"{len(SEV)} recs  sev={np.array(SEV).shape}  spat={np.array(SPAT).shape}"
          f"  grid={(gh, gw)} -> {os.path.basename(out_path)}")
    del enc
    gc.collect()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cache_all(only=None, view="full", input_size=224, disc_center="fixed"):
    names = [only] if only else ALL_ENCODERS
    for name in names:
        p = npz_path(name, view, input_size, disc_center)
        if only is None and os.path.exists(p):
            print(f"[{name}] cached already ({os.path.basename(p)}); skip")
            continue
        try:
            cache_encoder(name, view=view, input_size=input_size, disc_center=disc_center)
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
    baseline_names = [n for n in ALL_ENCODERS if os.path.exists(npz_path(n))]
    if "retfound_mae" not in baseline_names:
        raise SystemExit("cache retfound_mae first (the baseline): --cache --only retfound_mae")
    # Extra view/input_size/disc_center caches (high-res, disc crop, detected center, ...) are
    # discovered automatically and folded into the SAME comparison table; the baseline scan over
    # ALL_ENCODERS is untouched, so this is purely additive.
    configs = _discover_configs()
    entries = [(n, npz_path(n)) for n in baseline_names] + configs
    print("=" * 96)
    hdr = f"ENCODER BAKE-OFF — frozen-feature probes over 631 GRAPE recs, our folds; encoders: {baseline_names}"
    if configs:
        hdr += f"; configs: {[c[0] for c in configs]}"
    print(hdr)
    print("=" * 96)
    rows = {}
    for label, path in entries:
        d = np.load(path)
        sev, spat, md = d['sev'], d['spat'], d['md']
        vf52, lat, pid, fold = d['vf52'], d['lat'], d['pid'], d['fold']
        sc_oof, mae_oof = oof_sev_corr(sev, md, fold)
        sc_505 = sev_curve_505(sev, md, pid, fold)
        pc, rc, plo, phi = spatial_pcorr(spat, vf52, lat, pid, fold)
        rows[label] = dict(sev_corr_oof=sc_oof, sev_mae_oof=mae_oof, sev_corr_505=sc_505,
                          spatial_pcorr=pc, spatial_res_corr=rc, spatial_ci=[plo, phi],
                          grid=[int(x) for x in d['grid']])
        print(f"\n[{label}]  grid={tuple(int(x) for x in d['grid'])}")
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
    print(f"{'label':>28}{'Δsev@505':>11}{'Δsev_oof':>11}{'Δspatial':>11}   verdict")
    winners = []
    for label, _ in entries:
        if label == "retfound_mae":
            continue
        dsev = rows[label]['sev_corr_505'] - base['sev_corr_505']
        dsev_oof = rows[label]['sev_corr_oof'] - base['sev_corr_oof']
        dspat = rows[label]['spatial_pcorr'] - base['spatial_pcorr']
        passed = (dsev >= 0.03) or (dspat >= 0.05)
        verdict = "ADVANCE to A3" if passed else "sub-gate"
        if passed:
            winners.append(label)
        print(f"{label:>28}{dsev:>+11.3f}{dsev_oof:>+11.3f}{dspat:>+11.3f}   {verdict}")
    print("-" * 96)
    if winners:
        print(f"GATE PASSED: {winners} -> Task A3 (integrate winner, fold-0 scout, gated 5-fold).")
    else:
        print("GATE FAILED for every encoder/config -> STOP Phase A ('encoder swap does not help our "
              "data'); proceed to Phase B (data).")
    print("=" * 96)
    json.dump(rows, open(os.path.join(AUTO, "encoder_bakeoff.json"), "w"), indent=2, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true', help="pass 1: cache frozen features (torch, alone)")
    ap.add_argument('--only', default=None, help="restrict --cache to one encoder name")
    ap.add_argument('--view', default='full', choices=['full', 'disc'],
                     help="'full' = whole fundus image (baseline); 'disc' = laterality-aware disc crop")
    ap.add_argument('--input-size', type=int, default=224,
                     help="frozen-encoder input resolution; 224 is today's baseline (byte-identical)")
    ap.add_argument('--disc-center', default='fixed', choices=['fixed', 'detected'],
                     help="disc crop center: 'fixed' box (baseline) or 'detected' green-channel centroid")
    ap.add_argument('--probe', action='store_true', help="pass 2: curves + spatial + gate (numpy)")
    a = ap.parse_args()
    if a.cache:
        cache_all(a.only, view=a.view, input_size=a.input_size, disc_center=a.disc_center)
    if a.probe:
        probe()
    if not a.cache and not a.probe:
        ap.error("choose --cache and/or --probe")


if __name__ == "__main__":
    main()
