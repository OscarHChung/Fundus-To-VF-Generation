"""Diag C (design's D1) — is the fundus-only ceiling DATA-limited or INFORMATION-limited?

The whole verdict hinges on this and it was never run. Cache frozen RETFound features once over the
631 images, then fit a SEVERITY probe (features → per-eye MD) on n = 40..505 training eyes (20
patient-disjoint resamples each), scored OOF on held-out folds. Plot sev_corr vs n.

  - Still rising at n≈505 (Δ ≥ ~0.02 per doubling)  ⇒ DATA/encoder-limited: more data OR a better
    encoder can lift sev_corr, and native sub-4.0 is on the table. Levers (better encoder, more data)
    are worth building.
  - Flat beyond n≈120                                ⇒ INFORMATION-limited: the frozen RETFound
    representation has hit its ceiling; only a *different/better encoder* (not more data) can help,
    and metadata/longitudinal are the realistic sub-4.0 routes.

Two passes: (1) cache features [torch, run ALONE]; (2) probe [numpy]. Re-run skips the cache.

  python decoder/diag_d1_learning_curve.py            # cache (if needed) + learning curve
  python decoder/diag_d1_learning_curve.py --probe    # probe only (features already cached)
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cv_long")
FEAT = os.path.join(AUTO, "d1_retfound_features.npz")
CKPT = os.path.join(AUTO, "m1sev_f0_best.pth")   # any checkpoint — only its frozen encoder is used


def cache_features():
    import torch
    import eval_ckpt as E, training as T
    from torch.utils.data import DataLoader
    model = E.load_model(CKPT); model.eval()
    X, MD, RESID, LAT, PID, FOLD = [], [], [], [], [], []
    for f in range(5):
        ds = T.MultiImageDataset(os.path.join(CV_DIR, f"fold{f}_val.json"), T.FUNDUS_DIR,
                                 T.val_transform, mode='train')          # 1 full-view img/record
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        items = json.load(open(os.path.join(CV_DIR, f"fold{f}_val.json")))
        si = 0
        with torch.no_grad():
            for imgs, hvf, lat in loader:
                pre = model._encode_prefix(imgs.to(T.DEVICE)).float().cpu().numpy()   # (B,197,1024)
                cls = pre[:, 0, :]; patch = pre[:, 1:, :].mean(1)                     # (B,1024) each
                feat = np.concatenate([cls, patch], axis=1)                           # (B,2048)
                for b in range(feat.shape[0]):
                    vi = T.valid_indices_od if str(lat[b]).startswith('OD') else T.valid_indices_os
                    v = np.asarray(hvf[b]).reshape(-1)[vi].astype(np.float64)
                    v = v[v < T.MASKED_VALUE_THRESHOLD]
                    X.append(feat[b]); MD.append(v.mean())
                    LAT.append(str(lat[b])[:2]); PID.append(int(items[si]['PatientID'])); FOLD.append(f)
                    si += 1
    np.savez_compressed(FEAT, X=np.array(X), md=np.array(MD), lat=np.array(LAT),
                        pid=np.array(PID), fold=np.array(FOLD))
    print(f"cached {len(X)} features ({np.array(X).shape[1]}-d) → {os.path.basename(FEAT)}")


def ridge_fit_predict(Xtr, ytr, Xte, lam=50.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    b = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    return Xte @ b + ytr.mean()


def learning_curve(seed=0):
    d = np.load(FEAT)
    X, md, pid, fold = d['X'], d['md'], d['pid'], d['fold']
    rng = np.random.default_rng(seed)
    ns = [40, 80, 120, 170, 210, 300, 400, 505]
    print("=" * 84)
    print(f"D1 severity learning curve — frozen RETFound features ({X.shape[1]}-d), OOF sev_corr")
    print("-" * 84)
    print(f"{'n_train_eyes':>13}{'sev_corr':>11}{'sev_MAE':>10}   (mean ± sd over held-out folds × resamples)")
    prev = None
    curve = {}
    for n in ns:
        corrs, maes = [], []
        for f in range(5):
            te = fold == f
            pool_idx = np.where(~te)[0]
            pool_pids = np.unique(pid[pool_idx])
            if n > len(pool_idx):
                continue
            for _ in range(20):
                # patient-disjoint subsample of ~n eyes: draw patients until we reach n rows
                perm = rng.permutation(pool_pids); chosen, cnt = [], 0
                for p in perm:
                    rows = pool_idx[pid[pool_idx] == p]
                    chosen.extend(rows.tolist()); cnt += len(rows)
                    if cnt >= n:
                        break
                tr = np.array(chosen)
                pred = ridge_fit_predict(X[tr], md[tr], X[te])
                if md[te].std() > 1e-6:
                    corrs.append(np.corrcoef(pred, md[te])[0, 1])
                    maes.append(np.abs(pred - md[te]).mean())
        c = float(np.mean(corrs)); mae = float(np.mean(maes))
        curve[n] = c
        delta = f"  Δ{c - prev:+.3f}" if prev is not None else ""
        print(f"{n:>13}{c:>11.3f}{mae:>10.3f}{delta}")
        prev = c
    # verdict: change over the top doubling (210 → 505 ≈ 1.3 doublings)
    top = curve[505] - curve[210]
    print("-" * 84)
    print(f"sev_corr change n=210→505: {top:+.3f}   "
          f"({'STILL RISING → data/encoder-limited' if top >= 0.02 else 'FLAT → information-limited'})")
    print("refs: m1sev OOF sev_corr 0.807 ; p1disc (disc crop) 0.802 ; frontier for sub-4.0 needs ≈0.875")
    print("=" * 84)
    json.dump(curve, open(os.path.join(AUTO, 'diag_d1_curve.json'), 'w'), indent=2, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true', help="skip caching (features already on disk)")
    a = ap.parse_args()
    if not a.probe and not os.path.exists(FEAT):
        cache_features()
    learning_curve()


if __name__ == "__main__":
    main()
