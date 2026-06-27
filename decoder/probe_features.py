"""Linear probes on FROZEN RETFound features — locate the within-eye spatial signal.

Iter-A baselines showed the model adds ~no within-eye spatial signal (eyeCorr < a fixed
template). The decisive question: is the per-point spatial signal LINEARLY present in the
frozen features (→ decoder/training is leaving it on the table, fixable with no new data) or
ABSENT (→ need encoder adaptation / disc view)? We fit cheap ridge probes per per-patient
fold (OOF) and read eyeCorr/floor as an UPPER BOUND on what a linear decoder could extract.

All eyes are put in OD-canonical orientation (OS image flipped horizontally, OS field fliplr)
so one retinotopic mapping serves both and OD/OS data pool. A sanity check confirms the
mirror by correlating the OD vs mirrored-OS per-point mean templates.

Probes:
  CLS        — ridge from final CLS token (1024)            → global/severity + any CLS spatial
  ALLPATCH   — ridge from [mean-pooled patches ‖ CLS] (2048) → ≈ the model's PointHead input
  RETINO     — per-point ridge from the retinotopically-mapped patch (3×3 pooled, 1024)
  DISC       — per-point ridge from the disc-crop's mapped patch (tests disc/RNFL signal)
  MULTILAYER — ridge from concat CLS of blocks {6,12,18,24} (4096)  (richer mid-level detail)
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D
import training as T   # heavy: loads RETFound encoder + transforms

DEVICE = T.DEVICE
CACHE  = os.path.join("/tmp", "grape_probe_features.pt")


# ==============================================================
# Canonical (OD-oriented) extraction
# ==============================================================
def vec52_canonical(record):
    hvf = np.array(record['hvf'], dtype=np.float64)            # 8×9
    lat = str(record.get('Laterality', 'OD')).strip().upper()
    if not lat.startswith('OD'):
        hvf = np.fliplr(hvf)                                   # mirror field to OD orientation
    v = hvf.flatten()[D.valid_indices_od]
    v[v >= D.MASKED_VALUE_THRESHOLD] = np.nan
    return v


@torch.no_grad()
def encode_layers(x, layer_ids=(6, 12, 18, 24)):
    """Frozen RETFound forward WITHOUT MAE shuffle (mirrors training._encode), returning
    final CLS (1024), final patches (196,1024), and CLS at the requested blocks."""
    enc = T.base_model
    h = enc.patch_embed(x)
    h = h + enc.pos_embed[:, 1:, :]
    cls = (enc.cls_token + enc.pos_embed[:, :1, :]).expand(h.shape[0], -1, -1)
    h = torch.cat((cls, h), dim=1)
    layer_cls = {}
    for bi, blk in enumerate(enc.blocks, start=1):
        h = blk(h)
        if bi in layer_ids:
            layer_cls[bi] = h[:, 0, :].clone()
    h = enc.norm(h)
    return h[:, 0, :], h[:, 1:, :], layer_cls


def _to_tensor(pil):
    return T.val_transform(pil).unsqueeze(0).to(DEVICE)


def build_feature_cache(layer_ids=(6, 12, 18, 24)):
    if os.path.exists(CACHE):
        print(f"Loading cached features → {CACHE}")
        return torch.load(CACHE, weights_only=False)
    records = D.load_records()
    T.base_model.to(DEVICE).eval()
    feats = []
    print(f"Extracting frozen features for {len(records)} eyes (canonical orientation)…")
    for ri, r in enumerate(records):
        lat = str(r.get('Laterality', 'OD')).strip().upper(); is_od = lat.startswith('OD')
        imgs = r['FundusImage'] if isinstance(r['FundusImage'], list) else [r['FundusImage']]
        cls_f, patch_f, disc_f, lyr_f = [], [], [], {b: [] for b in layer_ids}
        for fn in imgs:
            pil = Image.open(os.path.join(T.FUNDUS_DIR, fn)).convert('RGB')
            if not is_od:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)       # OS → OD orientation
            cls, patch, lyr = encode_layers(_to_tensor(pil), layer_ids)
            cls_f.append(cls.cpu()); patch_f.append(patch.cpu())
            for b in layer_ids: lyr_f[b].append(lyr[b].cpu())
            # disc crop (already OD-oriented after the flip → use OD disc box)
            disc_pil = T.disc_crop_pil(pil, 'OD')
            _, dpatch, _ = encode_layers(_to_tensor(disc_pil), ())
            disc_f.append(dpatch.cpu())
        feats.append({
            'pid': r.get('PatientID', ri), 'lat': lat, 'is_od': is_od,
            'cls':   torch.cat(cls_f).mean(0).numpy(),                 # (1024,)
            'patch': torch.cat(patch_f).mean(0).numpy(),               # (196,1024)
            'disc':  torch.cat(disc_f).mean(0).numpy(),                # (196,1024)
            'layers': {b: torch.cat(lyr_f[b]).mean(0).numpy() for b in layer_ids},
            'y': vec52_canonical(r),                                   # (52,) nan-masked
        })
        if (ri + 1) % 50 == 0:
            print(f"  {ri+1}/{len(records)}")
    torch.save(feats, CACHE)
    print(f"Cached → {CACHE}")
    return feats


# ==============================================================
# Ridge (multi-output, closed form, standardized features)
# ==============================================================
def ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    if Y.ndim == 2:
        Y = np.where(np.isnan(Y), np.nanmean(Y, axis=0, keepdims=True), Y)
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1]); A[-1, -1] -= alpha   # don't penalize bias
    W = np.linalg.solve(A, Xs.T @ Y)
    return (W, mu, sd)


def ridge_pred(model, X):
    W, mu, sd = model
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return Xs @ W


# ==============================================================
# Probes
# ==============================================================
def _mapped_patch_idx():
    """For each of the 52 query points, the 3×3 patch neighborhood (flat idx) at its
    OD retinotopic location (training.VF_TO_PATCH_PRIOR)."""
    out = []
    for p in range(D.NUM_VALID_POINTS):
        pr, pc = T.VF_TO_PATCH_PRIOR[p]
        nb = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = pr + dr, pc + dc
                if 0 <= r < 14 and 0 <= c < 14:
                    nb.append(r * 14 + c)
        out.append(nb)
    return out


def _eval(feats, fold_val_idx, predict_fn):
    """predict_fn(train_feats, val_feats) -> list of (52,) preds aligned to val_feats."""
    preds, trues = [], []
    for j, val_idx in enumerate(fold_val_idx):
        val = [feats[i] for i in val_idx]
        train = [feats[i] for jj, f in enumerate(fold_val_idx) if jj != j for i in f]
        P = predict_fn(train, val)
        for f, p in zip(val, P):
            preds.append(p); trues.append(f['y'])
    return D.pooled_metrics(preds, trues)


def make_cls_probe(alpha):
    def fn(train, val):
        Xtr = np.vstack([f['cls'] for f in train]); Ytr = np.vstack([f['y'] for f in train])
        m = ridge_fit(Xtr, Ytr, alpha)
        Xva = np.vstack([f['cls'] for f in val])
        return list(ridge_pred(m, Xva))
    return fn


def make_allpatch_probe(alpha):
    def fn(train, val):
        Xtr = np.vstack([np.concatenate([f['patch'].mean(0), f['cls']]) for f in train])
        Ytr = np.vstack([f['y'] for f in train])
        m = ridge_fit(Xtr, Ytr, alpha)
        Xva = np.vstack([np.concatenate([f['patch'].mean(0), f['cls']]) for f in val])
        return list(ridge_pred(m, Xva))
    return fn


def make_layers_probe(alpha, layer_ids=(6, 12, 18, 24)):
    def fn(train, val):
        def feat(f): return np.concatenate([f['layers'][b] for b in layer_ids])
        Xtr = np.vstack([feat(f) for f in train]); Ytr = np.vstack([f['y'] for f in train])
        m = ridge_fit(Xtr, Ytr, alpha)
        return list(ridge_pred(m, np.vstack([feat(f) for f in val])))
    return fn


def make_retino_probe(alpha, key='patch'):
    nbs = _mapped_patch_idx()
    def fn(train, val):
        P = np.zeros((len(val), D.NUM_VALID_POINTS))
        for p in range(D.NUM_VALID_POINTS):
            idx = nbs[p]
            Xtr = np.vstack([train_f[key][idx].mean(0) for train_f in train])
            ytr = np.array([train_f['y'][p] for train_f in train])
            ok = ~np.isnan(ytr)
            m = ridge_fit(Xtr[ok], ytr[ok], alpha)
            Xva = np.vstack([val_f[key][idx].mean(0) for val_f in val])
            P[:, p] = ridge_pred(m, Xva)
        return list(P)
    return fn


def main():
    feats = build_feature_cache()
    # ---- sanity: OD vs mirrored-OS per-point template should agree ----
    od = np.vstack([f['y'] for f in feats if f['is_od']])
    osm = np.vstack([f['y'] for f in feats if not f['is_od']])
    tod, tos = np.nanmean(od, 0), np.nanmean(osm, 0)
    print(f"Mirror sanity — corr(OD template, mirrored-OS template) = "
          f"{np.corrcoef(tod, tos)[0,1]:.3f}  (want >0.9)\n")

    records = D.load_records()
    fold_val_idx = D.build_patient_folds(records, k=5)

    print(f"{'probe':<12}{'alpha':>7}  metrics")
    print("-" * 110)
    probes = {
        'CLS':        (make_cls_probe,      [10, 100, 1000]),
        'ALLPATCH':   (make_allpatch_probe, [10, 100, 1000]),
        'MULTILAYER': (make_layers_probe,   [10, 100, 1000]),
        'RETINO':     (make_retino_probe,   [1, 10, 100]),
        'DISC':       (lambda a: make_retino_probe(a, key='disc'), [1, 10, 100]),
    }
    best = {}
    for name, (ctor, alphas) in probes.items():
        rows = [(a, _eval(feats, fold_val_idx, ctor(a))) for a in alphas]
        a, m = min(rows, key=lambda r: r[1]['mae'])      # pick best-MAE alpha
        best[name] = m
        print(f"{name:<12}{a:>7}  {D.fmt(m)}")
    print("\nReference: perpoint_mean eyeCorr 0.488 / MAE 6.01 ; eye_oracle MAE 4.09 ; "
          "champion(leaky) MAE 4.12 eyeCorr 0.41.")
    print("Read: a probe eyeCorr >> 0.49 ⇒ spatial signal IS in the frozen features (decoder "
          "is leaving it on the table). All probes ≈0.49 ⇒ signal absent → need encoder/disc.")


def eval_champ():
    print("eval-champ: the existing champion trained on grape_train (211 of 263 eyes), so it "
          "has no clean CV holdout. Honest CV numbers require retraining per-fold (Iter B+).")


if __name__ == "__main__":
    main()
