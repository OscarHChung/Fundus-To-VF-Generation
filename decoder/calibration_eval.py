"""Iteration 8 — variance-matching output calibration (post-hoc, no retrain).

The champion regresses to the mean (slope 0.37, pooled corr 0.55 ⇒ σ_pred/σ_truth
≈ 0.67: predictions are under-dispersed, deep points under-deepened). This script
fits an affine calibration ŷ' = μ_t + b·(ŷ − μ_p) on the TRAIN-eval split (fit on
train only — no val leakage) and applies it to VAL, sweeping the strength s so b
runs from 1 (de-bias only) to σ_t/σ_p (full variance match). It prints the trade
curve (overall MAE ↔ floor/deep MAE ↔ slope) so we can pick the balanced point that
serves goal-1 (severe) without trashing goal-2 (overall sub-4).

Usage:
  python decoder/calibration_eval.py [path/to/checkpoint.pth]
Default checkpoint: the current champion (decoder/results/champion/best_model.pth).
On a severe improvement it writes decoder/results/champion/severe_champion.json.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training as T


def _pooled(model, cache):
    """Return pooled (pred, target) over all valid points of a precomputed cache."""
    P, G = [], []
    model.eval()
    with torch.no_grad():
        for item in cache:
            latent = item['latent'].to(T.DEVICE)
            lat    = item['lat']
            lat_s  = lat[0] if isinstance(lat, (list, tuple)) else lat
            pred   = model.decode_latent(latent, lat, average_multi=True).cpu()
            hvf    = item['hvf']
            hvf    = hvf.unsqueeze(0) if hvf.dim() == 1 else hvf
            valid_idx = T.valid_indices_od if lat_s.startswith('OD') else T.valid_indices_os
            t52 = hvf[0][valid_idx]
            p52 = pred[0]
            m   = t52 < T.MASKED_VALUE_THRESHOLD
            P.extend(p52[m].tolist())
            G.extend(t52[m].tolist())
    return np.array(P, dtype=np.float64), np.array(G, dtype=np.float64)


def _metrics(P, G):
    err = P - G
    ae  = np.abs(err)
    def band(lo, hi):
        m = (G >= lo) & (G < hi)
        return float(ae[m].mean()) if m.any() else float('nan'), int(m.sum())
    floor, fn = band(0, 10)
    deep,  dn = band(0, 16)
    slope = float(np.polyfit(G, P, 1)[0])
    corr  = float(np.corrcoef(P, G)[0, 1])
    return {'mae': float(ae.mean()), 'floor': floor, 'floor_n': fn,
            'deep': deep, 'deep_n': dn, 'slope': slope, 'corr': corr,
            'bias': float(err.mean())}


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(T.CHAMPION_DIR, "best_model.pth")
    print(f"Calibration eval — checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    use_dist   = ckpt.get('use_dist', False)
    dist_blend = ckpt.get('dist_blend', T.DIST_BLEND)

    model = T.PerPointVFModel(T.base_model, use_dist=use_dist, dist_blend=dist_blend)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(T.DEVICE)
    print(f"  loaded (missing={len(missing)}, unexpected={len(unexpected)}), "
          f"use_dist={use_dist}")

    # Build caches (encoder is frozen → identical to training's caches).
    val_ds = T.MultiImageDataset(T.VAL_JSON, T.FUNDUS_DIR, T.val_transform,
                                 mode='val', use_tta=T.USE_TTA)
    tr_ds  = T.MultiImageDataset(T.TRAIN_JSON, T.FUNDUS_DIR, T.val_transform,
                                 mode='val', use_tta=False)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=T.val_collate_fn)
    tr_loader  = DataLoader(tr_ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=T.val_collate_fn)
    print("Caching encoder features …")
    val_cache = T.precompute_features(model, val_loader, T.DEVICE, "val")
    tr_cache  = T.precompute_features(model, tr_loader,  T.DEVICE, "train-eval")

    Ptr, Gtr = _pooled(model, tr_cache)   # fit calibration on TRAIN only
    Pva, Gva = _pooled(model, val_cache)  # report on VAL

    mu_p, sig_p = Ptr.mean(), Ptr.std()
    mu_t, sig_t = Gtr.mean(), Gtr.std()
    b_full = sig_t / (sig_p + 1e-8)
    print(f"\nFit on train: μ_pred={mu_p:.2f} σ_pred={sig_p:.2f} | "
          f"μ_true={mu_t:.2f} σ_true={sig_t:.2f} | full var-match b={b_full:.3f}")

    base = _metrics(Pva, Gva)
    print(f"\nVAL raw (no calibration): MAE={base['mae']:.3f} floor={base['floor']:.2f} "
          f"deep={base['deep']:.2f} slope={base['slope']:.3f} corr={base['corr']:.3f} "
          f"bias={base['bias']:+.2f}")

    print(f"\n{'s':>5} {'b':>6} | {'MAE':>6} {'floor(0-10)':>11} {'deep(<16)':>10} "
          f"{'slope':>6} {'bias':>6}")
    print("-" * 60)
    rows = []
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        b = 1.0 + s * (b_full - 1.0)
        Pc = np.clip(mu_t + b * (Pva - mu_p), 0, 35)
        m = _metrics(Pc, Gva)
        m['s'] = s; m['b'] = b
        rows.append(m)
        print(f"{s:>5.2f} {b:>6.3f} | {m['mae']:>6.3f} {m['floor']:>11.2f} "
              f"{m['deep']:>10.2f} {m['slope']:>6.3f} {m['bias']:>+6.2f}")

    # Pick the balanced severe point: lowest floor MAE subject to overall MAE < 4.6.
    feasible = [r for r in rows if r['mae'] < 4.6]
    pick = min(feasible, key=lambda r: r['floor']) if feasible else \
           min(rows, key=lambda r: r['floor'])
    print(f"\n▸ Balanced pick: s={pick['s']:.2f} (b={pick['b']:.3f}) → "
          f"floor {base['floor']:.2f}→{pick['floor']:.2f}, "
          f"overall {base['mae']:.3f}→{pick['mae']:.3f}, "
          f"slope {base['slope']:.3f}→{pick['slope']:.3f}")

    # Save a SEVERE champion if this beats the stored one (guard overall MAE < 4.6).
    rec_path = os.path.join(T.CHAMPION_DIR, "severe_champion.json")
    prev = None
    if os.path.exists(rec_path):
        try:
            with open(rec_path) as f:
                prev = json.load(f)
        except Exception:
            prev = None
    prev_floor = prev.get('floor', float('inf')) if prev else float('inf')
    if pick['mae'] < 4.6 and pick['floor'] < prev_floor:
        import datetime
        rec = {'floor': pick['floor'], 'overall_mae': pick['mae'],
               'deep': pick['deep'], 'slope': pick['slope'], 'bias': pick['bias'],
               'method': 'varmatch_calibration', 'checkpoint': ckpt_path,
               'calibration': {'s': pick['s'], 'b': pick['b'],
                               'mu_pred': float(mu_p), 'mu_true': float(mu_t),
                               'sig_pred': float(sig_p), 'sig_true': float(sig_t)},
               'previous_floor': (None if not np.isfinite(prev_floor) else float(prev_floor)),
               'timestamp': datetime.datetime.now().isoformat(timespec='seconds')}
        os.makedirs(T.CHAMPION_DIR, exist_ok=True)
        with open(rec_path, 'w') as f:
            json.dump(rec, f, indent=2)
        pstr = f"{prev_floor:.2f}" if np.isfinite(prev_floor) else "—"
        print(f"\n🏆 NEW SEVERE CHAMPION: floor {pick['floor']:.2f} dB "
              f"(prev {pstr}) → {rec_path}")
    else:
        print(f"\n(Severe champion unchanged: floor {pick['floor']:.2f} ≥ best {prev_floor})")


if __name__ == "__main__":
    main()
