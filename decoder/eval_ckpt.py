"""Honest per-patient evaluation for any trained checkpoint, raw + variance-match calibrated.

Unlike calibration_eval.py (hard-wired to the leaky split), this takes explicit --train-json
/--val-json so a model trained on a per-patient CV fold is scored on its clean held-out, with
the calibration fit ONLY on that fold's train. Reports the full severe-relevant metric set
(MAE / floor / deep / slope / corr / eyeCorr / σ-ratio) so Session-3 levers are compared
apples-to-apples.

  python decoder/eval_ckpt.py <ckpt.pth> --train-json <fold_train.json> --val-json <fold_val.json>
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training as T
import diagnostics as D
from torch.utils.data import DataLoader


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    use_dist   = ckpt.get('use_dist', False)
    dist_blend = ckpt.get('dist_blend', T.DIST_BLEND)
    mean_res   = ckpt.get('mean_residual', False)
    global_head = ckpt.get('global_head', False)
    model = T.PerPointVFModel(T.base_model, use_dist=use_dist, dist_blend=dist_blend,
                              mean_residual=mean_res, global_head=global_head)
    model.load_state_dict(state, strict=False)
    model.to(T.DEVICE)
    return model


def per_eye_preds(model, json_path, use_tta=True):
    """Return (preds, trues): lists of (52,) arrays in OD/OS query order, nan at masked."""
    ds = T.MultiImageDataset(json_path, T.FUNDUS_DIR, T.val_transform, mode='val', use_tta=use_tta)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=T.val_collate_fn)
    cache = T.precompute_features(model, loader, T.DEVICE, os.path.basename(json_path))
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for item in cache:
            latent = item['latent'].to(T.DEVICE); lat = item['lat']
            lat_s = lat[0] if isinstance(lat, (list, tuple)) else lat
            pred = model.decode_latent(latent, lat, average_multi=True).cpu().numpy()[0]
            hvf = item['hvf']; hvf = hvf.unsqueeze(0) if hvf.dim() == 1 else hvf
            vi = T.valid_indices_od if lat_s.startswith('OD') else T.valid_indices_os
            t = hvf[0][vi].numpy().astype(np.float64)
            t[t >= T.MASKED_VALUE_THRESHOLD] = np.nan
            preds.append(pred.astype(np.float64)); trues.append(t)
    return preds, trues


def pooled_stats(preds, trues):
    P = np.concatenate([p[~np.isnan(t)] for p, t in zip(preds, trues)])
    Tt = np.concatenate([t[~np.isnan(t)] for t in trues])
    return P.mean(), P.std(), Tt.mean(), Tt.std()


def apply_calib(preds, mu_p, mu_t, b):
    return [np.clip(mu_t + b * (p - mu_p), 0, 35) for p in preds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--no-tta', action='store_true')
    a = ap.parse_args()
    use_tta = not a.no_tta

    model = load_model(a.ckpt)
    print(f"Eval {os.path.basename(a.ckpt)} | val={os.path.basename(a.val_json)} "
          f"train={os.path.basename(a.train_json)} | TTA={use_tta}")
    vp, vt = per_eye_preds(model, a.val_json, use_tta)
    tp, tt = per_eye_preds(model, a.train_json, use_tta)

    raw = D.pooled_metrics(vp, vt)
    print(f"\nRAW          {D.fmt(raw)}")

    mu_p, sig_p, mu_t, sig_t = pooled_stats(tp, tt)   # fit calibration on TRAIN only
    b_full = sig_t / (sig_p + 1e-8)
    print(f"\ncalib fit (train): μp={mu_p:.2f} σp={sig_p:.2f} | μt={mu_t:.2f} σt={sig_t:.2f} "
          f"| full b={b_full:.3f}")
    print(f"{'s':>5}{'b':>7} | metrics (val, calibrated)")
    rows = []
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        b = 1.0 + s * (b_full - 1.0)
        m = D.pooled_metrics(apply_calib(vp, mu_p, mu_t, b), vt); m['s'] = s; m['b'] = b
        rows.append(m)
        print(f"{s:>5.2f}{b:>7.3f} | {D.fmt(m)}")
    # balanced severe pick: lowest floor with overall MAE within 0.35 of raw
    feas = [r for r in rows if r['mae'] < raw['mae'] + 0.35]
    pick = min(feas or rows, key=lambda r: r['floor'])
    print(f"\n▸ severe pick s={pick['s']:.2f}: floor {raw['floor']:.2f}→{pick['floor']:.2f} "
          f"slope {raw['slope']:.3f}→{pick['slope']:.3f} MAE {raw['mae']:.3f}→{pick['mae']:.3f}")


if __name__ == "__main__":
    main()
