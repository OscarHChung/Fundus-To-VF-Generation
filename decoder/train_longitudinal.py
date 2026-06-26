"""Train + per-patient 5-fold CV for the LongitudinalVFModel.

Per fold: warm-start the fundus branch from the matching long_global_f{k} checkpoint (so the model
begins at the naive blend ~3.71 and only learns the prior-VF denoising delta + visit-1 sharpening),
train, then score out-of-fold. Pool OOF predictions over all 631 records, RAW + variance-match
calibrated, with the severity-stratified + persistence ablation reported.

  python decoder/train_longitudinal.py --tag long_prior --epochs 30
"""
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training as T
import diagnostics as D
import eval_ckpt as E
from longitudinal_model import LongitudinalVFModel
from longitudinal_dataset import LongitudinalDataset, collate_train, collate_val

CUR = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(CUR, "results", "cv_long")
AUTO = os.path.join(CUR, "results", "auto")
AE = os.path.join(CUR, "pretrained_vf_ae.pth")
DEVICE = T.DEVICE
from garway_heath_weighting import sector_weight_tensors
SECTORS = sector_weight_tensors(device=DEVICE, normalize=True)


def eval_oof(model, val_json):
    ds = LongitudinalDataset(val_json, mode="val", use_tta=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_val)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xs, hvf, lat, pv, pm, dt, hp in dl:
            xs = xs.to(DEVICE); V = xs.shape[0]
            latent = model._encode(xs)
            pv_e = pv[None].expand(V, -1).to(DEVICE); pm_e = pm[None].expand(V, -1).to(DEVICE)
            dt_e = dt[None].expand(V).to(DEVICE); hp_e = hp[None].expand(V).to(DEVICE)
            pred = model.decode_latent_long(latent, pv_e, pm_e, dt_e, hp_e,
                                            laterality=[lat] * V, average_multi=True)
            p = pred.cpu().numpy()[0]
            vi = T.valid_indices_od if lat.startswith("OD") else T.valid_indices_os
            t = hvf.numpy()[vi].astype(np.float64); t[t >= T.MASKED_VALUE_THRESHOLD] = np.nan
            preds.append(p.astype(np.float64)); trues.append(t)
    return preds, trues


def train_fold(fold, epochs, lr, aux_w, warm):
    tr = LongitudinalDataset(os.path.join(CV, f"fold{fold}_train.json"), mode="train")
    sampler = WeightedRandomSampler(tr.get_sample_severity(), num_samples=len(tr), replacement=True)
    dl = DataLoader(tr, batch_size=16, sampler=sampler, num_workers=0, collate_fn=collate_train)
    model = LongitudinalVFModel(T.base_model, AE, global_head=True).to(DEVICE)
    if warm and os.path.exists(warm.format(fold)):
        sd = torch.load(warm.format(fold), map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("model_state_dict", sd))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  warm-start fundus from {os.path.basename(warm.format(fold))} "
              f"(loaded {len(sd) - len(unexpected)} tensors)")
    # FREEZE the warm-started fundus branch (visit-1 stays at its long_global value, no degradation);
    # train ONLY the longitudinal params (delta denoiser + interval + gate).
    LONG = ("delta_head", "interval", "alpha", "no_prior_token")
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(LONG)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"  trainable (longitudinal only): {sum(p.numel() for p in params):,} params")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    val_json = os.path.join(CV, f"fold{fold}_val.json")
    best_mae, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for x, hvf, lats, pv, pm, dt, hp in dl:
            x, pv, pm, dt, hp = [t.to(DEVICE) for t in (x, pv, pm, dt, hp)]
            pred = model(x, pv, pm, dt, hp, laterality=lats, average_multi=False)
            loss, _, _ = T.compute_loss(pred, hvf, lats, epoch=ep,
                                        attn_weights=model._last_attn_weights,
                                        sector_weights=SECTORS, sector_combine="sector_only")
            loss = loss + 0.01 * (model._last_delta ** 2).mean()   # mild delta regularizer
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
        if ep >= 4 and (ep % 2 == 0 or ep == epochs - 1):
            vp, vt = eval_oof(model, val_json)
            mae = D.pooled_metrics(vp, vt)["mae"]
            if mae < best_mae:
                best_mae = mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                              if not k.startswith("ae.")}
            print(f"  fold{fold} ep{ep:2d} val MAE {mae:.3f} (best {best_mae:.3f})")
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return model, val_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="long_prior")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-w", type=float, default=0.3)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--warm", default=os.path.join(AUTO, "long_global_f{}_best.pth"))
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    vp_all, vt_all, vp_cal = [], [], []
    for f in folds:
        print(f"\n=== FOLD {f} ===")
        model, vj = train_fold(f, a.epochs, a.lr, a.aux_w, a.warm)
        torch.save({"model": model.state_dict(), "global_head": True, "longitudinal": True},
                   os.path.join(AUTO, f"{a.tag}_f{f}_best.pth"))
        vp, vt = eval_oof(model, vj)
        tp, tt = eval_oof(model, os.path.join(CV, f"fold{f}_train.json"))
        mu_p, sig_p, mu_t, sig_t = E.pooled_stats(tp, tt)
        b = sig_t / (sig_p + 1e-8)
        vp_all += vp; vt_all += vt; vp_cal += E.apply_calib(vp, mu_p, mu_t, b)
        print(f"  fold {f}: {D.fmt(D.pooled_metrics(vp, vt))}")
    raw = D.pooled_metrics(vp_all, vt_all); cal = D.pooled_metrics(vp_cal, vt_all)
    print("\n" + "=" * 90)
    print(f"LONGITUDINAL 5-FOLD OOF ({len(vt_all)} recs)  RAW   {D.fmt(raw)}")
    D.stratified_report(vp_all, vt_all)
    print(f"LONGITUDINAL 5-FOLD OOF ({len(vt_all)} recs)  CALIB {D.fmt(cal)}")
    D.stratified_report(vp_cal, vt_all)
    print("=" * 90)
    print("refs: fundus-only 4.29 ; persistence 3.71 ; target sub-3.74.")
    json.dump({"tag": a.tag, "raw": raw, "calib": cal, "n": len(vt_all)},
              open(os.path.join(AUTO, f"{a.tag}_cv.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
