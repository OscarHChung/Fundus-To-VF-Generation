"""Memory-frugal LoRA training via a cached frozen prefix (for a memory-tight box).

The frozen RETFound prefix (blocks[:n_frozen]) is deterministic given the input, so we compute it
ONCE per train/val eye (augmentation OFF), free the prefix blocks to reclaim ~1 GB, then train only
the LoRA suffix (last K blocks) + decoder on the cached prefix features. This removes both the
per-step 20-block forward AND the 1.2 GB prefix from memory, so encoder-gradient (LoRA) training
fits where the normal live path OOMs.

Inference is unchanged and fundus-only: eval_ckpt reloads a FULL LoRA model (prefix from base_model)
and runs the whole encoder under no_grad, so the saved checkpoint scores exactly like any other.

  python decoder/train_lora_cached.py --train-json <fold_train.json> --val-json <fold_val.json> \
      --out-tag lorac_f0 --epochs 40 --lora-blocks 4 --lora-rank 8
"""
import os, sys, gc, json, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training as T
import diagnostics as D
from garway_heath_weighting import sector_weight_tensors

AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "auto")


def cache_prefix(model, json_path):
    """Return list of dicts {prefix:(V,197,1024) cpu, hvf:(72,), lat:str} — frozen prefix per eye."""
    ds = T.MultiImageDataset(json_path, T.FUNDUS_DIR, T.val_transform, mode='val', use_tta=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=T.val_collate_fn)
    out = []
    model.eval()
    with torch.no_grad():
        for imgs, hvf, lat in loader:
            imgs = imgs.to(T.DEVICE) if imgs.dim() == 4 else imgs[0].to(T.DEVICE)
            pre = model._encode_prefix(imgs).cpu()                       # (V,197,1024)
            lat_s = lat[0] if isinstance(lat, (list, tuple)) else lat
            hv = hvf[0] if hvf.dim() > 1 else hvf
            out.append({'prefix': pre, 'hvf': hv, 'lat': lat_s})
    return out


def eye_severity_weight(hvf, lat):
    vi = T.valid_indices_od if lat.startswith('OD') else T.valid_indices_os
    v = np.asarray(hvf)[vi]; v = v[v < T.MASKED_VALUE_THRESHOLD]
    if len(v) == 0:
        return 1.0
    return max(1.0 + T.WEIGHT_SCALE * (T.MAX_DB - v.mean()) / T.MAX_DB, 1.0)


def val_metrics(model, cache):
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for item in cache:
            pre = item['prefix'].to(T.DEVICE)
            pred = model.forward_from_prefix(pre, [item['lat']], average_multi=True).cpu().numpy()[0]
            vi = T.valid_indices_od if item['lat'].startswith('OD') else T.valid_indices_os
            t = np.asarray(item['hvf'], float)[vi]; t[t >= T.MASKED_VALUE_THRESHOLD] = np.nan
            preds.append(pred.astype(np.float64)); trues.append(t)
    return preds, trues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--out-tag', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lora-rank', type=int, default=8)
    ap.add_argument('--lora-blocks', type=int, default=4)
    ap.add_argument('--lora-alpha', type=int, default=16)
    ap.add_argument('--lora-dropout', type=float, default=0.1)
    ap.add_argument('--lora-lr', type=float, default=2e-4)
    ap.add_argument('--head-lr', type=float, default=8e-4)
    ap.add_argument('--val-every', type=int, default=2)
    a = ap.parse_args()

    model = T.PerPointVFModel(T.base_model, global_head=True, lora=True, lora_rank=a.lora_rank,
                              lora_blocks=a.lora_blocks, lora_alpha=a.lora_alpha,
                              lora_dropout=a.lora_dropout, copy_encoder=False).to(T.DEVICE)
    n_frozen = len(model.encoder.blocks) - model._grad_blocks

    print(f"Caching frozen prefix (blocks[:{n_frozen}]) — train …", flush=True)
    train_cache = cache_prefix(model, a.train_json)
    print(f"  {len(train_cache)} train eyes cached. Val …", flush=True)
    val_cache = cache_prefix(model, a.val_json)
    print(f"  {len(val_cache)} val eyes cached.", flush=True)

    # Free the frozen prefix blocks (~1 GB) — only the LoRA suffix + decoder train from here.
    for i in range(n_frozen):
        model.encoder.blocks[i] = nn.Identity()
    gc.collect()
    if T.DEVICE.type == 'mps':
        torch.mps.empty_cache()
    print(f"  Freed {n_frozen} prefix blocks. Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    sector_weights = sector_weight_tensors(device=T.DEVICE, normalize=True)
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ('.A' in n or '.B' in n)]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not ('.A' in n or '.B' in n)]
    opt = torch.optim.AdamW([
        {'params': head_params, 'lr': a.head_lr, 'weight_decay': 5e-3},
        {'params': lora_params, 'lr': a.lora_lr, 'weight_decay': 1e-2}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)

    weights = [eye_severity_weight(it['hvf'], it['lat']) for it in train_cache]
    idx = list(range(len(train_cache)))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_cache), replacement=True)

    best = {'mae': float('inf')}
    out_best = os.path.join(AUTO, f"{a.out_tag}_best.pth")
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = list(sampler)
        ep_mae = ep_n = 0
        for s in range(0, len(order), a.batch_size):
            bidx = order[s:s + a.batch_size]
            pre = torch.cat([train_cache[i]['prefix'] for i in bidx]).to(T.DEVICE)   # (B,197,1024)
            hvf = torch.stack([torch.as_tensor(train_cache[i]['hvf'], dtype=torch.float32)
                               for i in bidx]).to(T.DEVICE)
            lat = [train_cache[i]['lat'] for i in bidx]
            pred = model.forward_from_prefix(pre, lat, average_multi=False)
            loss, mae, nv = T.compute_loss(pred, hvf, lat, epoch=epoch,
                                           attn_weights=model._last_attn_weights,
                                           sector_weights=sector_weights, sector_combine='sector_only')
            if nv > 0:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); ep_mae += mae * nv; ep_n += nv
            if T.DEVICE.type == 'mps' and (s // a.batch_size) % 8 == 0:
                torch.mps.empty_cache()
        sched.step()
        if epoch % a.val_every == 0 or epoch <= 3:
            vp, vt = val_metrics(model, val_cache)
            m = D.pooled_metrics(vp, vt)
            tag = ""
            if m['mae'] < best['mae']:
                best = {'mae': m['mae'], 'slope': m['slope'], 'corr': m['corr'], 'epoch': epoch}
                torch.save({'model': model.state_dict(), 'mae': m['mae'], 'slope': m['slope'],
                            'corr': m['corr'], 'epoch': epoch, 'use_dist': False,
                            'dist_blend': T.DIST_BLEND, 'mean_residual': False, 'global_head': True,
                            'lora': True, 'lora_rank': a.lora_rank, 'lora_blocks': a.lora_blocks,
                            'lora_alpha': a.lora_alpha, 'lora_dropout': a.lora_dropout}, out_best)
                tag = " ✓ saved"
            print(f"[E{epoch:02d}] train MAE {ep_mae/max(ep_n,1):.2f} | VAL MAE {m['mae']:.3f} "
                  f"slope {m['slope']:.3f} corr {m['corr']:.3f} σp/σt {m['sig_ratio']:.2f}{tag}", flush=True)
    print(f"BEST val: MAE {best['mae']:.3f} slope {best.get('slope',0):.3f} "
          f"corr {best.get('corr',0):.3f} @ep{best.get('epoch',0)} → {out_best}", flush=True)


if __name__ == "__main__":
    main()
