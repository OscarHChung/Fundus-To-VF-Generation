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


def cache_prefix(model, json_path, transform=None, n_passes=1, batch=16, denoised_lookup=None,
                 rnfl_lookup=None):
    """Return list of {prefix:(1,197,1024) cpu, hvf:(72,), lat:str} — one entry per (eye, view).
    Uses train-mode single-image samples + BATCHED encoder forward (fast). n_passes>1 with a random
    `transform` caches augmented views. denoised_lookup swaps TRAIN targets (Method B) — pass it only
    for the train cache; val stays RAW. rnfl_lookup (M2) attaches a z-scored 5-value RNFL target +
    mask per view (train cache only); DataLoader(shuffle=False) keeps entry k aligned to ds.samples[k]."""
    transform = transform or T.val_transform
    old_noise = T.LABEL_NOISE_STD; T.LABEL_NOISE_STD = 0.0     # cached targets stay RAW (no noise)
    rnfl_eyes = rnfl_lookup['eyes'] if rnfl_lookup else None
    if rnfl_lookup:
        rmean = np.asarray(rnfl_lookup['norm']['rnfl_mean'], dtype=np.float32)
        rstd  = np.asarray(rnfl_lookup['norm']['rnfl_std'],  dtype=np.float32)
    try:
        ds = T.MultiImageDataset(json_path, T.FUNDUS_DIR, transform, mode='train',
                                 denoised_lookup=denoised_lookup)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
        out = []
        model.eval()
        with torch.no_grad():
            for _ in range(n_passes):
                si = 0
                for imgs, hvf, lat in loader:
                    pre = model._encode_prefix(imgs.to(T.DEVICE)).half().cpu()   # fp16 (B,197,1024)
                    for b in range(pre.shape[0]):
                        entry = {'prefix': pre[b:b + 1], 'hvf': hvf[b], 'lat': lat[b]}
                        if rnfl_eyes is not None:
                            s = ds.samples[si]
                            rec = rnfl_eyes.get(f"{int(s['patient_id'])}_{s['laterality']}")
                            if rec is not None:
                                entry['rnfl'] = (np.asarray(rec['rnfl'], dtype=np.float32) - rmean) / rstd
                                entry['rnfl_mask'] = 1.0
                            else:
                                entry['rnfl'] = np.zeros(5, dtype=np.float32)
                                entry['rnfl_mask'] = 0.0
                        out.append(entry)
                        si += 1
    finally:
        T.LABEL_NOISE_STD = old_noise
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
            pre = item['prefix'].to(T.DEVICE).float()
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
    ap.add_argument('--warm-start', default=None,
                    help="checkpoint to warm-start the DECODER from (e.g. long_global_f0_best.pth)")
    ap.add_argument('--select', choices=['mae', 'mae_slope'], default='mae_slope',
                    help="checkpoint selection: min MAE, or min (MAE - 0.5*slope) to value slope")
    ap.add_argument('--aug-views', type=int, default=1,
                    help="cache this many augmented prefix views per TRAIN eye (>1 restores "
                         "augmentation regularization; val stays deterministic)")
    ap.add_argument('--denoised', action='store_true',
                    help="Method B: use per-eye trend-denoised TRAIN targets (val stays RAW)")
    # M1 — first-class severity (eye-mean / MD) head + de-shrink loss (default OFF ≡ long_global).
    ap.add_argument('--severity-head', action='store_true',
                    help="M1: add a CLS→MD head that replaces the field eye-mean, de-shrunk by a "
                         "batch-CCC loss on eye-means (the decisive between-eye lever)")
    ap.add_argument('--severity-blend', type=float, default=1.0,
                    help="mix the field mean: blend·severity_head + (1-blend)·emergent mean")
    ap.add_argument('--severity-weight', type=float, default=0.5,
                    help="λ on the eye-mean Huber (severity supervision)")
    ap.add_argument('--severity-ccc', type=float, default=0.5,
                    help="λ on the batch-CCC de-shrink term over eye-means")
    ap.add_argument('--severity-eye-scale', type=float, default=2.0,
                    help="eye-level severity reweight scale (focus moderate+severe eyes)")
    # M3 — variance reduction: EMA of the trainable (LoRA + decoder) params.
    ap.add_argument('--ema', action='store_true',
                    help="M3: keep an EMA of trainable params; eval + save the EMA weights")
    ap.add_argument('--ema-decay', type=float, default=0.998)
    # M2 — fundus→RNFL structural-surrogate aux head (TRAIN-ONLY; fundus-only at inference).
    ap.add_argument('--rnfl-aux', action='store_true',
                    help="M2: add a train-only CLS→RNFL[Mean,S,N,I,T] aux head to sharpen features")
    ap.add_argument('--rnfl-weight', type=float, default=0.3,
                    help="λ on the masked RNFL aux Huber (eyes without RNFL are masked out)")
    ap.add_argument('--rnfl-lookup', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vf_tests",
        "grape_rnfl_lookup.json"))
    a = ap.parse_args()
    denoised_lookup = None
    if a.denoised:
        with open(T.DENOISED_TARGETS_PATH) as f:
            denoised_lookup = json.load(f)
    rnfl_lookup = None
    if a.rnfl_aux:
        with open(a.rnfl_lookup) as f:
            rnfl_lookup = json.load(f)

    model = T.PerPointVFModel(T.base_model, global_head=True, lora=True, lora_rank=a.lora_rank,
                              lora_blocks=a.lora_blocks, lora_alpha=a.lora_alpha,
                              lora_dropout=a.lora_dropout, copy_encoder=False,
                              severity_head=a.severity_head,
                              severity_blend=a.severity_blend,
                              rnfl_aux=a.rnfl_aux).to(T.DEVICE)
    severity_cfg = None
    if a.severity_head:
        severity_cfg = dict(weight=a.severity_weight, ccc=a.severity_ccc,
                            eye_scale=a.severity_eye_scale)
    if a.warm_start:
        ck = torch.load(a.warm_start, map_location='cpu', weights_only=False)
        sd = ck.get('model', ck.get('model_state_dict', ck))
        # load only matching-shape keys (the decoder; encoder qkv differs due to LoRA wrapping)
        own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"Warm-started {len(keep)} tensors from {os.path.basename(a.warm_start)} "
              f"(decoder + frozen encoder; LoRA A/B fresh).", flush=True)
        # Free the ~1.2 GB warm-start checkpoint (encoder tensors included) — otherwise ck/sd/keep
        # stay resident in CPU RAM for the whole run. Pure memory hygiene; no effect on training.
        del ck, sd, own, keep
        gc.collect()
    n_frozen = len(model.encoder.blocks) - model._grad_blocks

    tr_tfm = T.train_transform if a.aug_views > 1 else T.val_transform
    print(f"Caching frozen prefix (blocks[:{n_frozen}]) — train ×{a.aug_views} views"
          f"{' +denoised' if a.denoised else ''} …", flush=True)
    train_cache = cache_prefix(model, a.train_json, tr_tfm, n_passes=a.aug_views,
                               denoised_lookup=denoised_lookup, rnfl_lookup=rnfl_lookup)
    if a.rnfl_aux:
        _nr = sum(int(e.get('rnfl_mask', 0.0)) for e in train_cache)
        print(f"  M2: RNFL aux targets on {_nr}/{len(train_cache)} train views", flush=True)
    print(f"  {len(train_cache)} train views cached. Val …", flush=True)
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
    # M3 — EMA over the trainable (LoRA A/B + decoder) params; the frozen prefix is already
    # freed so WeightEMA picks up exactly the trainable set. Evaluated + saved instead of the raw
    # weights to cut plateau variance (the fold-0→pooled overfit gap).
    ema = T.WeightEMA([model], decay=a.ema_decay) if a.ema else None
    if ema:
        print(f"  M3: EMA ON (decay={a.ema_decay}) over {len(ema.params)} trainable tensors",
              flush=True)

    weights = [eye_severity_weight(it['hvf'], it['lat']) for it in train_cache]
    idx = list(range(len(train_cache)))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_cache), replacement=True)

    best = {'mae': float('inf'), 'score': float('inf')}
    score_of = (lambda m: m['mae']) if a.select == 'mae' else \
               (lambda m: m['mae'] - 0.5 * m['slope'])   # value slope alongside MAE
    out_best = os.path.join(AUTO, f"{a.out_tag}_best.pth")
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = list(sampler)
        ep_mae = ep_n = 0
        for s in range(0, len(order), a.batch_size):
            bidx = order[s:s + a.batch_size]
            pre = torch.cat([train_cache[i]['prefix'] for i in bidx]).to(T.DEVICE).float()  # (B,197,1024)
            hvf = torch.stack([torch.as_tensor(train_cache[i]['hvf'], dtype=torch.float32)
                               for i in bidx]).to(T.DEVICE)
            lat = [train_cache[i]['lat'] for i in bidx]
            pred = model.forward_from_prefix(pre, lat, average_multi=False)
            loss, mae, nv = T.compute_loss(pred, hvf, lat, epoch=epoch,
                                           attn_weights=model._last_attn_weights,
                                           sector_weights=sector_weights, sector_combine='sector_only',
                                           severity_pred=model._last_severity,
                                           severity_cfg=severity_cfg)
            if a.rnfl_aux and model._last_rnfl is not None:
                rt = torch.stack([torch.as_tensor(train_cache[i]['rnfl'], dtype=torch.float32)
                                  for i in bidx]).to(T.DEVICE)                       # (B,5) z-scored
                rm = torch.tensor([train_cache[i]['rnfl_mask'] for i in bidx],
                                  dtype=torch.float32, device=T.DEVICE)              # (B,)
                aux = torch.nn.functional.smooth_l1_loss(
                    model._last_rnfl, rt, reduction='none').mean(dim=1)              # (B,)
                aux_loss = (aux * rm).sum() / rm.sum().clamp_min(1.0)               # masked mean
                loss = loss + a.rnfl_weight * aux_loss
            if nv > 0:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); ep_mae += mae * nv; ep_n += nv
                if ema: ema.update()
            if T.DEVICE.type == 'mps' and (s // a.batch_size) % 2 == 0:
                torch.mps.empty_cache()
        gc.collect()
        if T.DEVICE.type == 'mps':
            torch.mps.empty_cache()
        sched.step()
        if epoch % a.val_every == 0 or epoch <= 3:
            if ema: ema.apply_to()      # evaluate + save the EMA weights (M3)
            vp, vt = val_metrics(model, val_cache)
            m = D.pooled_metrics(vp, vt)
            tag = ""
            if score_of(m) < best['score']:
                best = {'mae': m['mae'], 'slope': m['slope'], 'corr': m['corr'], 'epoch': epoch,
                        'score': score_of(m)}
                torch.save({'model': model.state_dict(), 'mae': m['mae'], 'slope': m['slope'],
                            'corr': m['corr'], 'epoch': epoch, 'use_dist': False,
                            'dist_blend': T.DIST_BLEND, 'mean_residual': False, 'global_head': True,
                            'lora': True, 'lora_rank': a.lora_rank, 'lora_blocks': a.lora_blocks,
                            'lora_alpha': a.lora_alpha, 'lora_dropout': a.lora_dropout,
                            'severity_head': a.severity_head,
                            'severity_blend': a.severity_blend,
                            'rnfl_aux': a.rnfl_aux}, out_best)
                tag = " ✓ saved"
            if ema: ema.restore()       # back to raw weights for continued training
            print(f"[E{epoch:02d}] train MAE {ep_mae/max(ep_n,1):.2f} | VAL MAE {m['mae']:.3f} "
                  f"slope {m['slope']:.3f} corr {m['corr']:.3f} σp/σt {m['sig_ratio']:.2f}{tag}", flush=True)
    print(f"BEST val: MAE {best['mae']:.3f} slope {best.get('slope',0):.3f} "
          f"corr {best.get('corr',0):.3f} @ep{best.get('epoch',0)} → {out_best}", flush=True)


if __name__ == "__main__":
    main()
