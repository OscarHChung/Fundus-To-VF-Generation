#!/usr/bin/env python3
"""
Improved Decoder training script (full).
Uses encoder checkpoint produced by encoder fine-tuning (nbest_encoder.pth).
- Mask-aware VF targets (52 points)
- Richer latent: CLS token + mean pooled token concat
- Residual MLP decoder
- Normalization / denormalization (0-40 dB)
- Per-epoch printing (MAE in dB), early stopping, LR scheduler
- Final linear per-point calibration saved
"""

import os
import sys
import json
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# safe loading helper for RETFound weights containing argparse.Namespace
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])

# ------------------------
# Config / paths
# ------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "..", "data")
FUNDUS_DIR = os.path.join(BASE_DIR, "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "vf_tests", "grape_test.json")

# Encoder checkpoint produced earlier (nbest encoder)
ENCODER_CHECK = os.path.join(CURRENT_DIR, "..", "encoder", "nbest_encoder.pth")
# fallback options
if not os.path.exists(ENCODER_CHECK):
    ENCODER_CHECK = os.path.join(CURRENT_DIR, "../encoder/best_encoder_only.pth")
BEST_DECODER_SAVE = os.path.join(CURRENT_DIR, "best_decoder_nbest.pth")
CALIB_SAVE = os.path.join(CURRENT_DIR, "decoder_calibration.npz")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
PATIENCE = 8
MIN_DELTA = 1e-4
FINETUNE_TOP_K = 0  # set to >0 to tiny-finetune top K transformer blocks
WEIGHT_DECAY = 1e-5
PRINT_EVERY = 1

# ------------------------
# Mask of 52 valid points
# ------------------------
mask_OD = np.array([
    [False, False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, False, False, True,  True,  True,  True,  False, False]
], dtype=bool)

mask_od_flat = mask_OD.flatten()
valid_indices_od: List[int] = [i for i, v in enumerate(mask_od_flat) if v]
mask_os_flat = mask_od_flat[::-1].copy()
valid_indices_os: List[int] = [i for i, v in enumerate(mask_os_flat) if v]

VALID_COUNT = len(valid_indices_od)  # should be 52

# ------------------------
# Dataset
# ------------------------
class PairedDataset(Dataset):
    def __init__(self, json_path: str, img_dir: str, transform=None):
        with open(json_path, 'r') as f:
            self.items = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = os.path.join(self.img_dir, it['FundusImage'])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        hvf = np.array(it['hvf'], dtype=np.float32).reshape(-1)  # 72
        later = it.get('Laterality', it.get('laterality', 'OD')).strip().upper()
        if not later.startswith(('OD','OS')):
            later = 'OD'
        return img, torch.tensor(hvf, dtype=torch.float32), later

# ------------------------
# Normalization helpers (0-40 dB -> 0-1)
# ------------------------
def normalize_vf(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(0.0, 40.0) / 40.0

def denormalize_vf_scalar(norm: float) -> float:
    return float(norm * 40.0)

def denormalize_vf_tensor(t: torch.Tensor) -> torch.Tensor:
    return t * 40.0

# ------------------------
# RETFound import
# ------------------------
RETFOUND_DIR = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_MAE")
sys.path.insert(0, RETFOUND_DIR)
try:
    from models_mae import mae_vit_large_patch16_dec512d8b
except Exception as e:
    raise RuntimeError(f"Could not import RETFound constructor from {RETFOUND_DIR}: {e}")

# ------------------------
# Encoder wrapper (returns richer latent)
# ------------------------
class RetFoundEncoderWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        # forward_encoder returns tokens [B, num_patches, dim] (and maybe other outputs)
        tokens = self.model.forward_encoder(x, mask_ratio=0.0)[0]
        # tokens shape: [B, num_patches, dim] or [B, dim] if already collapsed
        if tokens.dim() == 3:
            cls = tokens[:, 0, :]                # [B, dim]
            avg = tokens.mean(dim=1)            # [B, dim]
            latent = torch.cat([cls, avg], dim=1)  # [B, 2*dim]
        else:
            # guard: if encoder already returns [B, dim]
            latent = tokens
        return latent

# ------------------------
# Decoder (residual MLP)
# ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class StrongDecoder(nn.Module):
    def __init__(self, latent_dim=2048, hidden=1024, out=VALID_COUNT, nblocks=3, dropout=0.2):
        super().__init__()
        self.front = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.res = nn.Sequential(*[ResidualBlock(hidden) for _ in range(nblocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(max(0.1, dropout/2)),
            nn.Linear(hidden // 2, out)
        )
    def forward(self, x):
        x = self.front(x)
        x = self.res(x)
        x = self.head(x)
        return x

# ------------------------
# Utilities: evaluate / calibration
# ------------------------
def evaluate_epoch(encoder_wrapped, decoder, loader, device, apply_calib: Tuple[np.ndarray,np.ndarray]=None):
    encoder_wrapped.eval(); decoder.eval()
    total_abs = 0.0
    total_n = 0
    preds_all = []
    targs_all = []
    with torch.no_grad():
        for imgs, hvf72, later in loader:
            imgs = imgs.to(device)
            hvf72 = hvf72.to(device)
            lat = encoder_wrapped(imgs)  # [B, 2*dim] or [B, dim]
            preds52 = decoder(lat)       # [B, 52]

            # build masked targets per sample
            batch_preds = []
            batch_targs = []
            for i, l in enumerate(later):
                idxes = valid_indices_od if str(l).upper().startswith('OD') else valid_indices_os
                targ52 = hvf72[i, idxes]            # raw dB
                pred52 = preds52[i, :].cpu().numpy()
                batch_preds.append(pred52)
                batch_targs.append(targ52.cpu().numpy())

            batch_preds = np.stack(batch_preds, axis=0)  # [B,52]
            batch_targs = np.stack(batch_targs, axis=0)  # [B,52]

            # apply calibration if requested (a,b per-point)
            if apply_calib is not None:
                a,b = apply_calib
                batch_preds = batch_preds * a[np.newaxis,:] + b[np.newaxis,:]

            total_abs += np.abs(batch_preds - batch_targs).sum()
            total_n += batch_targs.size

            preds_all.append(batch_preds)
            targs_all.append(batch_targs)

    mae = total_abs / total_n
    preds_all = np.concatenate(preds_all, axis=0)
    targs_all = np.concatenate(targs_all, axis=0)
    return mae, preds_all, targs_all

def fit_per_point_linear(preds: np.ndarray, targs: np.ndarray):
    # preds/targs: [N,52] -> fit y = a*x + b per point
    N, D = preds.shape
    a = np.zeros((D,), dtype=np.float32)
    b = np.zeros((D,), dtype=np.float32)
    for j in range(D):
        X = np.stack([preds[:, j], np.ones(N)], axis=1)
        y = targs[:, j]
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        a[j] = sol[0]; b[j] = sol[1]
    return a, b

# ------------------------
# Training loop
# ------------------------
def train_decoder(base_model, decoder, train_loader, val_loader, device,
                  epochs=EPOCHS, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA,
                  finetune_top_k=FINETUNE_TOP_K):
    # base_model: RETFound base model instance (not wrapper)
    encoder_wrapped = RetFoundEncoderWrapper(base_model).to(device)
    decoder = decoder.to(device)

    # freeze entire encoder by default
    for p in base_model.parameters():
        p.requires_grad = False

    # optionally unfreeze top K blocks for tiny finetuning
    if finetune_top_k > 0 and hasattr(base_model, "blocks"):
        for blk in base_model.blocks[-finetune_top_k:]:
            for p in blk.parameters():
                p.requires_grad = True
        print(f"Unfrozen top {finetune_top_k} blocks for small finetune.")

    # optimizer: decoder params + (optional) encoder finetune params
    optim_groups = [{"params": decoder.parameters(), "lr": lr}]
    if finetune_top_k > 0:
        enc_params = [p for p in base_model.parameters() if p.requires_grad]
        optim_groups.append({"params": enc_params, "lr": lr * 0.01})
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=WEIGHT_DECAY)

    criterion = nn.L1Loss(reduction='mean')  # normalized MAE (0-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val = float('inf')
    no_improve = 0
    best_epoch = -1

    for epoch in range(1, epochs+1):
        decoder.train()
        if finetune_top_k > 0:
            encoder_wrapped.train()
        else:
            encoder_wrapped.eval()

        train_sum_loss = 0.0
        train_abs = 0.0
        train_n = 0

        for imgs, hvf72, later in tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=False):
            imgs = imgs.to(device)
            hvf72 = hvf72.to(device)
            # build masked target normalized [0-1]
            targ_list = []
            for i, l in enumerate(later):
                idxes = valid_indices_od if str(l).upper().startswith('OD') else valid_indices_os
                targ72 = hvf72[i, idxes]    # dB
                targ_norm = normalize_vf(targ72).to(device)
                targ_list.append(targ_norm)
            target52 = torch.stack(targ_list).to(device)  # [B,52]

            optimizer.zero_grad()
            with torch.set_grad_enabled(finetune_top_k > 0):
                lat = encoder_wrapped(imgs)  # [B, dim] (if concatenated, dim=2048)
            preds_norm = decoder(lat)  # normalized predictions in [0-1] range (not clamped)
            loss = criterion(preds_norm, target52)
            loss.backward()
            optimizer.step()

            train_sum_loss += loss.item() * imgs.size(0)
            train_abs += torch.sum(torch.abs(preds_norm.detach() - target52)).item()
            train_n += target52.numel()

        train_loss = train_sum_loss / len(train_loader.dataset)
        train_mae_norm = train_abs / train_n
        train_mae_db = denormalize_vf_scalar(train_mae_norm)

        # validation
        decoder.eval()
        encoder_wrapped.eval()
        val_sum_loss = 0.0
        val_abs = 0.0
        val_n = 0
        all_preds = []; all_targs = []
        with torch.no_grad():
            for imgs, hvf72, later in val_loader:
                imgs = imgs.to(device); hvf72 = hvf72.to(device)
                lat = encoder_wrapped(imgs)
                preds_norm = decoder(lat)
                # build target
                tlist = []
                for i, l in enumerate(later):
                    idxes = valid_indices_od if str(l).upper().startswith('OD') else valid_indices_os
                    targ72 = hvf72[i, idxes]
                    tnorm = normalize_vf(targ72).to(device)
                    tlist.append(tnorm)
                t52 = torch.stack(tlist).to(device)
                val_sum_loss += nn.L1Loss(reduction='sum')(preds_norm, t52).item()
                val_abs += torch.sum(torch.abs(preds_norm - t52)).item()
                val_n += t52.numel()

                all_preds.append(preds_norm.cpu().numpy())
                all_targs.append(t52.cpu().numpy())

        val_loss_mean = val_sum_loss / len(val_loader.dataset)
        val_mae_norm = val_abs / val_n
        val_mae_db = denormalize_vf_scalar(val_mae_norm)

        # Pearson between flattened preds (denorm) and targs (denorm)
        all_preds_np = np.concatenate(all_preds, axis=0) if len(all_preds)>0 else np.zeros((0,VALID_COUNT))
        all_targs_np = np.concatenate(all_targs, axis=0) if len(all_targs)>0 else np.zeros((0,VALID_COUNT))
        # convert to dB for correlation (denormalize)
        if all_preds_np.size>0:
            flat_p = (all_preds_np.flatten() * 40.0)
            flat_t = (all_targs_np.flatten() * 40.0)
            pearson = float(np.corrcoef(flat_p, flat_t)[0,1]) if flat_p.std()>0 and flat_t.std()>0 else 0.0
        else:
            pearson = 0.0

        # scheduler step (use val loss mean)
        scheduler.step(val_loss_mean)

        if epoch % PRINT_EVERY == 0:
            print(f"[Epoch {epoch}] Train Loss={train_loss:.4f}, Train MAE={train_mae_db:.3f} dB, Val Loss={val_loss_mean:.4f}, Val MAE={val_mae_db:.3f} dB, Pearson={pearson:.3f}")

        # checkpointing (best by val_mae_db)
        if val_mae_db + min_delta < best_val:
            best_val = val_mae_db
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "decoder_state": decoder.state_dict(),
                "encoder_state": base_model.state_dict(),
                "val_mae_db": best_val,
                "epoch": epoch
            }, BEST_DECODER_SAVE)
            print(f"  Saved new best decoder -> {BEST_DECODER_SAVE} (Val MAE={best_val:.3f} dB)")
        else:
            no_improve += 1
            print(f"  No improve count: {no_improve}/{patience}")

        if no_improve >= patience:
            print("Stopping early due to no improvement.")
            break

    # post-train linear per-point calibration (fit on training set)
    print("Fitting per-point linear calibration on training set (denormalized values)...")
    preds_list = []; targs_list = []
    encoder_wrapped.eval(); decoder.eval()
    with torch.no_grad():
        for imgs, hvf72, later in train_loader:
            imgs = imgs.to(device); hvf72 = hvf72.to(device)
            lat = encoder_wrapped(imgs)
            pnorm = decoder(lat).cpu().numpy()   # [B,52] normalized
            # build denorm true
            tlist = []
            for i, l in enumerate(later):
                idxes = valid_indices_od if str(l).upper().startswith('OD') else valid_indices_os
                t52 = hvf72[i, idxes].cpu().numpy()  # dB
                tlist.append(t52)
            t52 = np.stack(tlist, axis=0)
            preds_list.append(pnorm * 40.0)
            targs_list.append(t52)
    preds_all = np.concatenate(preds_list, axis=0)
    targs_all = np.concatenate(targs_list, axis=0)
    a_vec, b_vec = fit_per_point_linear(preds_all, targs_all)
    np.savez(CALIB_SAVE, a=a_vec, b=b_vec)
    print(f"Saved per-point calibration -> {CALIB_SAVE}")

    # evaluate saved best decoder with and without calibration
    print("Evaluating best saved decoder (with calibration)...")
    ckpt = torch.load(BEST_DECODER_SAVE, map_location="cpu")
    decoder.load_state_dict(ckpt["decoder_state"])
    mae_before, _, _ = evaluate_epoch(encoder_wrapped, decoder, val_loader, device, apply_calib=None)
    mae_after, _, _ = evaluate_epoch(encoder_wrapped, decoder, val_loader, device, apply_calib=(a_vec, b_vec))
    print(f"Val MAE before calib: {mae_before:.3f} dB; after calib: {mae_after:.3f} dB")
    return best_val

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print("Using device:", DEVICE)
    # transforms matching encoder
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = PairedDataset(TRAIN_JSON, FUNDUS_DIR, transform)
    val_ds = PairedDataset(VAL_JSON, FUNDUS_DIR, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # instantiate RETFound base model and load encoder checkpoint
    base_model = mae_vit_large_patch16_dec512d8b()
    # load checkpoint (safe_globals already set at top)
    if not os.path.exists(ENCODER_CHECK):
        raise FileNotFoundError(f"Encoder checkpoint not found at {ENCODER_CHECK}. Run encoder script first.")
    ckpt = torch.load(ENCODER_CHECK, map_location="cpu")
    # ckpt may contain 'encoder_state' or whole base model state
    if isinstance(ckpt, dict) and 'encoder_state' in ckpt:
        base_model.load_state_dict(ckpt['encoder_state'], strict=False)
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        # some checkpoints saved under 'model'
        base_model.load_state_dict(ckpt['model'], strict=False)
    else:
        try:
            base_model.load_state_dict(ckpt, strict=False)
        except Exception:
            # fallback: if checkpoint contains base_model + head
            for k in ckpt.keys():
                if 'encoder' in k or 'model' in k:
                    try:
                        base_model.load_state_dict(ckpt[k], strict=False)
                        break
                    except Exception:
                        pass
            # if still not loaded, raise helpful error
    base_model.to(DEVICE)
    base_model.eval()

    # create decoder: latent_dim matches concatenated CLS+AVG => 2*1024
    latent_dim = 2048
    decoder = StrongDecoder(latent_dim=latent_dim, hidden=1024, out=VALID_COUNT, nblocks=4, dropout=0.25)

    # train
    train_decoder(base_model, decoder, train_loader, val_loader, DEVICE,
                  epochs=EPOCHS, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA,
                  finetune_top_k=FINETUNE_TOP_K)

    print("Done.")
