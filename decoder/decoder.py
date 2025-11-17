#!/usr/bin/env python3
"""
Decoder training script that loads encoder/nbest_encoder.pth (from Option B)
and trains a residual decoder on the 52 masked VF points. Prints MAE in dB.
Saves: best_decoder_nbest.pth and decoder_calibration.npz
"""

import os
import sys
import json
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# ---------- Paths & config ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "../data")
FUNDUS_DIR = os.path.join(BASE_DIR, "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "vf_tests", "grape_test.json")

ENCODER_CHECK = os.path.join(os.path.dirname(CURRENT_DIR), "encoder", "nbest_encoder.pth")
if not os.path.exists(ENCODER_CHECK):
    # fallback try encoder folder relative
    ENCODER_CHECK = os.path.join(CURRENT_DIR, "../encoder/nbest_encoder.pth")
BEST_DECODER_SAVE = os.path.join(CURRENT_DIR, "best_decoder_nbest.pth")
CALIB_SAVE = os.path.join(CURRENT_DIR, "decoder_calibration.npz")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
PATIENCE = 6
MIN_DELTA = 1e-3

# ----------------------
# Mask of 52 valid points
# ----------------------
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

valid_idx_od_t = torch.tensor(valid_indices_od, dtype=torch.long, device=DEVICE)
valid_idx_os_t = torch.tensor(valid_indices_os, dtype=torch.long, device=DEVICE)

# ---------- Dataset (returns full 72 hvf) ----------
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, transform=None):
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
        if self.transform:
            img = self.transform(img)
        hvf = np.array(it['hvf'], dtype=np.float32).reshape(-1)  # 72
        later = it.get('Laterality', it.get('laterality', 'OD')).strip().upper()
        if not later.startswith(('OD','OS')):
            later = 'OD'
        return img, torch.tensor(hvf, dtype=torch.float32), later

# ---------- Load RETFound model constructor ----------
RETFOUND_DIR = os.path.join(CURRENT_DIR, "../encoder/RETFound_MAE")
sys.path.insert(0, RETFOUND_DIR)
try:
    from models_mae import mae_vit_large_patch16_dec512d8b
except Exception as e:
    raise RuntimeError(f"Failed to import RETFound models: {e}")

# ---------- Encoder wrapper ----------
class RetFoundEncoderWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        latent = self.model.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        return latent

# ---------- Strong residual decoder ----------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

class StrongDecoder(nn.Module):
    def __init__(self, latent_dim=1024, hidden=1024, out=52, nblocks=4, dropout=0.2):
        super().__init__()
        self.front = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(nblocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.SiLU(),
            nn.Dropout(max(0.1, dropout/2)),
            nn.Linear(hidden//2, out)
        )
    def forward(self, x):
        x = self.front(x)
        x = self.res_blocks(x)
        x = self.head(x)
        return x

# ---------- Eval & calibration ----------
def evaluate(encoder, decoder, loader, device, apply_calib=None):
    encoder.eval(); decoder.eval()
    total_abs = 0.0
    total_n = 0
    preds_all = []
    targs_all = []
    with torch.no_grad():
        for imgs, hvf72, later in loader:
            imgs = imgs.to(device)
            hvf72 = hvf72.to(device)
            lat = encoder(imgs)
            preds72 = decoder(lat)  # predicted 52??? we'll select below

            # select masked indices per laterality
            batch_preds = []
            batch_targs = []
            for i, lat_idx in enumerate(later):
                if str(lat_idx).upper().startswith('OD'):
                    idxes = valid_indices_od
                else:
                    idxes = valid_indices_os
                targ52 = hvf72[i, idxes].cpu().numpy()
                pred52 = preds72[i, :].cpu().numpy()
                batch_targs.append(targ52)
                batch_preds.append(pred52)

            batch_preds = np.stack(batch_preds, axis=0)
            batch_targs = np.stack(batch_targs, axis=0)

            if apply_calib is not None:
                a, b = apply_calib
                batch_preds = batch_preds * a[np.newaxis, :] + b[np.newaxis, :]

            total_abs += np.abs(batch_preds - batch_targs).sum()
            total_n += batch_targs.size

            preds_all.append(batch_preds)
            targs_all.append(batch_targs)

    mae = total_abs / total_n
    preds_all = np.concatenate(preds_all, axis=0)
    targs_all = np.concatenate(targs_all, axis=0)
    return mae, preds_all, targs_all

def fit_per_point_linear(preds, targs):
    # preds/targs: [N,52]
    N, D = preds.shape
    a = np.zeros((D,), dtype=np.float32)
    b = np.zeros((D,), dtype=np.float32)
    for j in range(D):
        X = np.stack([preds[:, j], np.ones(N)], axis=1)
        y = targs[:, j]
        sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a[j] = sol[0]
        b[j] = sol[1]
    return a, b

# ---------- Training loop ----------
def train_decoder(encoder_model, decoder, train_loader, val_loader, device,
                  epochs=EPOCHS, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA,
                  finetune_top_k=0):
    # encoder_model: base RETFound class with loaded weights (not wrapper)
    encoder = RetFoundEncoderWrapper(encoder_model).to(device)
    decoder = decoder.to(device)
    # freeze encoder by default
    for p in encoder_model.parameters():
        p.requires_grad = False

    if finetune_top_k > 0 and hasattr(encoder_model, "blocks"):
        for blk in encoder_model.blocks[-finetune_top_k:]:
            for p in blk.parameters():
                p.requires_grad = True
        print(f"Unfrozen top {finetune_top_k} blocks for tiny finetune.")

    optim_params = [{'params': decoder.parameters(), 'lr': lr}]
    if finetune_top_k > 0:
        enc_params = [p for p in encoder_model.parameters() if p.requires_grad]
        optim_params.append({'params': enc_params, 'lr': lr * 0.01})
    optimizer = torch.optim.AdamW(optim_params, weight_decay=1e-5)
    criterion = nn.L1Loss(reduction='mean')

    best_val = float('inf')
    no_improve = 0

    for epoch in range(1, epochs+1):
        decoder.train()
        if finetune_top_k > 0:
            encoder.train()
        else:
            encoder.eval()

        train_abs = 0.0
        train_n = 0
        train_loss_sum = 0.0
        for imgs, hvf72, later in tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=False):
            imgs = imgs.to(device)
            hvf72 = hvf72.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(finetune_top_k > 0):
                lat = encoder(imgs)
            preds52 = decoder(lat)  # shape [B,52]

            # build target 52 per sample and compute loss
            targ_list = []
            for i, l in enumerate(later):
                if str(l).upper().startswith('OD'):
                    idxes = valid_indices_od
                else:
                    idxes = valid_indices_os
                targ52 = hvf72[i, idxes]
                targ_list.append(targ52)
            target52 = torch.stack(targ_list).to(device)  # [B,52]

            loss = criterion(preds52, target52)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)
            train_abs += torch.sum(torch.abs(preds52.detach() - target52)).item()
            train_n += target52.numel()

        train_mae = train_abs / train_n
        train_loss_mean = train_loss_sum / len(train_loader.dataset)

        # Validation
        decoder.eval()
        encoder.eval()
        val_abs = 0.0
        val_n = 0
        val_loss_sum = 0.0
        all_val_preds = []
        all_val_targs = []
        with torch.no_grad():
            for imgs, hvf72, later in val_loader:
                imgs = imgs.to(device)
                hvf72 = hvf72.to(device)
                lat = encoder(imgs)
                preds52 = decoder(lat)

                targ_list = []
                for i, l in enumerate(later):
                    if str(l).upper().startswith('OD'):
                        idxes = valid_indices_od
                    else:
                        idxes = valid_indices_os
                    targ_list.append(hvf72[i, idxes])
                target52 = torch.stack(targ_list).to(device)

                val_loss_sum += nn.L1Loss(reduction='sum')(preds52, target52).item()
                val_abs += torch.sum(torch.abs(preds52 - target52)).item()
                val_n += target52.numel()

                all_val_preds.append(preds52.cpu().numpy())
                all_val_targs.append(target52.cpu().numpy())

        val_mae = val_abs / val_n
        val_loss_mean = val_loss_sum / len(val_loader.dataset)

        print(f"[Epoch {epoch}] Train MAE={train_mae:.3f} dB, Val MAE={val_mae:.3f} dB")

        # Early stopping & checkpoint
        if val_mae + min_delta < best_val:
            best_val = val_mae
            no_improve = 0
            # save states
            torch.save({
                "decoder_state": decoder.state_dict(),
                "encoder_state": encoder_model.state_dict(),
                "val_mae": best_val
            }, BEST_DECODER_SAVE)
            print(f"  Saved new best decoder -> {BEST_DECODER_SAVE} (Val MAE={best_val:.3f})")
        else:
            no_improve += 1
            print(f"  No improve count: {no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print("Stopping early due to no improvement.")
            break

    # Post-training calibration on training set
    print("Computing per-point linear calibration on training set...")
    # gather train preds and targets
    encoder.eval(); decoder.eval()
    preds_list = []; targ_list = []
    with torch.no_grad():
        for imgs, hvf72, later in train_loader:
            imgs = imgs.to(device); hvf72 = hvf72.to(device)
            lat = encoder(imgs)
            p52 = decoder(lat).cpu().numpy()
            # build true target
            t52_list = []
            for i, l in enumerate(later):
                idxes = valid_indices_od if str(l).upper().startswith('OD') else valid_indices_os
                t52_list.append(hvf72[i, idxes].cpu().numpy())
            t52 = np.stack(t52_list, axis=0)
            preds_list.append(p52); targ_list.append(t52)
    preds_all = np.concatenate(preds_list, axis=0)
    targs_all = np.concatenate(targ_list, axis=0)
    a_vec, b_vec = fit_per_point_linear(preds_all, targs_all)
    np.savez(CALIB_SAVE, a=a_vec, b=b_vec)
    print(f"Saved calibration to {CALIB_SAVE}")

    # evaluate final saved best decoder with calibration
    print("Evaluating best saved decoder (with calibration) on validation set...")
    ckpt = torch.load(BEST_DECODER_SAVE, map_location="cpu")
    decoder.load_state_dict(ckpt["decoder_state"])
    mae_before, _, _ = evaluate(encoder, decoder, val_loader, DEVICE, apply_calib=None)
    mae_after, _, _ = evaluate(encoder, decoder, val_loader, DEVICE, apply_calib=(a_vec, b_vec))
    print(f"Val MAE before calib: {mae_before:.3f} dB; after calib: {mae_after:.3f} dB")
    return best_val

# ---------- Main ----------
if __name__ == "__main__":
    print("Using device:", DEVICE)
    if not os.path.exists(ENCODER_CHECK):
        raise FileNotFoundError(f"Encoder checkpoint not found at {ENCODER_CHECK}; run encoder script first.")

    # dataset transform matching encoder
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = PairedDataset(TRAIN_JSON, FUNDUS_DIR, transform)
    val_ds = PairedDataset(VAL_JSON, FUNDUS_DIR, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # load RETFound constructor
    mae_ctor = mae_vit_large_patch16_dec512d8b  # imported earlier via RETFOUND_DIR insertion
    base_model = mae_ctor()
    # load encoder checkpoint created by encoder script
    enc_ckpt = torch.load(ENCODER_CHECK, map_location="cpu")
    if 'encoder_state' in enc_ckpt:
        base_model.load_state_dict(enc_ckpt['encoder_state'], strict=False)
    else:
        # if saved differently, try load whole dict
        base_model.load_state_dict(enc_ckpt, strict=False)
    base_model.to(DEVICE)

    # build decoder
    decoder = StrongDecoder(latent_dim=1024, hidden=1024, out=52, nblocks=4, dropout=0.25)

    # train
    train_decoder(base_model, decoder, train_loader, val_loader, DEVICE,
                  epochs=EPOCHS, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA,
                  finetune_top_k=0)  # set to 1 or 2 to tiny fine-tune encoder blocks
