#!/usr/bin/env python3
"""
latent_signal_diagnostics.py

Standalone diagnostics to check how much VF signal is present in the RETFound encoder latents.

Outputs:
 - ridge_mae (denormalized to dB)
 - per-VF-point mean |corr| (saved)
 - saves latents/targets to decoder/diagnostics/
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# -------------------------
# Project-root path fix (so imports work)
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# -------------------------
# Recreate PairedDataset (match your decoder.py)
# -------------------------
import json
from torchvision import transforms
from PIL import Image
from typing import List
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=(224, 224)):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["FundusImage"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        vf_values = torch.tensor([v for row in item["hvf"] for v in row], dtype=torch.float32)
        laterality = item.get("Laterality", "OD").strip().upper()
        laterality = "OD" if not laterality.startswith(("OD", "OS")) else laterality
        return img, vf_values, laterality

# -------------------------
# Mask indices (same mask you used)
# -------------------------
mask_OD = np.array([
    [False, False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, True, True, False],
    [False, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, False, True],
    [True, True, True, True, True, True, True, False, True],
    [False, True, True, True, True, True, True, True, True],
    [False, False, True, True, True, True, True, True, False],
    [False, False, False, True, True, True, True, False, False]
], dtype=bool)

mask_flat = mask_OD.flatten()
valid_indices_od: List[int] = [i for i, v in enumerate(mask_flat) if v]
valid_indices_os: List[int] = [i for i, v in enumerate(mask_flat[::-1]) if v]

# -------------------------
# Device
# -------------------------
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", DEVICE)

# -------------------------
# Safe encoder loading
# -------------------------
# Attempt to import encoder export from your file.
# The encoder file historically defines either:
#  - encoder = RetFoundEncoderWrapper(base_model)  (an instance)
#  - or RetFoundEncoderWrapper class + base_model
ENCODER_MODULE = "encoder.retfound_encoder"
try:
    enc_mod = __import__(ENCODER_MODULE, fromlist=["*"])
except Exception as e:
    raise RuntimeError(f"Failed to import '{ENCODER_MODULE}': {e}")

# Try various exported names
enc_obj = None
if hasattr(enc_mod, "encoder"):
    enc_obj = getattr(enc_mod, "encoder")     # likely an instance
elif hasattr(enc_mod, "RetFoundEncoderWrapper") and hasattr(enc_mod, "base_model"):
    Wrapper = getattr(enc_mod, "RetFoundEncoderWrapper")
    base_model = getattr(enc_mod, "base_model")
    print("Detected RetFoundEncoderWrapper class + base_model - instantiating wrapper.")
    enc_obj = Wrapper(base_model)
elif hasattr(enc_mod, "RetFoundEncoderWrapper"):
    # If only class is present, try to instantiate without args (rare)
    Wrapper = getattr(enc_mod, "RetFoundEncoderWrapper")
    try:
        enc_obj = Wrapper()
    except Exception as e:
        raise RuntimeError("Encoder module exports a class but cannot be auto-instantiated. "
                           "Please export an 'encoder' instance in encoder/retfound_encoder.py or adjust this script.") from e
else:
    raise RuntimeError("Couldn't find encoder export in encoder.retfound_encoder. "
                       "Make sure the module exports either 'encoder' (instance) or 'RetFoundEncoderWrapper' + 'base_model'.")

# Move encoder to device (this will move underlying model parameters too)
if isinstance(enc_obj, nn.Module):
    enc_obj.to(DEVICE)
    enc_obj.eval()
else:
    raise RuntimeError("Loaded encoder object is not an nn.Module instance. Check encoder export.")

encoder = enc_obj
print("Encoder loaded and moved to device.")

# -------------------------
# Utility: normalize/denormalize VF
# -------------------------
def normalize_vf(vf_tensor):
    # vf_tensor: torch tensor with raw dB values [0..40+], return normalized 0..1
    return vf_tensor.clamp(0, 40) / 40.0

def denormalize_mae(norm_mae):
    # norm_mae is normalized (0..1), convert back to dB (0..40)
    return norm_mae * 40.0

# -------------------------
# Paths & datasets
# -------------------------
BASE_DIR = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
FUNDUS_DIR = os.path.join(BASE_DIR, "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "vf_tests", "grape_train.json")
VAL_JSON   = os.path.join(BASE_DIR, "vf_tests", "grape_test.json")

train_ds = PairedDataset(TRAIN_JSON, FUNDUS_DIR)
val_ds   = PairedDataset(VAL_JSON, FUNDUS_DIR)

print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

# -------------------------
# Extraction helper
# -------------------------
def extract_latents_and_targets(dataset, batch_size=8, max_samples=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    lat_list = []
    tgt_list = []

    with torch.no_grad():
        for imgs, vf72, laterality in loader:
            imgs = imgs.to(DEVICE)
            # get latent (encoder expects input; wrapper handles device inside forward)
            latent = encoder(imgs)      # [B, latent_dim]
            latent = latent.detach().cpu().numpy()

            # build 52-point targets per sample using mask depending on laterality
            bs = vf72.shape[0]
            targets_batch = np.zeros((bs, len(valid_indices_od)), dtype=np.float32)
            for i in range(bs):
                l = laterality[i]
                vf = vf72[i]  # on CPU
                if str(l).upper().startswith("OD"):
                    vals = vf.numpy()[valid_indices_od]
                else:
                    vals = vf.numpy()[valid_indices_os]
                # normalize to 0..1
                vals = np.clip(vals, 0, 40) / 40.0
                targets_batch[i] = vals

            lat_list.append(latent)
            tgt_list.append(targets_batch)

            if max_samples is not None and sum(len(x) for x in lat_list) >= max_samples:
                break

    X = np.concatenate(lat_list, axis=0)
    Y = np.concatenate(tgt_list, axis=0)
    return X, Y

# -------------------------
# Run extraction
# -------------------------
t0 = time.time()
print("Extracting train latents...")
X_train, Y_train = extract_latents_and_targets(train_ds, batch_size=8)
print("Extracting val latents...")
X_val, Y_val = extract_latents_and_targets(val_ds, batch_size=8)
print(f"Extraction done in {time.time()-t0:.1f}s")
print("Shapes:", X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

# -------------------------
# Save raw latents/targets (optional)
# -------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUT_DIR, "Y_val.npy"), Y_val)
print("Saved latents/targets to", OUT_DIR)

# -------------------------
# Linear Ridge baseline
# -------------------------
print("\nFitting Ridge regression baseline (multi-output)...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
pred_val = ridge.predict(X_val)
mae_norm = mean_absolute_error(Y_val, pred_val)  # normalized (0..1)
mae_db = denormalize_mae(mae_norm)
print(f"Ridge MAE (normalized) = {mae_norm:.4f} → {mae_db:.3f} dB")

# -------------------------
# Correlation analysis
# -------------------------
print("\nComputing per-latent <-> per-VF-point correlations (this may take a while)...")
# compute correlation between each latent dim and each VF point across train set
n_lat = X_train.shape[1]
n_pts = Y_train.shape[1]
corr = np.zeros((n_lat, n_pts), dtype=np.float32)

# zero-mean normalize for correlation
Xc = X_train - X_train.mean(axis=0, keepdims=True)
Yc = Y_train - Y_train.mean(axis=0, keepdims=True)
Xs = Xc.std(axis=0, keepdims=True)
Ys = Yc.std(axis=0, keepdims=True)

# avoid divide by zero
Xs[Xs == 0] = 1.0
Ys[Ys == 0] = 1.0

for j in range(n_pts):
    # vectorized computation for latent correlations with VF point j
    y = Yc[:, j:j+1]  # (N,1)
    # correlation for all latent dims: cov(X, y)/ (stdX * stdy)
    cov = (Xc * y).mean(axis=0)
    corr[:, j] = cov / (Xs.flatten() * Ys[0, j])

# per VF point mean absolute correlation
mean_abs_corr_per_point = np.mean(np.abs(corr), axis=0)
np.save(os.path.join(OUT_DIR, "latent_vf_corr.npy"), corr)
np.save(os.path.join(OUT_DIR, "mean_abs_corr_per_point.npy"), mean_abs_corr_per_point)

print("Saved correlation matrices to diagnostics/")
print("\nPer-VF-point mean |corr| (sorted low->high):")
order = np.argsort(mean_abs_corr_per_point)
for idx in order:
    print(f" point {idx:02d}: mean|corr| = {mean_abs_corr_per_point[idx]:.4f}")

print("\nSummary statistics:")
print(" Overall mean |corr|:", float(mean_abs_corr_per_point.mean()))
print(" Min |corr|:", float(mean_abs_corr_per_point.min()))
print(" Max |corr|:", float(mean_abs_corr_per_point.max()))

# -------------------------
# Interpretation heuristic
# -------------------------
print("\n=== INTERPRETATION ===")
print(f"Ridge baseline MAE: {mae_db:.3f} dB (lower is better)")
mean_corr = float(mean_abs_corr_per_point.mean())
if mae_db <= 4.5:
    print(" → Linear probe suggests encoder latents already contain strong VF signal. Focus on decoder architecture.")
elif mae_db <= 6.0:
    print(" → Mixed result: try decoder improvements + consider selective encoder fine-tuning.")
else:
    print(" → Encoder likely bottleneck; consider fine-tuning encoder or changing encoder backbone.")

print(f"Average |corr| across VF points = {mean_corr:.4f}")
if mean_corr < 0.08:
    print(" → Low average correlations; encoder might not encode fine-grained per-location VF info.")
elif mean_corr < 0.15:
    print(" → Moderate correlations; decoder improvements might help but consider encoder tuning.")
else:
    print(" → Good correlations; decoder likely can be improved to reach sub-4 dB with the right training.")

print("\nDiagnostics and arrays have been saved in:", OUT_DIR)
