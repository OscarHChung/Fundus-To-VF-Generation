#!/usr/bin/env python3
"""
Decoder Training Script - Masked MAE + Curriculum Learning + Early Stopping
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List

# ===========================
# Device
# ===========================
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS backend.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")
device = get_device()

# ===========================
# Mask indices (52 valid points)
# ===========================
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

mask_od_flat = mask_OD.flatten()
valid_indices_od: List[int] = [i for i, v in enumerate(mask_od_flat) if v]
mask_os_flat = mask_od_flat[::-1].copy()
valid_indices_os: List[int] = [i for i, v in enumerate(mask_os_flat) if v]
valid_idx_od_t = torch.tensor(valid_indices_od, dtype=torch.long, device=device)
valid_idx_os_t = torch.tensor(valid_indices_os, dtype=torch.long, device=device)

# ===========================
# Datasets
# ===========================
class VFOnlyDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vf_values = torch.tensor([v for row in item['hvf'] for v in row], dtype=torch.float32)
        laterality = item.get('Laterality', 'OD').strip().upper()
        laterality = 'OD' if not laterality.startswith(('OD', 'OS')) else laterality
        return vf_values, laterality

class PairedDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=(224, 224)):
        with open(json_path, 'r') as f:
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
        img_path = os.path.join(self.img_dir, item['FundusImage'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        vf_values = torch.tensor([v for row in item['hvf'] for v in row], dtype=torch.float32)
        laterality = item.get('Laterality', 'OD').strip().upper()
        laterality = 'OD' if not laterality.startswith(('OD', 'OS')) else laterality
        return img, vf_values, laterality

# ===========================
# Decoder
# ===========================
class Decoder52(nn.Module):
    def __init__(self, latent_dim=1024, out_dim=52, use_laterality=True):
        super().__init__()
        self.use_laterality = use_laterality
        input_dim = latent_dim + (1 if use_laterality else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, latent, laterality=None):
        x = latent
        if self.use_laterality:
            if laterality is None:
                raise ValueError("Laterality tensor required")
            if laterality.ndim == 1:
                laterality = laterality.unsqueeze(1)
            x = torch.cat([latent, laterality.float()], dim=1)
        return self.net(x)

# ===========================
# Helpers
# ===========================
def laterality_to_tensor(laterality_list):
    return torch.tensor([1 if str(l).upper().startswith('OD') else 0 for l in laterality_list],
                        dtype=torch.float32, device=device).unsqueeze(1)

def normalize_vf(vf_tensor):
    return vf_tensor.clamp(0, 40) / 40.0

def denormalize_vf(vf_tensor):
    return vf_tensor * 40.0

def masked_mae(pred, target, laterality_list):
    masked_targets = []
    masked_preds = []
    for i, lat in enumerate(laterality_list):
        idx = valid_indices_od if str(lat).upper().startswith('OD') else valid_indices_os
        masked_targets.append(target[i, idx])
        masked_preds.append(pred[i, :len(idx)])
    masked_targets = torch.stack(masked_targets)
    masked_preds = torch.stack(masked_preds)
    mae = torch.mean(torch.abs(masked_preds - masked_targets))
    return mae

# ===========================
# Curriculum Learning Scheduler
# ===========================
def smooth_vf(vf_tensor, factor=0.5):
    """Smooth VF map for curriculum learning (coarse to fine)."""
    return vf_tensor * factor + vf_tensor.mean(dim=1, keepdim=True) * (1 - factor)

# ===========================
# Training
# ===========================
def train_decoder(encoder_model, train_dataset, val_dataset,
                  save_path='decoder.pt', latent_dim=1024,
                  epochs=50, batch_size=8, lr=1e-4,
                  early_stop_patience=5, curriculum=True):

    decoder = Decoder52(latent_dim=latent_dim, out_dim=52, use_laterality=True).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.L1Loss(reduction='none')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_mae = float('inf')
    no_improve_epochs = 0

    encoder_model.eval()
    decoder.train()

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        total_points = 0

        for img, vf72, laterality in train_loader:
            img = img.to(device)
            vf72 = normalize_vf(vf72.to(device))
            lat_tensor = laterality_to_tensor(laterality)

            with torch.no_grad():
                latent = encoder_model(img)

            if curriculum:
                vf72_curr = smooth_vf(vf72, factor=max(0.3, 1.0 - epoch/epochs))
            else:
                vf72_curr = vf72

            pred = decoder(latent, laterality=lat_tensor)

            # Masked loss
            masked_targets = []
            masked_preds = []
            for i, lat in enumerate(laterality):
                idx = valid_indices_od if str(lat).upper().startswith('OD') else valid_indices_os
                masked_targets.append(vf72_curr[i, idx])
                masked_preds.append(pred[i, :len(idx)])
            masked_targets = torch.stack(masked_targets)
            masked_preds = torch.stack(masked_preds)

            loss = torch.mean(torch.abs(masked_preds - masked_targets))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * masked_targets.numel()
            total_points += masked_targets.numel()

        avg_train_loss = total_loss / total_points

        # Validation
        val_mae = evaluate_decoder(encoder_model, decoder, val_loader)
        if val_mae < best_val_mae - 1e-3:
            best_val_mae = val_mae
            torch.save(decoder.state_dict(), save_path)
            no_improve_epochs = 0
            new_best_str = f"  New best model saved with MAE {best_val_mae:.3f} dB"
        else:
            no_improve_epochs += 1
            new_best_str = ""

        print(f"[Epoch {epoch}] Train loss={avg_train_loss:.4f}, Validation MAE={val_mae:.3f} dB{new_best_str}")

        if no_improve_epochs >= early_stop_patience:
            print(f"Stopping early after {no_improve_epochs} epochs with marginal improvement.")
            break

    return decoder

# ===========================
# Evaluation
# ===========================
def evaluate_decoder(encoder_model, decoder_model, loader):
    decoder_model.eval()
    encoder_model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for img, vf72, laterality in loader:
            img = img.to(device)
            vf72 = normalize_vf(vf72.to(device))
            lat_tensor = laterality_to_tensor(laterality)

            latent = encoder_model(img)
            pred = decoder_model(latent, laterality=lat_tensor)

            batch_mae = masked_mae(pred, vf72, laterality)
            total_mae += batch_mae.item() * len(laterality)
            total_samples += len(laterality)

    if total_samples == 0:
        raise ValueError("No samples in evaluation loader!")

    avg_mae = total_mae / total_samples
    return avg_mae

# ===========================
# Main
# ===========================
if __name__ == '__main__':
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    fundus_dir = os.path.join(base_dir, 'fundus', 'grape_fundus_images')

    train_dataset = PairedDataset(os.path.join(base_dir, 'vf_tests', 'grape_train.json'), fundus_dir)
    val_dataset = PairedDataset(os.path.join(base_dir, 'vf_tests', 'grape_test.json'), fundus_dir)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from encoder.retfound_encoder import encoder as RetEncoder
    enc_model = RetEncoder.to(device)

    # Sanity check encoder output
    with torch.no_grad():
        s_img, _, _ = train_dataset[0]
        z = enc_model(s_img.unsqueeze(0).to(device))
        print('Encoder latent shape:', z.shape)

    decoder_model = train_decoder(enc_model, train_dataset, val_dataset,
                                  save_path=os.path.join(base_dir, 'decoder.pt'),
                                  epochs=50, batch_size=8, lr=1e-4,
                                  early_stop_patience=5,
                                  curriculum=True)

    final_mae = evaluate_decoder(enc_model, decoder_model, DataLoader(val_dataset, batch_size=8))
    print(f"Final Test Set Average MAE: {final_mae:.3f} dB")
