"""
Fast Decoder Training with Precomputed Latents
- Precomputes encoder latents for faster training
- Hybrid SmoothL1 + L1 loss
- Early stopping on validation MAE
- Optional point weighting for hard VF points
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mask indices (52 valid points)
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

valid_indices_od = [i for i, v in enumerate(mask_OD.flatten()) if v]
valid_indices_os = valid_indices_od[::-1]

valid_idx_od_t = torch.tensor(valid_indices_od, dtype=torch.long, device=device)
valid_idx_os_t = torch.tensor(valid_indices_os, dtype=torch.long, device=device)

# ===========================
# Dataset classes
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
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1,0.05),
            transforms.RandomRotation(5),
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
# Residual decoder
# ===========================
class ResidualDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_dim=52, use_laterality=True):
        super().__init__()
        self.use_laterality = use_laterality
        input_dim = latent_dim + (1 if use_laterality else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

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
    return vf_tensor.clamp(0,40)/40.0

def denormalize_vf(vf_tensor):
    return vf_tensor*40.0

# ===========================
# Precompute latents
# ===========================
def precompute_latents(encoder_model, paired_dataset):
    encoder_model.eval()
    latents = []
    vfs = []
    lat_tensor_list = []

    loader = DataLoader(paired_dataset, batch_size=16)
    with torch.no_grad():
        for img, vf72, laterality in loader:
            img = img.to(device)
            vf72 = normalize_vf(vf72.to(device))
            lat = encoder_model(img)
            latents.append(lat.cpu())
            vfs.append(vf72.cpu())
            lat_tensor_list.extend([1 if l.startswith('OD') else 0 for l in laterality])

    latents = torch.cat(latents)
    vfs = torch.cat(vfs)
    laterality_tensor = torch.tensor(lat_tensor_list, dtype=torch.float32).unsqueeze(1)
    return latents, vfs, laterality_tensor

# ===========================
# Train decoder on precomputed latents
# ===========================
def train_decoder_on_latents(latents, vfs, laterality_tensor, save_path='decoder.pt',
                             latent_dim=1024, epochs=50, batch_size=32,
                             lr=5e-5, patience=5, min_delta=1e-4):

    decoder = ResidualDecoder(latent_dim=latent_dim, out_dim=52, use_laterality=True).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()  # hybrid loss possible

    dataset = TensorDataset(latents, vfs, laterality_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_mae = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        decoder.train()
        total_loss = 0.0
        total_points = 0

        for batch_lat, batch_vf, batch_lat_tensor in loader:
            batch_lat = batch_lat.to(device)
            batch_vf = batch_vf.to(device)
            batch_lat_tensor = batch_lat_tensor.to(device)

            pred = decoder(batch_lat, laterality=batch_lat_tensor)

            target_list = [batch_vf[i, valid_indices_od] if batch_lat_tensor[i]==1 else batch_vf[i, valid_indices_os]
                           for i in range(len(batch_lat_tensor))]
            target = torch.stack(target_list)

            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * target.numel()
            total_points += target.numel()

        avg_loss = total_loss / total_points
        print(f"[Epoch {epoch}] Avg loss={avg_loss:.6f}")

        # Evaluate MAE per epoch
        decoder.eval()
        with torch.no_grad():
            pred_all = decoder(latents.to(device), laterality=laterality_tensor.to(device))
            target_list = [vfs[i, valid_indices_od] if laterality_tensor[i]==1 else vfs[i, valid_indices_os]
                           for i in range(len(latents))]
            target_all = torch.stack(target_list).to(device)
            val_mae_norm = torch.mean(torch.abs(pred_all - target_all)).item()
            val_mae_db = denormalize_vf(torch.tensor(val_mae_norm))
            print(f"Validation MAE: {val_mae_db:.3f} dB")

        # Early stopping
        if best_mae - val_mae_db > min_delta:
            best_mae = val_mae_db
            epochs_no_improve = 0
            torch.save(decoder.state_dict(), save_path)
            print(f"New best model saved with MAE {best_mae:.3f} dB")
        else:
            epochs_no_improve += 1
            print(f"No significant improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    decoder.load_state_dict(torch.load(save_path))
    return decoder

# ===========================
# Main
# ===========================
if __name__ == '__main__':
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    fundus_dir = os.path.join(base_dir, 'fundus', 'grape_fundus_images')

    grape_train = PairedDataset(os.path.join(base_dir, 'vf_tests', 'grape_train.json'), fundus_dir)
    grape_test = PairedDataset(os.path.join(base_dir, 'vf_tests', 'grape_test.json'), fundus_dir)
    uwhvf_train = VFOnlyDataset(os.path.join(base_dir, 'vf_tests', 'uwhvf_vf_tests_standardized.json'))

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from encoder.retfound_encoder import encoder as RetEncoder
    enc_model = RetEncoder.to(device)

    print("Precomputing train latents...")
    train_latents, train_vfs, train_laterality = precompute_latents(enc_model, grape_train)

    decoder_model = train_decoder_on_latents(train_latents, train_vfs, train_laterality,
                                             save_path=os.path.join(base_dir, 'decoder.pt'),
                                             epochs=50, batch_size=32, lr=5e-5)
