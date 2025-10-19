# train_decoder_with_grape.py
# Trains the decoder with UWHVF VF tests first, then fine-tunes using GRAPE paired data (fundus + VF)

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===========================
# 1. Datasets
# ===========================

class PairedDataset(Dataset):
    def __init__(self, json_path, fundus_dir, transform=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.fundus_paths = [os.path.join(fundus_dir, entry["FundusImage"]) for entry in data]
        self.vf_arrays = [torch.tensor(np.array(entry["hvf"], dtype=float).flatten(), dtype=torch.float32) for entry in data]
        self.eye_sides = [entry["Laterality"] for entry in data]
        self.transform = transform

    def __len__(self):
        return len(self.fundus_paths)

    def __getitem__(self, idx):
        img_path = self.fundus_paths[idx]
        vf = self.vf_arrays[idx]
        eye_side = self.eye_sides[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, vf, eye_side


class VFOnlyDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.vf_arrays = [torch.tensor(np.array(entry["hvf"], dtype=float).flatten(), dtype=torch.float32) for entry in data]

    def __len__(self):
        return len(self.vf_arrays)

    def __getitem__(self, idx):
        return self.vf_arrays[idx]

# ===========================
# 2. Decoder Model
# ===========================
class VFDecoder(nn.Module):
    def __init__(self, latent_dim=1024, hidden_dim=1024, output_dim=72):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, latent):
        return self.model(latent)

# ===========================
# 3. Loss Functions
# ===========================
def masked_mse_loss_pretrain(pred, target, mask_value=100.0):
    mask = target != mask_value
    return ((pred[mask] - target[mask]) ** 2).mean()

def apply_mask_to_preds(preds, target, eye_sides, mask_value=100.0):
    """
    preds: [batch, n_points]
    target: same shape
    eye_sides: list of 'OD'/'OS'
    """
    preds = preds.clone()
    target = target.clone()
    batch_size, n_points = preds.shape

    # Example anatomical mask: first half = OD, second half = OS
    half = n_points // 2
    mask_OD = torch.ones(half, dtype=torch.bool, device=preds.device)
    mask_OS = mask_OD.flip(dims=[0])

    for i in range(batch_size):
        if eye_sides[i] == "OD":
            mask = torch.cat([mask_OD, mask_OD])  # repeat if needed
        else:
            mask = torch.cat([mask_OS, mask_OS])
        mask = mask & (target[i] != mask_value)  # also mask out 100 points

        preds[i][~mask] = mask_value  # enforce masked positions
    return preds

def masked_mse_loss(preds, target, eye_sides, mask_value=100.0):
    preds_masked = apply_mask_to_preds(preds, target, eye_sides, mask_value)
    loss = ((preds_masked - target)**2)
    valid = target != mask_value
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device), preds_masked
    return loss[valid].mean(), preds_masked

# ===========================
# 4. Pretrain decoder on UWHVF
# ===========================
def pretrain_decoder(vf_json, latent_dim=1024, epochs=10, batch_size=64, lr=1e-3, device='cpu'):
    dataset = VFOnlyDataset(vf_json)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    output_dim = 72

    decoder = VFDecoder(latent_dim=latent_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for vfs in loader:
            vfs = vfs.to(device)
            latent = torch.randn(vfs.size(0), latent_dim).to(device)  # random latent for pretraining
            preds = decoder(latent)
            loss = masked_mse_loss_pretrain(preds, vfs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * vfs.size(0)

        epoch_loss /= len(dataset)
        print(f"[Pretrain] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return decoder

# ===========================
# 5. Fine-tune decoder on GRAPE
# ===========================
def finetune_decoder(decoder, dataset, encoder, epochs=20, batch_size=16, lr=1e-4, device='cpu'):
    decoder.train()
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for fundus_imgs, vfs, eye_sides in tqdm(loader, desc=f"[Finetune] Epoch {epoch+1}/{epochs}"):
            fundus_imgs = fundus_imgs.to(device)
            vfs = vfs.to(device)

            optimizer.zero_grad()
            latent = encoder(fundus_imgs)
            preds = decoder(latent)

            # masked loss + enforce mask
            loss, preds_masked = masked_mse_loss(preds, vfs, eye_sides)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Finetune] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return decoder

# ===========================
# 6. Run pipeline
# ===========================
if __name__ == "__main__":
    from encoder.retfound_encoder import encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 1024

    # Paths
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")
    uwhvf_json = os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json")

    # Transform for encoder input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Create GRAPE dataset
    paired_dataset = PairedDataset(grape_json, grape_fundus_dir, transform=transform)

    # 1) Pretrain decoder
    decoder = pretrain_decoder(uwhvf_json, latent_dim=latent_dim, epochs=10, device=device)

    # 2) Fine-tune decoder
    decoder = finetune_decoder(decoder, paired_dataset, encoder, epochs=20, batch_size=16, device=device)

    # 3) Example prediction
    sample_img_path = os.path.join(grape_fundus_dir, "1_OD_1.jpg")
    sample_img = Image.open(sample_img_path).convert("RGB")
    sample_img = transform(sample_img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = encoder(sample_img)
        vf_pred = decoder(latent)

        # Apply mask using actual target if available, else keep all as valid
        # Here we just keep all as valid for demo
        vf_pred_masked = vf_pred.clone()  # no target available, keep predictions as-is

    print("Predicted VF:", vf_pred_masked)
