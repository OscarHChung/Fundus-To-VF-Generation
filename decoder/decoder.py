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
    def __init__(self, latent_dim=1024, hidden_dim=1024, output_dim=54):
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

def masked_mse_loss(pred, target, eye_sides):
    """Masked MSE"""
    total_loss = 0.0
    count = 0

    mask_OD = torch.tensor([
        [False, False, False, True, True, True, True, False, False],
        [False, False, True, True, True, True, True, True, False],
        [False, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, False, True],
        [True, True, True, True, True, True, True, False, True],
        [False, True, True, True, True, True, True, True, True],
        [False, False, True, True, True, True, True, True, False],
        [False, False, False, True, True, True, True, False, False],
    ], dtype=torch.bool, device=device)

    mask_OS = torch.flip(mask_OD, dims=[1])

    for i in range(pred.shape[0]):
        eye_mask = mask_OD if eye_sides[i] == 'OD' else mask_OS
        p = pred[i].reshape(8, 9)
        t = target[i].reshape(8, 9)
        mask_valid = (t != 100) & eye_mask
        if mask_valid.sum() > 0:
            loss = ((p[mask_valid] - t[mask_valid]) ** 2).mean()
            total_loss += loss
            count += 1

    return total_loss / max(count, 1)

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

            loss = masked_mse_loss(preds, vfs, eye_sides)
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
    print("Predicted VF:", vf_pred)
