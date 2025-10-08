# This file trains the decoder with UWHVF VF tests first, then fine-tunes the model using GRAPE paired data

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ===========================
# 1. Datasets
# ===========================

# GRAPE
class PairedDataset(Dataset):
    def __init__(self, json_path, fundus_dir, transform=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.fundus_paths = [os.path.join(fundus_dir, entry["FundusImage"]) for entry in data]
        self.vf_arrays = [np.array(entry["hvf"], dtype=np.float32).flatten() for entry in data]
        self.transform = transform

    def __len__(self):
        return len(self.fundus_paths)

    def __getitem__(self, idx):
        img_path = self.fundus_paths[idx]
        vf = torch.tensor(self.vf_arrays[idx], dtype=torch.float32)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, vf
    
# UWHVF
class VFOnlyDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.vf_arrays = [np.array(entry["hvf"], dtype=np.float32).flatten() for entry in data]

    def __len__(self):
        return len(self.vf_arrays)

    def __getitem__(self, idx):
        vf = torch.tensor(self.vf_arrays[idx], dtype=torch.float32)
        return vf

# ===========================
# 2. Model
# ===========================

class VFDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=1024, output_dim=54):
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

# Example encoder: replace with pretrained RetFound
class DummyFundusEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224*224*3, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)

# ===========================
# 3. Loss Function
# ===========================
def masked_mse_loss(pred, target, mask_value=100.0):
    mask = target != mask_value
    return ((pred[mask] - target[mask]) ** 2).mean()

# ===========================
# 4. Pretrain Decoder on VF-only dataset
# ===========================
def pretrain_decoder(vf_json, latent_dim=512, epochs=10, batch_size=64, lr=1e-3, device=None):
    dataset = VFOnlyDataset(vf_json)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    output_dim = dataset[0].shape[0]
    decoder = VFDecoder(latent_dim=latent_dim, output_dim=output_dim)
    decoder = decoder.to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for vfs in loader:
            vfs = vfs.to(device)
            # Use random latent vectors for pretraining
            latent = torch.randn(vfs.size(0), latent_dim).to(device)
            preds = decoder(latent)
            loss = masked_mse_loss(preds, vfs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * vfs.size(0)
        epoch_loss /= len(dataset)
        print(f"[Pretrain] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return decoder

# ===========================
# 5. Fine-tune decoder with paired dataset
# ===========================
def finetune_decoder(decoder, paired_json, fundus_dir, encoder=None, epochs=20, batch_size=16, lr=1e-4, device=None):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    dataset = PairedDataset(paired_json, fundus_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    decoder = decoder.to(device)
    if encoder is None:
        encoder = DummyFundusEncoder().to(device)
    encoder.eval()  # freeze encoder if pretrained

    optimizer = optim.Adam(decoder.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, vfs in loader:
            imgs, vfs = imgs.to(device), vfs.to(device)
            with torch.no_grad():
                latent = encoder(imgs)
            preds = decoder(latent)
            loss = masked_mse_loss(preds, vfs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
        epoch_loss /= len(dataset)
        print(f"[Finetune] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return decoder


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 512

    # Paths
    uhwvf_json = ""
    grape_json = ""
    grape_fundus_dir = ""

    # 1) Pretrain decoder on UWHVF (VF test only)
    decoder = pretrain_decoder(uhwvf_json, latent_dim=latent_dim, epochs=10, device=device)

    # 2) Fine-tune decoder on GRAPE paired dataset after UWHVF
    encoder = DummyFundusEncoder(latent_dim=latent_dim).to(device)  # replace with RetFound
    decoder = finetune_decoder(decoder, grape_json, grape_fundus_dir, encoder=encoder, epochs=20, device=device)

    # 3) Make prediction
    '''from torchvision.transforms.functional import to_tensor
    sample_img_path = os.path.join(grape_fundus_dir, "1_OD_1.jpg")
    sample_img = Image.open(sample_img_path).convert("RGB")
    sample_img = transforms.Resize((224,224))(sample_img)
    sample_img = to_tensor(sample_img).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = encoder(sample_img)
        vf_pred = decoder(latent)
    print("Predicted VF:", vf_pred)'''
