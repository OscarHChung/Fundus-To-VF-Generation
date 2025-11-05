# ==============================================
# train_decoder_with_grape_debug.py
# ==============================================
# Trains the decoder with UWHVF VF tests first, then fine-tunes using GRAPE paired data (fundus + VF)
# Adds full masking verification + print debugging.
# ==============================================

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
import matplotlib as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===========================
# Pretty print for VF output
# ===========================
def pretty_print_vf(vf_array, eye_side="OD", mask_value=100.0, decimals=1):
    """
    Nicely prints an 8x9 visual field array with alignment and masking.
    """
    vf_array = vf_array.reshape(8, 9)
    print(f"\n{'='*15} {eye_side} Prediction {'='*15}")

    for row in vf_array:
        row_str = ""
        for val in row:
            if val == mask_value:
                row_str += "   ·   "  # dot placeholder for masked
            else:
                row_str += f"{val:6.{decimals}f} "
        print(row_str)
    print("="*50)


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
# 3. Mask + Loss Functions
# ===========================
_mask_OD_np = np.array([
    [False, False, False,  True,  True,  True,  True, False, False],
    [False, False,  True,  True,  True,  True,  True,  True, False],
    [False,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  False,  True],
    [True,  True,  True,  True,  True,  True,  True,  False,  True],
    [False, True,  True,  True,  True,  True,  True,  True,  True],
    [False, False,  True,  True,  True,  True,  True,  True,  False],
    [False, False, False,  True,  True,  True,  True, False, False]
], dtype=bool)

_mask_OD_flat_np = _mask_OD_np.flatten()
_mask_OS_flat_np = _mask_OD_flat_np[::-1].copy()

# Now can convert to torch tensors
_mask_OD_flat = torch.tensor(_mask_OD_flat_np, dtype=torch.bool)
_mask_OS_flat = torch.tensor(_mask_OS_flat_np, dtype=torch.bool)

def masked_mse_loss_pretrain(pred, target, mask_value=100.0):
    mask = target != mask_value
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return ((pred[mask] - target[mask]) ** 2).mean()

def apply_mask_to_preds(preds, target, eye_sides, mask_value=100.0):
    preds_masked = preds.clone()
    for i in range(preds.size(0)):
        if eye_sides[i] == 'OD':
            mask = _mask_OD_flat.clone()
        else:
            mask = _mask_OS_flat.clone()

        if torch.all(target[i] == 0):
            print(f"[DEBUG] Detected inference mode for {eye_sides[i]}")
        else:
            target_mask = target[i] != 0
            mask = mask & target_mask

        preds_masked[i][~mask] = mask_value
        preds_example = preds_masked[i].view(8, 9).detach().cpu().numpy()
    return preds_masked


def masked_mse_loss(preds, target, eye_sides, mask_value=100.0):
    preds_masked = apply_mask_to_preds(preds, target, eye_sides, mask_value)
    valid = (target != mask_value)
    valid &= (preds_masked != mask_value)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device), preds_masked

    diff = preds_masked - target
    loss = (diff[valid] ** 2).mean()
    return loss, preds_masked

# ===========================
# 4. Pretrain decoder
# ===========================
def pretrain_decoder(vf_json, latent_dim=1024, epochs=1, batch_size=4, lr=1e-3, device='cpu'):
    dataset = VFOnlyDataset(vf_json)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    output_dim = 72

    decoder = VFDecoder(latent_dim=latent_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for vfs in loader:
            vfs = vfs.to(device)
            latent = torch.randn(vfs.size(0), latent_dim).to(device)
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
# 5. Fine-tune decoder
# ===========================
def finetune_decoder(decoder, dataset, encoder, epochs=1, batch_size=2, lr=1e-4, device='cpu'):
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
    from encoder.retfound_encoder import encoder  # ← your encoder module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 1024

    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")
    uwhvf_json = os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    paired_dataset = PairedDataset(grape_json, grape_fundus_dir, transform=transform)

    print("\n========== PRETRAINING ==========")
    decoder = pretrain_decoder(uwhvf_json, latent_dim=latent_dim, epochs=1, device=device)

    print("\n========== FINE-TUNING ==========")
    decoder = finetune_decoder(decoder, paired_dataset, encoder, epochs=1, batch_size=2, device=device)

    print("\n========== SAMPLE PREDICTION ==========")
    sample_img_path = os.path.join(grape_fundus_dir, "1_OD_1.jpg")
    sample_img = Image.open(sample_img_path).convert("RGB")
    sample_img = transform(sample_img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = encoder(sample_img)
        vf_pred = decoder(latent)

        # Apply masking test for both OD and OS
        vf_masked_OD = apply_mask_to_preds(vf_pred.clone(), torch.full_like(vf_pred, 100.0), ["OD"])
        vf_masked_OS = apply_mask_to_preds(vf_pred.clone(), torch.full_like(vf_pred, 100.0), ["OS"])

        print("\n[TEST PRINT] Raw prediction shape:", vf_pred.shape)
        print("[TEST PRINT] Masked OD shape:", vf_masked_OD.shape)
        print("[TEST PRINT] Masked OS shape:", vf_masked_OS.shape)

        # Pretty human-readable display
        pretty_print_vf(vf_masked_OD.cpu().numpy(), eye_side="OD")
        pretty_print_vf(vf_masked_OS.cpu().numpy(), eye_side="OS")
