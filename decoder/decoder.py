# ==============================================
# train_decoder_with_grape_debug_savejson.py
# ==============================================
# Trains the decoder with UWHVF VF tests first, then fine-tunes using GRAPE paired data (fundus + VF)
# Now saves actual vs prediction results for each image into JSON
# and plots MAE distribution across all VF test points.
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
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===========================
# Pretty print for VF output
# ===========================
def pretty_print_vf(vf_array, eye_side="OD", mask_value=100.0, decimals=1):
    vf_array = vf_array.reshape(8, 9)
    print(f"\n{'='*15} {eye_side} {'='*15}")
    for row in vf_array:
        row_str = ""
        for val in row:
            if val == mask_value:
                row_str += "   ·   "
            else:
                row_str += f"{val:6.{decimals}f} "
        print(row_str)
    print("=" * 50)


# ===========================
# 1. Datasets
# ===========================
class PairedDataset(Dataset):
    def __init__(self, json_path, fundus_dir, transform=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.entries = data
        self.fundus_paths = [os.path.join(fundus_dir, entry["FundusImage"]) for entry in data]
        self.vf_arrays = [torch.tensor(np.array(entry["hvf"], dtype=float).flatten(), dtype=torch.float32) for entry in data]
        self.eye_sides = [entry["Laterality"] for entry in data]
        self.ids = [entry.get("id", os.path.basename(entry["FundusImage"])) for entry in data]
        self.transform = transform

    def __len__(self):
        return len(self.fundus_paths)

    def __getitem__(self, idx):
        img_path = self.fundus_paths[idx]
        vf = self.vf_arrays[idx]
        eye_side = self.eye_sides[idx]
        img_id = self.ids[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, vf, eye_side, img_id


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
        preds_masked[i][~mask] = mask_value
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
        for fundus_imgs, vfs, eye_sides, _ in tqdm(loader, desc=f"[Finetune] Epoch {epoch+1}/{epochs}"):
            fundus_imgs = fundus_imgs.to(device)
            vfs = vfs.to(device)
            optimizer.zero_grad()
            latent = encoder(fundus_imgs)
            preds = decoder(latent)
            loss, _ = masked_mse_loss(preds, vfs, eye_sides)
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
    from encoder.retfound_encoder import encoder  # your encoder

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

    print("\n========== SAVING PREDICTIONS ==========")
    results = []
    mae_all_points = []

    decoder.eval()
    with torch.no_grad():
        for img, vf_true, eye_side, img_id in tqdm(paired_dataset, desc="Evaluating"):
            img_tensor = img.unsqueeze(0).to(device)
            vf_true_np = vf_true.numpy()
            latent = encoder(img_tensor)
            vf_pred = decoder(latent)

            vf_pred_masked = apply_mask_to_preds(vf_pred.clone(), vf_true.unsqueeze(0), [eye_side])
            vf_pred_np = vf_pred_masked.cpu().numpy().flatten()

            # MAE per valid point
            valid_mask = (vf_true_np != 100.0) & (vf_pred_np != 100.0)
            mae_points = np.abs(vf_true_np[valid_mask] - vf_pred_np[valid_mask])
            mae_all_points.extend(mae_points.tolist())
            mae_mean = float(np.mean(mae_points)) if len(mae_points) > 0 else None

            results.append({
                "id": img_id,
                "eye_side": eye_side,
                "actual_vf": vf_true_np.tolist(),
                "predicted_vf": vf_pred_np.tolist(),
                "mae": mae_mean
            })

    # Save results JSON
    output_json = os.path.join(base_dir, "predictions_vs_actuals.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved predictions and actuals to {output_json}")

    # ===========================
    # Plot MAE distribution
    # ===========================
    mae_all_points = np.array(mae_all_points)
    plt.figure(figsize=(8, 6))
    plt.hist(mae_all_points, bins=30)
    plt.title("Distribution of MAE Across All VF Points")
    plt.xlabel("Mean Absolute Error (dB)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
