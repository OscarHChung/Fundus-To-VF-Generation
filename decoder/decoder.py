# ==============================================
# Two‑Phase Model (with k-fold cross validation)
# Phase 1: Decoder pre‑training on VF‑only dataset
# Phase 2: Fine‑tuning on paired fundus + VF dataset (with K‑fold cross‑validation)
# Saves predictions and MAE per fold and averages MAE
# ==============================================

import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoder.retfound_encoder import encoder

# ===========================
# Datasets
# ===========================
class VFOnlyDataset(Dataset):
    """Dataset for VF-only images (no fundus) used in decoder pretraining."""
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.vf_arrays = [torch.tensor(np.array(e["hvf"], dtype=float).flatten(), dtype=torch.float32) for e in data]
        self.eye_sides = [e.get("Laterality", "OD") for e in data]

    def __len__(self):
        return len(self.vf_arrays)

    def __getitem__(self, idx):
        return self.vf_arrays[idx], self.eye_sides[idx]


class PairedDataset(Dataset):
    """Dataset for paired fundus image + VF data."""
    def __init__(self, json_path, fundus_dir, transform=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.entries = data
        self.fundus_paths = [os.path.join(fundus_dir, e["FundusImage"]) for e in data]
        self.vf_arrays = [torch.tensor(np.array(e["hvf"], dtype=float).flatten(), dtype=torch.float32) for e in data]
        self.eye_sides = [e["Laterality"] for e in data]
        self.ids = [e.get("id", os.path.basename(e["FundusImage"])) for e in data]
        self.transform = transform

    def __len__(self):
        return len(self.fundus_paths)

    def __getitem__(self, idx):
        img = Image.open(self.fundus_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.vf_arrays[idx], self.eye_sides[idx], self.ids[idx]


# ===========================
# Decoder Model
# ===========================
class VFDecoder(nn.Module):
    def __init__(self, latent_dim=1024, hidden_dim=2048, output_dim=72, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ===========================
# Masking
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
])
_mask_OD_flat = torch.tensor(_mask_OD_np.flatten(), dtype=torch.bool)
_mask_OS_flat = torch.tensor(_mask_OD_np.flatten()[::-1].copy(), dtype=torch.bool)

def apply_mask(preds, eye_sides, mask_value=100.0):
    preds_masked = preds.clone()
    for i in range(preds.size(0)):
        mask = _mask_OD_flat if eye_sides[i] == 'OD' else _mask_OS_flat
        preds_masked[i][~mask] = mask_value
    return preds_masked

def masked_loss(preds, targets, eye_sides, mask_value=100.0):
    preds_masked = apply_mask(preds, eye_sides, mask_value)
    valid = (targets != mask_value)
    valid &= (preds_masked != mask_value)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    mse = ((preds_masked[valid] - targets[valid]) ** 2).mean()
    mae = torch.abs(preds_masked[valid] - targets[valid]).mean()
    # hybrid loss: combining MSE + MAE to balance large error penalty + typical error
    return 0.7 * mse + 0.3 * mae

# ===========================
# Stage 1: Pre‑train decoder on VF‑only
# ===========================
def pretrain_decoder(decoder, vf_loader, device, epochs=10, lr=1e-4):
    decoder.train()
    optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for vf_arr, eye_sides in tqdm(vf_loader, desc=f"Pretrain Epoch {epoch+1}/{epochs}"):
            vf_arr = vf_arr.to(device)
            batch_size = vf_arr.size(0)
            latent = torch.randn(batch_size, 1024, device=device)
            preds = decoder(latent)
            loss = loss_fn(preds, vf_arr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(vf_loader)
        print(f"Pretrain Epoch {epoch+1}: Loss={avg:.4f}")
    torch.save(decoder.state_dict(), "decoder_pretrained.pt")
    print("Saved pretrained decoder weights.")
    return decoder

# ===========================
# Stage 2: Fine‑tune on paired data with K‑Fold CV
# ===========================
def train_and_evaluate_kfold(dataset, encoder, latent_dim, device,
                             n_splits=5, epochs=15, lr=1e-4, batch_size=4):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold_idx+1}/{n_splits} ---")
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        decoder = VFDecoder(latent_dim=latent_dim).to(device)
        # load pretrained weights
        decoder.load_state_dict(torch.load("decoder_pretrained.pt"))
        
        optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            decoder.train()
            total_loss = 0.0
            for imgs, vfs, eye_sides, _ in tqdm(train_loader, desc=f"Train Fold {fold_idx+1} Epoch {epoch+1}/{epochs}"):
                imgs, vfs = imgs.to(device), vfs.to(device)
                latent = encoder(imgs)
                preds = decoder(latent)
                loss = masked_loss(preds, vfs, eye_sides)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train = total_loss / len(train_loader)

            # validation on test_subset each epoch
            decoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, vfs, eye_sides, _ in test_loader:
                    imgs, vfs = imgs.to(device), vfs.to(device)
                    latent = encoder(imgs)
                    preds = decoder(latent)
                    val_loss += masked_loss(preds, vfs, eye_sides).item()
            avg_val = val_loss / len(test_loader)
            print(f"Fold {fold_idx+1} Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(decoder.state_dict(), f"best_decoder_fold{fold_idx+1}.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered for fold", fold_idx+1)
                    break

        # Load best model for this fold
        decoder.load_state_dict(torch.load(f"best_decoder_fold{fold_idx+1}.pt"))

        # Evaluate and save predictions
        results = []
        maes = []
        decoder.eval()
        with torch.no_grad():
            for imgs, vfs, eye_sides, ids in tqdm(test_loader, desc="Test Eval"):
                imgs = imgs.to(device)
                vfs_np = vfs.numpy()  # shape (batch, 72)
                latent = encoder(imgs)
                preds = decoder(latent)  # shape (batch, 72)
                preds_masked = apply_mask(preds.clone(), eye_sides)
                pred_np = preds_masked.cpu().numpy()  # shape (batch, 72)
                for i in range(len(ids)):
                    actual = vfs_np[i]
                    pred = pred_np[i]
                    valid_mask = actual != 100.0
                    mae_i = np.mean(np.abs(actual[valid_mask] - pred[valid_mask])) if valid_mask.any() else np.nan
                    maes.append(mae_i)
                    results.append({
                        "id": ids[i],
                        "eye_side": eye_sides[i],
                        "actual_vf": actual.tolist(),
                        "predicted_vf": pred.tolist(),
                        "mae_mean": float(mae_i)
                    })


        fold_mae = np.nanmean(maes)
        print(f"Fold {fold_idx+1} MAE: {fold_mae:.4f} dB")
        fold_maes.append(fold_mae)

        # save per‑fold JSON
        out_path = os.path.join("predictions_fold%d.json" % (fold_idx+1))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    avg_mae_all = sum(fold_maes)/len(fold_maes)
    print(f"\n=== Cross-Validation Summary ===")
    for i,m in enumerate(fold_maes):
        print(f"  Fold {i+1}: MAE = {m:.4f} dB")
    print(f"Average MAE across all {n_splits} folds: {avg_mae_all:.4f} dB")

    return avg_mae_all

# ===========================
# Run Pipeline
# ===========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 1024

    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"

    # Stage 1: VF‑only pretraining
    uwhvf_json = os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json")
    vf_dataset = VFOnlyDataset(uwhvf_json)
    vf_loader = DataLoader(vf_dataset, batch_size=8, shuffle=True)
    decoder = VFDecoder(latent_dim=latent_dim).to(device)
    decoder = pretrain_decoder(decoder, vf_loader, device, epochs=10, lr=1e-4)

    # Stage 2: Paired fundus + VF fine‑tuning with K‑Fold
    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    paired_dataset = PairedDataset(grape_json, grape_fundus_dir, transform)

    avg_mae = train_and_evaluate_kfold(paired_dataset, encoder, latent_dim, device,
                                       n_splits=5, epochs=15, lr=1e-4, batch_size=4)

    print(f"\nFinal average MAE after cross-validation: {avg_mae:.4f} dB")
