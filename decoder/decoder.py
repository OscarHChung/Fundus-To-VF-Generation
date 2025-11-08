# ==============================================
# Trains the decoder with UWHVF VF tests first, then fine-tunes using GRAPE paired data (fundus + VF)
# Saves actual vs prediction results into JSON with MAE computed only on valid points.
# ==============================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoder.retfound_encoder import encoder

# ===========================
# GRAPE Dataset
# ===========================
class PairedDataset(Dataset):
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
# Decoder
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


# ===========================
# Loss Function
# ===========================
def masked_loss(preds, targets, eye_sides, mask_value=100.0):
    preds_masked = apply_mask(preds, eye_sides, mask_value)
    valid = (targets != mask_value)
    valid &= (preds_masked != mask_value)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    mse = ((preds_masked[valid] - targets[valid]) ** 2).mean()
    mae = torch.abs(preds_masked[valid] - targets[valid]).mean()
    return 0.7 * mse + 0.3 * mae


# ===========================
# Training & Evaluation
# ===========================
def train_model(encoder, decoder, train_loader, val_loader, device, epochs=15, lr=1e-4):
    optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss = float('inf')
    patience, patience_counter = 3, 0

    for epoch in range(epochs):
        decoder.train()
        total_loss = 0.0
        for imgs, vfs, eye_sides, _ in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
            imgs, vfs = imgs.to(device), vfs.to(device)
            latent = encoder(imgs)
            preds = decoder(latent)
            loss = masked_loss(preds, vfs, eye_sides)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # Validation
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, vfs, eye_sides, _ in val_loader:
                imgs, vfs = imgs.to(device), vfs.to(device)
                latent = encoder(imgs)
                preds = decoder(latent)
                val_loss += masked_loss(preds, vfs, eye_sides).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(decoder.state_dict(), "best_decoder.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    decoder.load_state_dict(torch.load("best_decoder.pt"))
    return decoder


# =================================================
# Evaluation of MAE per image / Saving data to JSON
# =================================================
def evaluate_model(encoder, decoder, dataset, device, save_path):
    results, mae_all = [], []
    decoder.eval()

    with torch.no_grad():
        for img, vf_true, eye_side, img_id in tqdm(dataset, desc="Testing"):
            img_tensor = img.unsqueeze(0).to(device)
            vf_true_np = vf_true.numpy()
            latent = encoder(img_tensor)
            vf_pred = decoder(latent)
            vf_pred_masked = apply_mask(vf_pred.clone(), [eye_side])
            vf_pred_np = vf_pred_masked.cpu().numpy().flatten()

            valid_mask = vf_true_np != 100.0
            mae = np.mean(np.abs(vf_true_np[valid_mask] - vf_pred_np[valid_mask])) if valid_mask.any() else np.nan
            mae_all.append(mae)
            results.append({
                "id": img_id,
                "eye_side": eye_side,
                "actual_vf": vf_true_np.tolist(),
                "predicted_vf": vf_pred_np.tolist(),
                "mae_mean": float(mae)
            })

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    valid_mae = [m for m in mae_all if not np.isnan(m)]
    print(f"\nAverage MAE across test set: {np.mean(valid_mae):.3f} dB")
    return np.mean(valid_mae)


# ===========================
# Run Pipeline
# ===========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = PairedDataset(grape_json, grape_fundus_dir, transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    decoder = VFDecoder().to(device)

    print("\n=== Training ===")
    decoder = train_model(encoder, decoder, train_loader, val_loader, device)

    print("\n=== Evaluating ===")
    evaluate_model(encoder, decoder, test_set, device, save_path=os.path.join(base_dir, "predictions_vs_actuals_improved.json"))
