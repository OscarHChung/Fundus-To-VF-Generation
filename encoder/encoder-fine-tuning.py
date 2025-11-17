import os
import sys
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import argparse

# safe loading helper (RETFound checkpoint contains argparse.Namespace)
torch.serialization.add_safe_globals([argparse.Namespace])

# ---------- Configuration ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "../data")
FUNDUS_DIR = os.path.join(BASE_DIR, "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "vf_tests", "grape_test.json")

RETFOUND_DIR = os.path.join(CURRENT_DIR, "RETFound_MAE")
RETFOUND_WEIGHTS = os.path.join(CURRENT_DIR, "RETFound_cfp_weights.pth")
OUT_NBEST = os.path.join(CURRENT_DIR, "nbest_encoder.pth")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 8
EPOCHS = 20
UNFREEZE_LAST_N = 1   # Only unfreeze 1 block for slower improvement
LR_ENCODER = 1e-6
LR_HEAD = 1e-4

PEARSON_MIN = 0.40
PEARSON_MAX = 0.60  # Early stopping window to preserve latent richness

# ---------- Dataset ----------
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            self.items = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.img_dir, item['FundusImage'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        hvf = np.array(item['hvf'], dtype=np.float32).reshape(-1)
        hvf = torch.tensor(hvf, dtype=torch.float32)
        laterality = item.get('Laterality', item.get('laterality', 'OD'))
        return img, hvf, laterality

# ---------- Load RETFound ----------
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

print("Loading RETFound base weights...")
with torch.serialization.safe_globals([argparse.Namespace]):
    base_ckpt = torch.load(RETFOUND_WEIGHTS, map_location="cpu")

base_model = mae_vit_large_patch16_dec512d8b()
if isinstance(base_ckpt, dict) and 'model' in base_ckpt:
    base_model.load_state_dict(base_ckpt['model'], strict=False)
else:
    base_model.load_state_dict(base_ckpt, strict=False)
base_model.to(DEVICE)
base_model.train()

# ---------- Encoder Wrapper ----------
class RetFoundEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        latent = self.model.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        return latent

# ---------- Regression Head ----------
class RegressionHead(nn.Module):
    def __init__(self, latent_dim=1024, hidden=512, out=72):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.mlp(x)

# ---------- Utilities ----------
def flatten_preds_targets(preds_list, targs_list):
    preds = np.concatenate(preds_list, axis=0)
    targs = np.concatenate(targs_list, axis=0)
    return preds.flatten(), targs.flatten()

def safe_pearson(x, y):
    if x.std() == 0 or y.std() == 0:
        return 0.0
    r = np.corrcoef(x, y)[0,1]
    return float(r) if not np.isnan(r) else 0.0

# ---------- Training ----------
def train_encoder():
    # Strong augmentations to prevent memorization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = PairedDataset(TRAIN_JSON, FUNDUS_DIR, transform)
    val_ds = PairedDataset(VAL_JSON, FUNDUS_DIR, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    encoder = RetFoundEncoderWrapper(base_model)

    # Freeze all, then unfreeze only last block(s)
    for p in base_model.parameters():
        p.requires_grad = False

    if hasattr(base_model, "blocks") and UNFREEZE_LAST_N > 0:
        for blk in base_model.blocks[-UNFREEZE_LAST_N:]:
            for p in blk.parameters():
                p.requires_grad = True
        print(f"Unfrozen last {UNFREEZE_LAST_N} transformer block(s).")
    else:
        print("Warning: base_model has no attribute 'blocks'; no encoder layers unfrozen.")

    head = RegressionHead(latent_dim=1024, hidden=512, out=72).to(DEVICE)

    enc_params = [p for p in base_model.parameters() if p.requires_grad]
    optim_groups = []
    if len(enc_params) > 0:
        optim_groups.append({"params": enc_params, "lr": LR_ENCODER})
    optim_groups.append({"params": head.parameters(), "lr": LR_HEAD})
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=1e-5)
    criterion = nn.L1Loss(reduction='mean')

    best_saved = None

    for epoch in range(1, EPOCHS+1):
        base_model.train()
        head.train()
        train_loss = 0.0

        for imgs, hvf72, _ in tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=False):
            imgs = imgs.to(DEVICE)
            hvf72 = hvf72.to(DEVICE)
            optimizer.zero_grad()
            lat = encoder(imgs)
            preds72 = head(lat)
            loss = criterion(preds72, hvf72)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        base_model.eval()
        head.eval()
        val_loss = 0.0
        preds_list, targs_list = [], []

        with torch.no_grad():
            for imgs, hvf72, _ in val_loader:
                imgs = imgs.to(DEVICE)
                hvf72 = hvf72.to(DEVICE)
                lat = encoder(imgs)
                preds72 = head(lat)
                val_loss += criterion(preds72, hvf72).item() * imgs.size(0)
                preds_list.append(preds72.cpu().numpy())
                targs_list.append(hvf72.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        flat_p, flat_t = flatten_preds_targets(preds_list, targs_list)
        pearson = safe_pearson(flat_p, flat_t)

        print(f"[Epoch {epoch}] Train MAE={train_loss:.4f}, Val MAE={val_loss:.4f}, Pearson={pearson:.4f}")

        # Save checkpoint only if Pearson in target window
        if PEARSON_MIN <= pearson <= PEARSON_MAX:
            save_dict = {
                "encoder_state": base_model.state_dict(),
                "head_state": head.state_dict(),
                "pearson": pearson,
                "epoch": epoch
            }
            torch.save(save_dict, OUT_NBEST)
            torch.save(save_dict, os.path.join(CURRENT_DIR, f"moderate_encoder_epoch{epoch}.pth"))
            best_saved = OUT_NBEST
            print(f"  Saved moderate encoder to {OUT_NBEST} (Pearson={pearson:.4f})")

        # Early stopping: stop if Pearson exceeds upper bound
        if pearson > PEARSON_MAX:
            print(f"Pearson {pearson:.4f} > {PEARSON_MAX}, stopping fine-tuning to preserve latent richness.")
            break

    print("Done. nbest saved at:", best_saved)

if __name__ == "__main__":
    print("Device:", DEVICE)
    train_encoder()
