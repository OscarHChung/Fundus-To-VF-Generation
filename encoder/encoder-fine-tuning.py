import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

# =====================================================
# Dataset class
# =====================================================
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['FundusImage'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        vf = torch.tensor(item['hvf'], dtype=torch.float32).view(-1)
        laterality = item.get('laterality', 'OD')
        return img, vf, laterality

# =====================================================
# Load RETFound encoder
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
retfound_dir = os.path.join(current_dir, '../encoder/RETFound_MAE')
sys.path.insert(0, retfound_dir)
from models_mae import mae_vit_large_patch16_dec512d8b

checkpoint_path = os.path.join(current_dir, "../encoder/RETFound_cfp_weights.pth")
with torch.serialization.safe_globals([__import__('argparse').Namespace]):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)

# =====================================================
# Encoder wrapper
# =====================================================
class RetFoundEncoderWrapper(nn.Module):
    def __init__(self, model, latent_dim=1024, unfreeze_last_n=4):
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last n transformer blocks
        for blk in self.model.blocks[-unfreeze_last_n:]:
            for param in blk.parameters():
                param.requires_grad = True

    def forward(self, x):
        latent = self.model.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        return latent

# =====================================================
# Regression head (MLP)
# =====================================================
class RegressionHead(nn.Module):
    def __init__(self, latent_dim=1024, output_dim=52):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# =====================================================
# Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================================================
# Training function
# =====================================================
def train_encoder_only(encoder, head, train_loader, val_loader, device,
                       epochs=20, lr_encoder=1e-5, lr_head=1e-4):
    encoder.to(device)
    head.to(device)
    
    # Separate optimizers for encoder and head
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': lr_encoder},
        {'params': head.parameters(), 'lr': lr_head}
    ])
    criterion = nn.L1Loss()
    best_pearson = -1.0

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        train_loss = 0

        for imgs, vfs, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            imgs, vfs = imgs.to(device), vfs.to(device)
            optimizer.zero_grad()
            latents = encoder(imgs)
            preds = head(latents)
            loss = criterion(preds, vfs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        encoder.eval()
        head.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, vfs, _ in val_loader:
                imgs, vfs = imgs.to(device), vfs.to(device)
                latents = encoder(imgs)
                preds = head(latents)
                val_loss += criterion(preds, vfs).item() * imgs.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(vfs.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        # Compute Pearson correlation across flattened arrays
        pearson = np.corrcoef(all_preds.flatten(), all_targets.flatten())[0,1]

        print(f"[Epoch {epoch}] Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Pearson={pearson:.4f}")

        if pearson > best_pearson:
            best_pearson = pearson
            torch.save({
                'encoder_state': encoder.state_dict(),
                'head_state': head.state_dict()
            }, os.path.join(current_dir, "best_encoder_only.pth"))
            print(f"  New best encoder saved with Pearson={best_pearson:.4f}")

    print("Fine-tuning complete. Best Pearson:", best_pearson)
    return encoder, head

# =====================================================
# Main
# =====================================================
if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Using device:", device)

    base_dir = os.path.join(current_dir, "../data")
    fundus_dir = os.path.join(base_dir, 'fundus', 'grape_fundus_images')

    train_dataset = PairedDataset(os.path.join(base_dir, 'vf_tests', 'grape_train.json'), fundus_dir, transform)
    val_dataset = PairedDataset(os.path.join(base_dir, 'vf_tests', 'grape_test.json'), fundus_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    encoder = RetFoundEncoderWrapper(base_model, latent_dim=1024, unfreeze_last_n=4)
    output_dim = train_dataset[0][1].shape[0]
    head = RegressionHead(latent_dim=1024, output_dim=output_dim)

    encoder, head = train_encoder_only(encoder, head, train_loader, val_loader, device,
                                       epochs=20, lr_encoder=1e-5, lr_head=1e-4)
