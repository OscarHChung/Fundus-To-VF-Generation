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

# =====================================================
# Paths
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# Dataset
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

        # Use raw HVF values for MAE
        vf = torch.tensor(np.array(item['hvf'], dtype=np.float32), dtype=torch.float32).view(-1)
        laterality = item.get('laterality', 'OD')
        return img, vf, laterality

# =====================================================
# Encoder wrapper
# =====================================================
class RetFoundEncoderWrapper(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model

    def forward(self, x):
        # Forward full image (mask_ratio=0.0)
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]  # CLS token
        return latent

# =====================================================
# Improved Decoder
# =====================================================
class ImprovedDecoder(nn.Module):
    def __init__(self, latent_dim=1024, output_dim=52):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# =====================================================
# Training function
# =====================================================
def train_decoder(encoder, decoder, train_loader, val_loader, device, epochs=30, lr=1e-4):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()  # Freeze encoder

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE directly on real values
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_mae = float('inf')

    for epoch in range(1, epochs + 1):
        decoder.train()
        train_loss, train_mae = 0, 0

        for imgs, vfs, _ in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            imgs, vfs = imgs.to(device), vfs.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                latents = encoder(imgs)

            preds = decoder(latents)
            loss = criterion(preds, vfs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_mae += torch.mean(torch.abs(preds - vfs)).item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # Validation
        decoder.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for imgs, vfs, _ in val_loader:
                imgs, vfs = imgs.to(device), vfs.to(device)
                latents = encoder(imgs)
                preds = decoder(latents)

                loss = criterion(preds, vfs)
                val_loss += loss.item() * imgs.size(0)
                val_mae += torch.mean(torch.abs(preds - vfs)).item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"[Epoch {epoch}] Train Loss={train_loss:.4f}, Train MAE={train_mae:.4f}, "
              f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'decoder_state': decoder.state_dict(),
                'val_mae': best_val_mae
            }, os.path.join(current_dir, "best_decoder.pth"))
            print(f"  New best decoder saved with Val MAE {best_val_mae:.4f}")

    print("Training complete. Best Val MAE:", best_val_mae)

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    base_dir = os.path.join(current_dir, "../data")
    fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_train.json"), fundus_dir, transform)
    val_dataset = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_test.json"), fundus_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Load best encoder
    retfound_dir = os.path.join(current_dir, '../encoder/RETFound_MAE')
    sys.path.insert(0, retfound_dir)
    from models_mae import mae_vit_large_patch16_dec512d8b

    encoder_model = mae_vit_large_patch16_dec512d8b()
    encoder_ckpt = torch.load(os.path.join(current_dir, "../encoder/best_encoder_only.pth"), map_location="cpu")
    encoder_model.load_state_dict(encoder_ckpt['encoder_state'], strict=False)
    encoder = RetFoundEncoderWrapper(encoder_model)

    # Initialize improved decoder
    output_dim = train_dataset[0][1].shape[0]
    decoder = ImprovedDecoder(latent_dim=1024, output_dim=output_dim)

    # Train decoder
    train_decoder(encoder, decoder, train_loader, val_loader, device, epochs=30, lr=1e-4)
