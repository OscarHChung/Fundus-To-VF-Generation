import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm

# =====================================================
# Dataset class with augmentation
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
# Load RETFound
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

# Partial fine-tuning: last 4 blocks + norm layers
for name, param in base_model.named_parameters():
    if any(k in name for k in ['blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm']):
        param.requires_grad = True
    else:
        param.requires_grad = False
base_model.train()

# =====================================================
# Encoder wrapper
# =====================================================
class RetFoundEncoderWrapper(nn.Module):
    def __init__(self, model, latent_dim=1024):
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim

    def forward(self, x):
        latent = self.model.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        return latent

# =====================================================
# Upgraded Decoder
# =====================================================
class UpgradedDecoder(nn.Module):
    def __init__(self, latent_dim=1024, output_dim=52, hidden_dims=[1024, 512, 256], dropout=0.2):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hdim))
            input_dim = hdim
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =====================================================
# Transform + augmentation
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================================================
# Training function with separate LR & gradient clipping
# =====================================================
def train_model(encoder, decoder, train_loader, val_loader, device, epochs=30, lr_encoder=1e-5, lr_decoder=1e-4):
    encoder.to(device)
    decoder.to(device)

    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': lr_encoder},
        {'params': decoder.parameters(), 'lr': lr_decoder}
    ])
    criterion = nn.L1Loss()

    best_val_mae = float('inf')

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        train_loss = 0
        for imgs, vfs, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, vfs = imgs.to(device), vfs.to(device)
            optimizer.zero_grad()
            latents = encoder(imgs)
            preds = decoder(latents)
            loss = criterion(preds, vfs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, vfs, _ in val_loader:
                imgs, vfs = imgs.to(device), vfs.to(device)
                latents = encoder(imgs)
                preds = decoder(latents)
                loss = criterion(preds, vfs)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Epoch {epoch}] Train Loss={train_loss:.4f}, Val MAE={val_loss:.4f}")
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save({
                'encoder_state': encoder.state_dict(),
                'decoder_state': decoder.state_dict()
            }, os.path.join(current_dir, "best_model_finetuned.pth"))
            print(f"  New best model saved with Val MAE {best_val_mae:.4f}")

    print("Training complete. Best Val MAE:", best_val_mae)
    return encoder, decoder

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

    encoder = RetFoundEncoderWrapper(base_model)
    output_dim = train_dataset[0][1].shape[0]
    decoder = UpgradedDecoder(latent_dim=1024, output_dim=output_dim)

    encoder, decoder = train_model(encoder, decoder, train_loader, val_loader, device, epochs=30)

