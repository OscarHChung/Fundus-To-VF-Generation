#!/usr/bin/env python3
"""
Stage 2: Fundus→VF Fine-tuning with Pre-trained Decoder
- Loads pre-trained VF decoder from Stage 1
- Fine-tunes encoder + decoder on GRAPE paired data
- Decoder already knows VF structure, learns fundus mapping
"""

import os, sys, json, numpy as np
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============== Config ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 12
EPOCHS = 80
LR_ENCODER = 5e-6  # Very small for encoder
LR_DECODER = 1e-4  # Small for pre-trained decoder
WEIGHT_DECAY = 1e-4
PATIENCE = 20
NUM_ENCODER_BLOCKS = 4

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
PRETRAINED_DECODER = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")
BASE_DIR = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
BEST_SAVE = os.path.join(CURRENT_DIR, "best_final_model.pth")

# ============== Mask Definition ==============
mask_OD = np.array([
    [False, False, False, True,  True,  True,  True,  False, False],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [True,  True,  True,  True,  True,  True,  True,  False, True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True ],
    [False, False, True,  True,  True,  True,  True,  True,  False],
    [False, False, False, True,  True,  True,  True,  False, False]
], dtype=bool)

valid_indices_od: List[int] = [i for i, v in enumerate(mask_OD.flatten()) if v]
valid_indices_os: List[int] = list(reversed(valid_indices_od))

# ============== Load RETFound ==============
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)
print(f"✓ Loaded RETFound base model")

# Load pre-trained decoder
if not os.path.exists(PRETRAINED_DECODER):
    raise FileNotFoundError(f"Pre-trained decoder not found at {PRETRAINED_DECODER}. Run pretraining.py first!")

pretrained_checkpoint = torch.load(PRETRAINED_DECODER, map_location='cpu', weights_only=False)
print(f"✓ Loaded pre-trained VF decoder (Val MAE: {pretrained_checkpoint['val_mae']:.3f} dB)")

# ============== Dataset ==============
retfound_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class GRAPEDataset(Dataset):
    def __init__(self, json_path: str, fundus_dir: str, transform):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.fundus_dir = fundus_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.fundus_dir, item['FundusImage'])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        hvf = np.array(item['hvf'], dtype=np.float32).flatten()
        laterality = item.get('Laterality', 'OD').strip().upper()
        return img_tensor, torch.tensor(hvf), laterality

# ============== Unified Model with Pre-trained Decoder ==============
class UnifiedModelWithPretraining(nn.Module):
    def __init__(self, encoder_model, pretrained_decoder_state, num_blocks_to_finetune=4):
        super().__init__()
        self.encoder = encoder_model
        
        # Freeze encoder initially
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        if hasattr(self.encoder, 'blocks'):
            total_blocks = len(self.encoder.blocks)
            for i in range(total_blocks - num_blocks_to_finetune, total_blocks):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True
            print(f"✓ Unfrozen last {num_blocks_to_finetune} encoder blocks")
        
        # Projection from encoder (1024) to VF decoder input (52)
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 52)
        )
        
        # Load pre-trained VF decoder
        from pretraining import VFAutoDecoder
        self.vf_decoder = VFAutoDecoder(input_dim=52, hidden_dims=[256, 512, 512, 256])
        self.vf_decoder.load_state_dict(pretrained_decoder_state)
        print(f"✓ Loaded pre-trained VF decoder weights")
    
    def forward(self, x):
        # Get encoder features
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]  # CLS token
        
        # Project to VF space
        vf_features = self.projection(latent)
        
        # Decode with pre-trained decoder
        pred = self.vf_decoder(vf_features)
        
        return pred

# ============== Loss Function ==============
def compute_masked_mae(pred: torch.Tensor, target: torch.Tensor, 
                       laterality: List[str]) -> Tuple[torch.Tensor, float]:
    device = pred.device
    target = target.to(device)
    total_loss = 0.0
    total_points = 0
    
    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        valid_idx_t = torch.tensor(valid_idx, dtype=torch.long, device=device)
        
        pred_valid = pred[i]
        target_valid = target[i, valid_idx_t]
        
        abs_error = torch.abs(pred_valid - target_valid)
        total_loss += abs_error.sum()
        total_points += len(valid_idx)
    
    mae = total_loss / total_points
    return mae, mae.item()

def pearson_correlation(pred: np.ndarray, target: np.ndarray, 
                       laterality: List[str]) -> float:
    all_pred = []
    all_target = []
    
    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        all_pred.extend(pred[i].tolist())
        all_target.extend(target[i, valid_idx].tolist())
    
    all_pred = np.array(all_pred)
    all_target = np.array(all_target)
    
    pred_mean = all_pred.mean()
    target_mean = all_target.mean()
    
    num = np.sum((all_pred - pred_mean) * (all_target - target_mean))
    denom = np.sqrt(np.sum((all_pred - pred_mean)**2) * np.sum((all_target - target_mean)**2))
    
    return num / (denom + 1e-8)

# ============== Training & Evaluation ==============
def evaluate(model, dataloader):
    model.eval()
    total_mae = 0.0
    all_preds = []
    all_targets = []
    all_lats = []
    
    with torch.no_grad():
        for imgs, hvf, laterality in dataloader:
            imgs = imgs.to(DEVICE)
            hvf = hvf.to(DEVICE)
            
            pred = model(imgs)
            
            _, mae = compute_masked_mae(pred, hvf, laterality)
            total_mae += mae * len(imgs)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(hvf.cpu().numpy())
            all_lats.extend(laterality)
    
    avg_mae = total_mae / len(dataloader.dataset)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    corr = pearson_correlation(all_preds, all_targets, all_lats)
    
    return avg_mae, corr

def train():
    print("=" * 60)
    print("Stage 2: Fundus→VF Fine-tuning")
    print("=" * 60)
    
    # Create datasets
    train_dataset = GRAPEDataset(TRAIN_JSON, FUNDUS_DIR, retfound_transform)
    val_dataset = GRAPEDataset(VAL_JSON, FUNDUS_DIR, retfound_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"GRAPE Train samples: {len(train_dataset)}")
    print(f"GRAPE Val samples: {len(val_dataset)}")
    
    # Create model with pre-trained decoder
    model = UnifiedModelWithPretraining(
        base_model,
        pretrained_checkpoint['model_state_dict'],
        num_blocks_to_finetune=NUM_ENCODER_BLOCKS
    )
    model.to(DEVICE)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    projection_params = sum(p.numel() for p in model.projection.parameters())
    decoder_params = sum(p.numel() for p in model.vf_decoder.parameters())
    print(f"Trainable encoder params: {encoder_params:,}")
    print(f"Projection params: {projection_params:,}")
    print(f"VF decoder params: {decoder_params:,}")
    print(f"Total trainable: {encoder_params + projection_params + decoder_params:,}")
    
    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': LR_ENCODER},
        {'params': model.projection.parameters(), 'lr': LR_DECODER},
        {'params': model.vf_decoder.parameters(), 'lr': LR_DECODER}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, hvf, laterality in pbar:
            imgs = imgs.to(DEVICE)
            hvf = hvf.to(DEVICE)
            
            pred = model(imgs)
            loss, mae = compute_masked_mae(pred, hvf, laterality)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        val_mae, val_corr = evaluate(model, val_loader)
        train_mae, train_corr = evaluate(model, train_loader)
        
        print(f"\n[Epoch {epoch}]")
        print(f"  Train MAE: {train_mae:.3f} dB | Corr: {train_corr:.3f}")
        print(f"  Val MAE:   {val_mae:.3f} dB | Corr: {val_corr:.3f}")
        
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'val_corr': val_corr
            }, BEST_SAVE)
            print(f"  ✓ New best model saved! (MAE: {val_mae:.3f} dB, Corr: {val_corr:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs")
                break
    
    print(f"\n" + "=" * 60)
    print(f"Training Complete!")
    print(f"Best Val MAE: {best_val_mae:.3f} dB")
    print(f"Model saved to: {BEST_SAVE}")
    print("=" * 60)

if __name__ == "__main__":
    train()
