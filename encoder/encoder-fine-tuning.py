#!/usr/bin/env python3
"""
Simple Encoder Fine-tuning for VF Prediction
- Fine-tunes last N blocks of RETFound encoder
- Adds lightweight prediction head for 52 VF values
- Careful not to overfit on limited GRAPE data
- Target: 0.7-0.8 correlation, leaving room for decoder
"""

import os
import sys
import json
import numpy as np
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
BATCH_SIZE = 8
EPOCHS = 50
LR_ENCODER = 2e-5  # Increased from 5e-6
LR_HEAD = 1e-3     # Much higher for prediction head
WEIGHT_DECAY = 1e-4
PATIENCE = 12
NUM_BLOCKS_TO_FINETUNE = 4  # Increased from 2 to 4

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "RETFound_cfp_weights.pth")
BASE_DIR = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
BEST_SAVE = os.path.join(CURRENT_DIR, "best_encoder_finetuned.pth")

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

# Safe loading
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)

print(f"✓ Loaded RETFound base model")

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
        
        # Load image
        img_path = os.path.join(self.fundus_dir, item['FundusImage'])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        # Get VF values (flattened)
        hvf = np.array(item['hvf'], dtype=np.float32).flatten()
        
        # Get laterality
        laterality = item.get('Laterality', 'OD').strip().upper()
        
        return img_tensor, torch.tensor(hvf), laterality

# ============== Encoder + Prediction Head ==============
class EncoderWithPredictionHead(nn.Module):
    def __init__(self, encoder_model, num_blocks_to_finetune=2):
        super().__init__()
        self.encoder = encoder_model
        self.num_blocks_to_finetune = num_blocks_to_finetune
        
        # Freeze all parameters first
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        if hasattr(self.encoder, 'blocks'):
            total_blocks = len(self.encoder.blocks)
            for i in range(total_blocks - num_blocks_to_finetune, total_blocks):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True
            print(f"✓ Unfrozen last {num_blocks_to_finetune} encoder blocks (blocks {total_blocks - num_blocks_to_finetune} to {total_blocks - 1})")
        
        # Simple prediction head (intentionally simple to avoid overfitting)
        # RETFound ViT-Large has 1024 dim embeddings
        embed_dim = 1024
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 52)
        )
    
    def forward(self, x, mask_ratio=0.0):
        # Get encoder output
        latent = self.encoder.forward_encoder(x, mask_ratio=mask_ratio)[0]
        
        # Use CLS token
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        
        # Predict VF values
        pred = self.prediction_head(latent)
        return pred

# ============== Loss Function ==============
def compute_masked_mae(pred: torch.Tensor, target: torch.Tensor, 
                       laterality: List[str]) -> Tuple[torch.Tensor, float]:
    """Compute MAE only on valid points."""
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
    """Compute Pearson correlation across all valid points."""
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
    """Evaluate model on dataset."""
    model.eval()
    total_mae = 0.0
    all_preds = []
    all_targets = []
    all_lats = []
    
    with torch.no_grad():
        for imgs, hvf, laterality in dataloader:
            imgs = imgs.to(DEVICE)
            hvf = hvf.to(DEVICE)
            
            pred = model(imgs, mask_ratio=0.0)
            
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
    """Main training loop."""
    # Create datasets
    train_dataset = GRAPEDataset(TRAIN_JSON, FUNDUS_DIR, retfound_transform)
    val_dataset = GRAPEDataset(VAL_JSON, FUNDUS_DIR, retfound_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = EncoderWithPredictionHead(base_model, num_blocks_to_finetune=NUM_BLOCKS_TO_FINETUNE)
    model.to(DEVICE)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Optimizer with different learning rates for encoder and head
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.prediction_head.parameters())
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': head_params, 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, hvf, laterality in pbar:
            imgs = imgs.to(DEVICE)
            hvf = hvf.to(DEVICE)
            
            # Forward pass
            pred = model(imgs, mask_ratio=0.0)
            
            # Compute loss
            loss, mae = compute_masked_mae(pred, hvf, laterality)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        val_mae, val_corr = evaluate(model, val_loader)
        train_mae, train_corr = evaluate(model, train_loader)
        
        print(f"\n[Epoch {epoch}]")
        print(f"  Train MAE: {train_mae:.3f} dB | Corr: {train_corr:.3f}")
        print(f"  Val MAE:   {val_mae:.3f} dB | Corr: {val_corr:.3f}")
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'encoder_state': model.encoder.state_dict(),
                'prediction_head_state': model.prediction_head.state_dict(),
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
    
    print(f"\nFine-tuning complete!")
    print(f"Best Val MAE: {best_val_mae:.3f} dB")
    print(f"Model saved to: {BEST_SAVE}")
    print(f"\nNext steps:")
    print(f"1. Update encoder/retfound_encoder.py to load this checkpoint")
    print(f"2. Re-train the decoder with the fine-tuned encoder")

if __name__ == "__main__":
    train()