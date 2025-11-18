#!/usr/bin/env python3
"""
Simple MLP Decoder for Fundus->VF
- Direct mapping from encoder features to 52 VF points
- Residual connections for better gradients
- Layer normalization for stability
- Proper masked loss computation
"""

import os, sys, json, numpy as np
from typing import List, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============== Config ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
ENCODER_DIR = os.path.join(BASE_DIR, "encoder")
BEST_SAVE = os.path.join(CURRENT_DIR, "best_decoder.pth")

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

# ============== Load Encoder ==============
sys.path.insert(0, ENCODER_DIR)
from retfound_encoder import encoder, retfound_transform

encoder.to(DEVICE)
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

print(f"✓ Loaded frozen encoder on {DEVICE}")

# ============== Dataset ==============
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

# ============== Decoder Model ==============
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.net(x))

class VFDecoder(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, 
                 output_dim: int = 52, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)

# ============== Loss Functions ==============
def compute_masked_mae(pred: torch.Tensor, target: torch.Tensor, 
                       laterality: List[str]) -> Tuple[torch.Tensor, float]:
    """
    Compute MAE only on valid (non-masked) points.
    Returns: (loss_tensor, mae_in_dB)
    """
    device = pred.device
    target = target.to(device)  # Move target to same device as pred
    total_loss = 0.0
    total_points = 0
    
    for i, lat in enumerate(laterality):
        # Get valid indices based on laterality
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        valid_idx_t = torch.tensor(valid_idx, dtype=torch.long, device=device)
        
        # Extract valid predictions and targets
        pred_valid = pred[i, :52]  # Only first 52 outputs
        target_valid = target[i, valid_idx_t]
        
        # Compute absolute error
        abs_error = torch.abs(pred_valid - target_valid)
        total_loss += abs_error.sum()
        total_points += len(valid_idx)
    
    mae_normalized = total_loss / total_points
    mae_db = mae_normalized.item()  # Already in dB scale
    
    return mae_normalized, mae_db

def pearson_correlation(pred: np.ndarray, target: np.ndarray, 
                       laterality: List[str]) -> float:
    """Compute Pearson correlation across all valid points."""
    all_pred = []
    all_target = []
    
    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        all_pred.extend(pred[i, :52].tolist())
        all_target.extend(target[i, valid_idx].tolist())
    
    all_pred = np.array(all_pred)
    all_target = np.array(all_target)
    
    # Pearson correlation
    pred_mean = all_pred.mean()
    target_mean = all_target.mean()
    
    num = np.sum((all_pred - pred_mean) * (all_target - target_mean))
    denom = np.sqrt(np.sum((all_pred - pred_mean)**2) * np.sum((all_target - target_mean)**2))
    
    return num / (denom + 1e-8)

# ============== Training & Evaluation ==============
def evaluate(decoder, dataloader):
    """Evaluate decoder on a dataset."""
    decoder.eval()
    total_mae = 0.0
    all_preds = []
    all_targets = []
    all_lats = []
    
    with torch.no_grad():
        for imgs, hvf, laterality in dataloader:
            imgs = imgs.to(DEVICE)
            hvf = hvf.to(DEVICE)  # Move targets to device
            
            # Get encoder features
            with torch.no_grad():
                features = encoder(imgs)  # [B, 1024]
            
            # Decoder prediction
            pred = decoder(features)  # [B, 52]
            
            # Compute MAE
            _, mae_db = compute_masked_mae(pred, hvf, laterality)
            total_mae += mae_db * len(imgs)
            
            # Store for correlation
            all_preds.append(pred.cpu().numpy())
            all_targets.append(hvf.cpu().numpy())
            all_lats.extend(laterality)
    
    avg_mae = total_mae / len(dataloader.dataset)
    
    # Compute correlation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    corr = pearson_correlation(all_preds, all_targets, all_lats)
    
    return avg_mae, corr

def train():
    """Main training loop."""
    # Create datasets
    train_dataset = GRAPEDataset(TRAIN_JSON, FUNDUS_DIR, retfound_transform)
    val_dataset = GRAPEDataset(VAL_JSON, FUNDUS_DIR, retfound_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create decoder
    decoder = VFDecoder(
        input_dim=1024,
        hidden_dim=512,
        output_dim=52,
        num_blocks=3,
        dropout=0.2
    ).to(DEVICE)
    
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, hvf, laterality in pbar:
            imgs = imgs.to(DEVICE)
            hvf = hvf.to(DEVICE)  # Move targets to device
            
            # Get encoder features (frozen)
            with torch.no_grad():
                features = encoder(imgs)
            
            # Forward pass
            pred = decoder(features)
            
            # Compute loss
            loss, mae_db = compute_masked_mae(pred, hvf, laterality)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        val_mae, val_corr = evaluate(decoder, val_loader)
        train_mae, train_corr = evaluate(decoder, train_loader)
        
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
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_corr': val_corr
            }, BEST_SAVE)
            print(f"  ✓ New best model saved! (MAE: {val_mae:.3f} dB)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs")
                break
    
    print(f"\nTraining complete! Best Val MAE: {best_val_mae:.3f} dB")
    print(f"Model saved to: {BEST_SAVE}")

if __name__ == "__main__":
    train()
