#!/usr/bin/env python3
"""
Stage 1: VF Auto-encoder Pre-training
- Train decoder to reconstruct VF from corrupted/masked VF
- Uses 28,943 UWHVF samples (VF only, no images)
- Learns VF spatial structure and relationships
- Pre-trained decoder will be used in Stage 2
"""

import os, json, numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============== Config ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 128  # Large batch since no images
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
CORRUPTION_RATIO = 0.3  # Mask 30% of points

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "..")
UWHVF_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "uwhvf_vf_tests_standardized.json")
GRAPE_TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
GRAPE_VAL_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
PRETRAINED_DECODER_SAVE = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")

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

# ============== Dataset ==============
class VFDataset(Dataset):
    """Dataset for VF auto-encoder training."""
    def __init__(self, json_path: str, corruption_ratio: float = 0.3):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.corruption_ratio = corruption_ratio
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        hvf = np.array(item['hvf'], dtype=np.float32).flatten()
        laterality = item.get('Laterality', 'OD').strip().upper()
        
        # Get valid indices
        valid_idx = valid_indices_od if laterality.startswith('OD') else valid_indices_os
        
        # Extract only valid values (52 points)
        hvf_valid = hvf[valid_idx]
        
        # Create corrupted version by masking random points
        corrupted = hvf_valid.copy()
        n_corrupt = int(len(corrupted) * self.corruption_ratio)
        corrupt_indices = np.random.choice(len(corrupted), n_corrupt, replace=False)
        corrupted[corrupt_indices] = 0.0  # Mask with zeros
        
        return torch.tensor(corrupted), torch.tensor(hvf_valid), laterality

# ============== VF Decoder ==============
class VFAutoDecoder(nn.Module):
    """Auto-encoder decoder for VF reconstruction."""
    def __init__(self, input_dim: int = 52, hidden_dims: List[int] = [256, 512, 512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Encoder part (compress)
        for hidden_dim in hidden_dims[:len(hidden_dims)//2]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Decoder part (expand)
        for hidden_dim in hidden_dims[len(hidden_dims)//2:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============== Loss Function ==============
def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Simple MAE loss."""
    mae = torch.mean(torch.abs(pred - target))
    return mae, mae.item()

def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Pearson correlation."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    
    num = np.sum((pred_flat - pred_mean) * (target_flat - target_mean))
    denom = np.sqrt(np.sum((pred_flat - pred_mean)**2) * np.sum((target_flat - target_mean)**2))
    
    return num / (denom + 1e-8)

# ============== Training & Evaluation ==============
def evaluate(model, dataloader):
    """Evaluate model on dataset."""
    model.eval()
    total_mae = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for corrupted, target, _ in dataloader:
            corrupted = corrupted.to(DEVICE)
            target = target.to(DEVICE)
            
            pred = model(corrupted)
            
            _, mae = compute_mae(pred, target)
            total_mae += mae * len(corrupted)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    avg_mae = total_mae / len(dataloader.dataset)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    corr = pearson_correlation(all_preds, all_targets)
    
    return avg_mae, corr

def train():
    """Main training loop."""
    print("=" * 60)
    print("Stage 1: VF Auto-encoder Pre-training")
    print("=" * 60)
    
    # Load datasets
    uwhvf_dataset = VFDataset(UWHVF_JSON, corruption_ratio=CORRUPTION_RATIO)
    
    # Split UWHVF into train/val (90/10 split)
    train_size = int(0.9 * len(uwhvf_dataset))
    val_size = len(uwhvf_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        uwhvf_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"UWHVF Train samples: {len(train_dataset):,}")
    print(f"UWHVF Val samples: {len(val_dataset):,}")
    print(f"Corruption ratio: {CORRUPTION_RATIO * 100}%")
    
    # Create model
    model = VFAutoDecoder(input_dim=52, hidden_dims=[256, 512, 512, 256])
    model.to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for corrupted, target, _ in pbar:
            corrupted = corrupted.to(DEVICE)
            target = target.to(DEVICE)
            
            # Forward pass
            pred = model(corrupted)
            
            # Compute loss
            loss, mae = compute_mae(pred, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'val_corr': val_corr
            }, PRETRAINED_DECODER_SAVE)
            print(f"  ✓ New best model saved! (MAE: {val_mae:.3f} dB, Corr: {val_corr:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs")
                break
    
    print(f"\n" + "=" * 60)
    print(f"Stage 1 Complete!")
    print(f"Best Val MAE: {best_val_mae:.3f} dB")
    print(f"Pre-trained decoder saved to: {PRETRAINED_DECODER_SAVE}")
    print(f"\nNext: Run training.py to fine-tune on GRAPE")
    print("=" * 60)

if __name__ == "__main__":
    train()
