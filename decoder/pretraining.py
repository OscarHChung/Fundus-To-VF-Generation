#!/usr/bin/env python3
"""
Stage 1: VF Auto-encoder Pre-training - MPS OPTIMIZED
- Train decoder to reconstruct VF from corrupted/masked VF
- Uses 28,943 UWHVF samples (VF only, no images)
- MPS-friendly: no hangs, explicit operations
"""

import os, json, numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============== Config ==============
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# ============== Config ==============
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✓ Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print(f"⚠️  MPS not available, using CPU")

BATCH_SIZE = 256  # Larger batch on CPU
EPOCHS = 80  # Fewer epochs since simpler architecture
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
CORRUPTION_RATIO = 0.35
MASKED_VALUE_THRESHOLD = 99.0

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
    """Enhanced dataset for VF auto-encoder training with multiple corruption strategies."""
    def __init__(self, json_path: str, corruption_ratio: float = 0.35):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.corruption_ratio = corruption_ratio
    
    def __len__(self):
        return len(self.data)
    
    def corrupt_vf(self, hvf_valid):
        """Apply multiple corruption strategies"""
        corrupted = hvf_valid.copy()
        n_corrupt = int(len(corrupted) * self.corruption_ratio)
        
        # Strategy 1: Random masking (50% of corruptions)
        n_random = n_corrupt // 2
        random_indices = np.random.choice(len(corrupted), n_random, replace=False)
        corrupted[random_indices] = 0.0
        
        # Strategy 2: Add noise (30% of corruptions)
        n_noise = int(n_corrupt * 0.3)
        noise_indices = np.random.choice(len(corrupted), n_noise, replace=False)
        noise = np.random.randn(n_noise) * 3.0  # Gaussian noise
        corrupted[noise_indices] = np.clip(corrupted[noise_indices] + noise, 0, 40)
        
        # Strategy 3: Regional dropout (20% of corruptions)
        # Simulate loss of quadrants
        if np.random.rand() < 0.2:
            quadrant_size = len(corrupted) // 4
            quadrant_start = np.random.randint(0, len(corrupted) - quadrant_size)
            corrupted[quadrant_start:quadrant_start + quadrant_size] = 0.0
        
        return corrupted
    
    def __getitem__(self, idx):
        item = self.data[idx]
        hvf = np.array(item['hvf'], dtype=np.float32).flatten()
        laterality = item.get('Laterality', 'OD').strip().upper()
        
        # Get valid indices
        valid_idx = valid_indices_od if laterality.startswith('OD') else valid_indices_os
        
        # Extract only valid values (52 points)
        hvf_valid = hvf[valid_idx]
        
        # Filter out already masked values
        mask = hvf_valid < MASKED_VALUE_THRESHOLD
        hvf_clean = hvf_valid.copy()
        hvf_clean[~mask] = 0.0  # Set invalid values to 0
        
        # Create corrupted version
        corrupted = self.corrupt_vf(hvf_clean)
        
        return torch.tensor(corrupted), torch.tensor(hvf_clean), laterality

# ============== Enhanced VF Decoder ==============
class VFAutoDecoder(nn.Module):
    """Simplified decoder without attention (MPS-friendly and matches training)"""
    def __init__(self, input_dim: int = 52):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, input_dim)
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        output = self.network(x)
        output = output + self.residual_weight * x
        return output

# ============== Loss Functions ==============
def compute_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, float]:
    """MAE loss with optional masking."""
    if mask is not None:
        # Only compute loss on valid (non-masked) values
        valid_pred = pred[mask]
        valid_target = target[mask]
        if valid_pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True), 0.0
        mae = torch.mean(torch.abs(valid_pred - valid_target))
    else:
        mae = torch.mean(torch.abs(pred - target))
    
    return mae, mae.item()

def compute_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """MSE loss with optional masking."""
    if mask is not None:
        valid_pred = pred[mask]
        valid_target = target[mask]
        if valid_pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.mean((valid_pred - valid_target) ** 2)
    return torch.mean((pred - target) ** 2)

def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Pearson correlation."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Filter out zeros (masked values)
    mask = (target_flat > 0) & (target_flat < MASKED_VALUE_THRESHOLD)
    if mask.sum() < 2:
        return 0.0
    
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
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
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for corrupted, target, _ in dataloader:
            corrupted = corrupted.to(DEVICE)
            target = target.to(DEVICE)
            
            pred = model(corrupted)
            
            # Create mask for valid values
            mask = (target > 0) & (target < MASKED_VALUE_THRESHOLD)
            
            _, mae = compute_mae(pred, target, mask)
            
            # Weight by number of valid points
            n_valid = mask.sum(dim=1).float()
            total_mae += mae * corrupted.size(0)
            total_samples += corrupted.size(0)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    corr = pearson_correlation(all_preds, all_targets)
    
    return avg_mae, corr

def train():
    """Main training loop."""
    print("=" * 60)
    print("Stage 1: Enhanced VF Auto-encoder Pre-training")
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
    print(f"Strategies: Random masking, Gaussian noise, Regional dropout")
    
    # Create model
    model = VFAutoDecoder(input_dim=52)
    model.to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing with warmup
    warmup_epochs = 5
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - warmup_epochs, eta_min=LR * 0.01
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for corrupted, target, _ in pbar:
            corrupted = corrupted.to(DEVICE)
            target = target.to(DEVICE)
            
            # Forward pass
            pred = model(corrupted)
            
            # Create mask for valid values
            mask = (target > 0) & (target < MASKED_VALUE_THRESHOLD)
            
            # Combined loss: MSE + MAE
            mse_loss = compute_mse(pred, target, mask)
            mae_loss, mae_val = compute_mae(pred, target, mask)
            loss = 0.7 * mse_loss + 0.3 * mae_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mae += mae_val
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae_val:.3f}'
            })
        
        scheduler.step()
        
        # Validation
        val_mae, val_corr = evaluate(model, val_loader)
        train_mae, train_corr = evaluate(model, train_loader)
        
        avg_epoch_mae = epoch_mae / n_batches if n_batches > 0 else 0
        
        print(f"\n[Epoch {epoch}]")
        print(f"  Train MAE: {train_mae:.3f} dB | Corr: {train_corr:.3f}")
        print(f"  Val MAE:   {val_mae:.3f} dB | Corr: {val_corr:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'val_corr': val_corr,
                'corruption_ratio': CORRUPTION_RATIO
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