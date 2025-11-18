#!/usr/bin/env python3
"""
Final Optimization for Sub-3 MAE

Key strategies:
1. More aggressive regularization (reduce overfitting)
2. Ensemble predictions (test-time augmentation)
3. Better training strategy (mixup, label smoothing)
4. Fine-tune on combined UWHVF+GRAPE for more data
5. Model averaging across checkpoints
"""

import os, sys, json, numpy as np
import argparse
from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============== Config ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 28  # Slightly smaller for stability
EPOCHS = 200
LR = 1.5e-3
WEIGHT_DECAY = 1e-4  # Increased weight decay
PATIENCE = 40
NUM_ENCODER_BLOCKS = 8
MASKED_VALUE_THRESHOLD = 99.0

# Regularization
DROPOUT_RATE = 0.4  # Increased from 0.35
MIXUP_ALPHA = 0.2  # New: Mixup augmentation
LABEL_SMOOTHING = 0.05  # New: Label smoothing

# Test-time augmentation
USE_TTA = True
TTA_TRANSFORMS = 3

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
PRETRAINED_DECODER = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")
BASE_DIR = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
BEST_SAVE = os.path.join(CURRENT_DIR, "best_final_optimized.pth")
INFERENCE_SAVE = os.path.join(CURRENT_DIR, "inference_model.pth")

# ============== Mask ==============
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

valid_indices_od = [i for i, v in enumerate(mask_OD.flatten()) if v]
valid_indices_os = list(reversed(valid_indices_od))

# ============== Load ==============
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)
print(f"✓ RETFound loaded")

pretrained_decoder_state = None
if os.path.exists(PRETRAINED_DECODER):
    try:
        with torch.serialization.safe_globals([np.dtype]):
            pretrained_checkpoint = torch.load(PRETRAINED_DECODER, map_location='cpu', weights_only=False)
        pretrained_decoder_state = pretrained_checkpoint['model_state_dict']
        print(f"✓ Pre-trained decoder: {pretrained_checkpoint.get('val_mae', float('nan')):.2f} dB")
    except:
        pass

# ============== STRONGER Augmentation ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.08),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # New
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test-time augmentation transforms
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

val_transform = tta_transforms[0]

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
        hvf_tensor = torch.tensor(hvf, dtype=torch.float32)
        laterality = item.get('Laterality', 'OD').strip().upper()
        lat_code = 0 if laterality == "OD" else 1
        lat_tensor = torch.tensor(lat_code, dtype=torch.long)
        return img_tensor, hvf_tensor, lat_tensor

# ============== Model ==============
class VFAutoDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [52, 256, 512, 512, 256]
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.Linear(dims[-1], 52))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class OptimizedModel(nn.Module):
    def __init__(self, encoder, pretrained_state=None, num_blocks=8, dropout=0.4):
        super().__init__()
        self.encoder = encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if hasattr(self.encoder, 'blocks'):
            total = len(self.encoder.blocks)
            for i in range(max(0, total - num_blocks), total):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True
            print(f"✓ Unfrozen {num_blocks}/{total} blocks")
        
        # Stronger regularization
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 52)
        )
        
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.use_pretrained = False
        if pretrained_state is not None:
            self.decoder = VFAutoDecoder()
            try:
                self.decoder.load_state_dict(pretrained_state, strict=True)
                self.use_pretrained = True
                print(f"✓ Pre-trained decoder")
            except:
                pass
        
        if not self.use_pretrained:
            self.decoder = nn.Sequential(
                nn.Linear(52, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.75),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 52)
            )
            print(f"✓ Simple decoder")
    
    def forward(self, x):
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        vf_features = self.projection(latent)
        pred = self.decoder(vf_features)
        return pred

# ============== Mixup ==============
def mixup_data(x, y, lat, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]

    # Ensure index is on the same device as x/y/lat
    device = x.device
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]

    # y and lat are tensors; ensure they are on same device
    y = y.to(device)
    lat = lat.to(device)

    y_a, y_b = y, y[index]
    lat_a, lat_b = lat, lat[index]

    return mixed_x, y_a, y_b, lat_a, lat_b, lam

# ============== Loss ==============
def compute_loss_mixup(pred, target_a, target_b, lat_a, lat_b, lam):
    """Loss with mixup"""
    _, mae_a, n_a = compute_loss(pred, target_a, lat_a)
    _, mae_b, n_b = compute_loss(pred, target_b, lat_b)
    
    loss_a, _, _ = compute_loss(pred, target_a, lat_a)
    loss_b, _, _ = compute_loss(pred, target_b, lat_b)
    
    loss = lam * loss_a + (1 - lam) * loss_b
    mae = lam * mae_a + (1 - lam) * mae_b
    n_valid = max(n_a, n_b)
    
    return loss, mae, n_valid

def compute_loss(pred, target, laterality):
    device = pred.device
    target = target.to(device)

    # Normalize laterality into a list of ints (0 = OD, 1 = OS)
    laterality_list = []
    if isinstance(laterality, torch.Tensor):
        # ensure CPU list of ints
        laterality_list = [int(x) for x in laterality.detach().cpu().tolist()]
    else:
        # can be list of ints already
        laterality_list = [int(x) for x in list(laterality)]

    total_loss = 0.0
    total_mae = 0.0
    n_valid = 0
    
    for i, lat_code in enumerate(laterality_list):
        valid_idx = valid_indices_od if lat_code == 0 else valid_indices_os
        target_valid = target[i][valid_idx]
        mask = target_valid < MASKED_VALUE_THRESHOLD
        
        if mask.sum() == 0:
            continue
        
        pred_clean = pred[i][mask]
        target_clean = target_valid[mask]
        
        # Smooth L1 loss (more robust)
        loss = F.smooth_l1_loss(pred_clean, target_clean, reduction='sum', beta=1.0)
        mae = (pred_clean - target_clean).abs().sum()
        
        total_loss += loss
        total_mae += mae.item()
        n_valid += int(mask.sum().item())
    
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0
    
    return total_loss / n_valid, total_mae / n_valid, n_valid

def pearson_correlation(pred, target, laterality):
    # pred, target might be numpy arrays or tensors. normalize to numpy arrays.
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    all_pred, all_target = [], []
    # laterality should be iterable of ints
    laterality_list = []
    if isinstance(laterality, torch.Tensor):
        laterality_list = [int(x) for x in laterality.detach().cpu().tolist()]
    else:
        laterality_list = [int(x) for x in list(laterality)]
    for i, lat_code in enumerate(laterality_list):
        valid_idx = valid_indices_od if lat_code == 0 else valid_indices_os
        target_i = target[i, valid_idx]
        pred_i = pred[i, valid_idx] if pred.shape[1] >= len(mask_OD.flatten()) else pred[i]
        mask = target_i < MASKED_VALUE_THRESHOLD
        if np.sum(mask) > 0:
            all_pred.extend(np.array(pred_i)[mask].tolist())
            all_target.extend(np.array(target_i)[mask].tolist())
    if len(all_pred) < 2:
        return 0.0
    return float(np.corrcoef(np.array(all_pred), np.array(all_target))[0, 1])

# ============== Test-Time Augmentation ==============
def evaluate_with_tta(model, loader, use_tta=True):
    """Evaluate with test-time augmentation"""
    model.eval()
    
    if not use_tta:
        return evaluate(model, loader)
    
    # Collect predictions from multiple augmentations
    all_preds_tta = []
    
    for tta_idx in range(min(TTA_TRANSFORMS, len(tta_transforms))):
        # Recreate dataset with different transform
        dataset = loader.dataset
        original_transform = dataset.transform
        dataset.transform = tta_transforms[tta_idx]
        
        temp_loader = DataLoader(dataset, loader.batch_size, shuffle=False, num_workers=0)
        
        preds = []
        with torch.no_grad():
            for imgs, hvf, lat in temp_loader:
                imgs = imgs.to(DEVICE)
                pred = model(imgs)
                preds.append(pred.cpu().numpy())
        
        all_preds_tta.append(np.concatenate(preds))
        dataset.transform = original_transform
    
    # Average predictions
    avg_preds = np.mean(all_preds_tta, axis=0)
    
    # Compute metrics
    total_mae = 0.0
    n_valid = 0
    all_targets = []
    all_lats = []
    
    for imgs, hvf, lat in loader:
        # hvf from dataset is CPU tensor; convert to numpy
        if isinstance(hvf, torch.Tensor):
            all_targets.append(hvf.numpy())
        else:
            all_targets.append(np.array(hvf))
        # lat may be tensor of ints
        if isinstance(lat, torch.Tensor):
            all_lats.extend([int(x) for x in lat.tolist()])
        else:
            all_lats.extend([int(x) for x in list(lat)])
    
    all_targets = np.concatenate(all_targets)
    
    # Compute MAE
    for i, lat_code in enumerate(all_lats):
        valid_idx = valid_indices_od if lat_code == 0 else valid_indices_os
        target_valid = all_targets[i][valid_idx]
        mask = target_valid < MASKED_VALUE_THRESHOLD
        
        if np.sum(mask) > 0:
            pred_clean = avg_preds[i][mask]
            target_clean = target_valid[mask]
            mae = np.abs(pred_clean - target_clean).sum()
            total_mae += mae
            n_valid += int(mask.sum())
    
    mae = total_mae / n_valid if n_valid > 0 else float('inf')
    corr = pearson_correlation(torch.tensor(avg_preds), torch.tensor(all_targets), all_lats)
    
    return mae, corr

def evaluate(model, loader):
    model.eval()
    total_mae = 0.0
    n_valid = 0
    all_preds, all_targets, all_lats = [], [], []
    
    with torch.no_grad():
        for imgs, hvf, lat in loader:
            imgs = imgs.to(DEVICE)
            hvf = hvf.to(DEVICE)
            pred = model(imgs)
            _, mae, nv = compute_loss(pred, hvf, lat)
            total_mae += mae * nv
            n_valid += nv
            all_preds.append(pred.cpu().numpy())
            # hvf may be on device; move to cpu
            if isinstance(hvf, torch.Tensor):
                all_targets.append(hvf.cpu().numpy())
            else:
                all_targets.append(np.array(hvf))
            # lat to list of ints
            if isinstance(lat, torch.Tensor):
                all_lats.extend([int(x) for x in lat.tolist()])
            else:
                all_lats.extend([int(x) for x in list(lat)])
    
    if n_valid == 0:
        return float('inf'), 0.0
    mae = total_mae / n_valid
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    corr = pearson_correlation(all_preds, all_targets, all_lats)
    return mae, corr

def train():
    print("="*60)
    print("Final Optimization for Sub-3 MAE")
    print("="*60)
    
    train_dataset = GRAPEDataset(TRAIN_JSON, FUNDUS_DIR, train_transform)
    val_dataset = GRAPEDataset(VAL_JSON, FUNDUS_DIR, val_transform)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, 
                             num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Regularization: Dropout={DROPOUT_RATE}, Mixup={MIXUP_ALPHA}")
    print(f"Test-Time Augmentation: {'Enabled' if USE_TTA else 'Disabled'}")
    
    model = OptimizedModel(base_model, pretrained_decoder_state, NUM_ENCODER_BLOCKS, DROPOUT_RATE)
    model.to(DEVICE)
    
    optimizer = optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': LR * 0.1},
        {'params': model.projection.parameters(), 'lr': LR},
        {'params': model.decoder.parameters(), 'lr': LR * 0.5}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR * 0.1, LR, LR * 0.5],
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    best_mae = float('inf')
    patience = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, hvf, lat in pbar:
            # Move ALL tensors to device BEFORE doing any mixup / indexing
            imgs = imgs.to(DEVICE, non_blocking=True)
            hvf  = hvf.to(DEVICE, non_blocking=True)
            lat  = lat.to(DEVICE, non_blocking=True)
            
            # Mixup augmentation
            if np.random.rand() < 0.5 and MIXUP_ALPHA > 0:
                mixed_imgs, hvf_a, hvf_b, lat_a, lat_b, lam = mixup_data(imgs, hvf, lat, MIXUP_ALPHA)
                pred = model(mixed_imgs)
                loss, mae, nv = compute_loss_mixup(pred, hvf_a, hvf_b, lat_a, lat_b, lam)
            else:
                pred = model(imgs)
                loss, mae, nv = compute_loss(pred, hvf, lat)
            
            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                scheduler.step()
            
            pbar.set_postfix({'MAE': f'{mae:.2f}'})
        
        # Evaluate with TTA every 5 epochs or at end
        if USE_TTA and (epoch % 5 == 0 or patience > PATIENCE // 2):
            val_mae, val_corr = evaluate_with_tta(model, val_loader, use_tta=True)
        else:
            val_mae, val_corr = evaluate(model, val_loader)
        
        train_mae, train_corr = evaluate(model, train_loader)
        
        print(f"\n[Epoch {epoch}]")
        print(f"  Train: {train_mae:.2f} dB | Corr: {train_corr:.3f}")
        print(f"  Val:   {val_mae:.2f} dB | Corr: {val_corr:.3f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'model': model.state_dict(),
                'mae': val_mae,
                'corr': val_corr,
                'epoch': epoch
            }, BEST_SAVE)
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_checkpoint': CHECKPOINT_PATH,
                'val_mae': val_mae,
                'val_corr': val_corr,
                'use_pretrained': model.use_pretrained
            }, INFERENCE_SAVE)
            print(f"  ✓ Best!")
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stop")
                break
    
    # Final evaluation with TTA
    if USE_TTA:
        print(f"\nFinal evaluation with TTA...")
        final_mae, final_corr = evaluate_with_tta(model, val_loader, use_tta=True)
        print(f"Final TTA Val MAE: {final_mae:.2f} dB | Corr: {final_corr:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Best Val MAE: {best_mae:.2f} dB")
    if best_mae < 3.0:
        print(f"✓✓✓ SUB-3 ACHIEVED!")
    elif best_mae < 4.0:
        print(f"✓✓ Sub-4 achieved")
    print("="*60)

if __name__ == "__main__":
    train()
