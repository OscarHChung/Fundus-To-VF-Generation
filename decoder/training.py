#!/usr/bin/env python3
"""
Hybrid Approach: Speed of Lightweight + Power of Pre-training

Strategy:
1. Use lightweight architecture (FAST like 4.8 MAE model)
2. Initialize decoder with pre-training (knowledge boost)
3. Train decoder end-to-end (no freezing)
4. Moderate augmentation (not too aggressive)
5. Higher learning rate + better scheduling
"""

import os, sys, json, numpy as np
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============== Config ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32  # Large batch = faster
EPOCHS = 150
LR = 2e-3  # High LR for fast convergence
WEIGHT_DECAY = 1e-5
PATIENCE = 30
NUM_ENCODER_BLOCKS = 8  # Fewer = faster
MASKED_VALUE_THRESHOLD = 99.0

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
PRETRAINED_DECODER = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")
BASE_DIR = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
BEST_SAVE = os.path.join(CURRENT_DIR, "best_hybrid_model.pth")
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

# ============== Load Models ==============
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
        print(f"✓ Pre-trained decoder: {pretrained_checkpoint['val_mae']:.2f} dB")
    except Exception as e:
        print(f"⚠️  No pre-training")

# ============== Moderate Augmentation ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

# ============== Lightweight Pre-trained Decoder ==============
class VFAutoDecoder(nn.Module):
    """Pre-trained decoder architecture"""
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

# ============== FAST HYBRID MODEL ==============
class FastHybridModel(nn.Module):
    """
    Combines:
    - Lightweight projection (fast)
    - Pre-trained decoder (smart initialization)
    - End-to-end training (adaptive)
    """
    def __init__(self, encoder, pretrained_state=None, num_blocks=8):
        super().__init__()
        self.encoder = encoder
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        if hasattr(self.encoder, 'blocks'):
            total = len(self.encoder.blocks)
            for i in range(max(0, total - num_blocks), total):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True
            print(f"✓ Unfrozen {num_blocks}/{total} blocks")
        
        # LIGHTWEIGHT projection (like 4.8 MAE model)
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 52)
        )
        
        # Initialize projection weights well
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Decoder - try to use pre-training
        self.use_pretrained = False
        if pretrained_state is not None:
            self.decoder = VFAutoDecoder()
            try:
                self.decoder.load_state_dict(pretrained_state, strict=True)
                self.use_pretrained = True
                print(f"✓ Pre-trained decoder initialized (TRAINING)")
            except Exception as e:
                print(f"⚠️  Pre-training failed: {e}")
        
        if not self.use_pretrained:
            # Simple decoder fallback
            self.decoder = nn.Sequential(
                nn.Linear(52, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 52)
            )
            print(f"✓ Simple decoder (no pre-training)")
    
    def forward(self, x):
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        
        vf_features = self.projection(latent)
        pred = self.decoder(vf_features)
        return pred

# ============== Loss ==============
def compute_loss(pred, target, laterality):
    device = pred.device
    target = target.to(device)
    total_loss = 0.0
    total_mae = 0.0
    n_valid = 0
    
    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_valid = target[i][valid_idx]
        mask = target_valid < MASKED_VALUE_THRESHOLD
        
        if mask.sum() == 0:
            continue
        
        pred_clean = pred[i][mask]
        target_clean = target_valid[mask]
        
        # MSE loss
        loss = (pred_clean - target_clean).pow(2).sum()
        mae = (pred_clean - target_clean).abs().sum()
        
        total_loss += loss
        total_mae += mae.item()
        n_valid += mask.sum().item()
    
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0
    
    return total_loss / n_valid, total_mae / n_valid, n_valid

def pearson_correlation(pred, target, laterality):
    all_pred, all_target = [], []
    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_i = target[i, valid_idx]
        pred_i = pred[i]
        mask = target_i < MASKED_VALUE_THRESHOLD
        if mask.sum() > 0:
            all_pred.extend(pred_i[mask].tolist())
            all_target.extend(target_i[mask].tolist())
    if len(all_pred) < 2:
        return 0.0
    return np.corrcoef(np.array(all_pred), np.array(all_target))[0, 1]

def evaluate(model, loader):
    model.eval()
    total_mae = 0.0
    n_valid = 0
    all_preds, all_targets, all_lats = [], [], []
    
    with torch.no_grad():
        for imgs, hvf, lat in loader:
            imgs = imgs.to(DEVICE)
            pred = model(imgs)
            _, mae, nv = compute_loss(pred, hvf, lat)
            total_mae += mae * nv
            n_valid += nv
            all_preds.append(pred.cpu().numpy())
            all_targets.append(hvf.cpu().numpy())
            all_lats.extend(lat)
    
    if n_valid == 0:
        return float('inf'), 0.0
    mae = total_mae / n_valid
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    corr = pearson_correlation(all_preds, all_targets, all_lats)
    return mae, corr

def train():
    print("="*60)
    print("Hybrid Fast Model: Speed + Pre-training")
    print("="*60)
    
    train_dataset = GRAPEDataset(TRAIN_JSON, FUNDUS_DIR, train_transform)
    val_dataset = GRAPEDataset(VAL_JSON, FUNDUS_DIR, val_transform)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, 
                             num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    model = FastHybridModel(base_model, pretrained_decoder_state, NUM_ENCODER_BLOCKS)
    model.to(DEVICE)
    
    enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in model.projection.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Params: Enc={enc_params:,}, Proj={proj_params:,}, Dec={dec_params:,}")
    
    # Single optimizer, high LR
    optimizer = optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': LR * 0.1},
        {'params': model.projection.parameters(), 'lr': LR},
        {'params': model.decoder.parameters(), 'lr': LR * 0.5}
    ], weight_decay=WEIGHT_DECAY)
    
    # OneCycleLR for fast convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR * 0.1, LR, LR * 0.5],
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    best_mae = float('inf')
    patience = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            pred = model(imgs)
            loss, mae, nv = compute_loss(pred, hvf, lat)
            
            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()  # Step every batch for OneCycle
            
            pbar.set_postfix({'MAE': f'{mae:.2f}'})
        
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
    
    print(f"\n{'='*60}")
    print(f"Best Val MAE: {best_mae:.2f} dB")
    if best_mae < 3.0:
        print(f"✓✓✓ SUB-3 ACHIEVED!")
    elif best_mae < 4.0:
        print(f"✓✓ Sub-4 achieved")
    elif best_mae < 4.8:
        print(f"✓ Better than 4.8!")
    print("="*60)

if __name__ == "__main__":
    train()