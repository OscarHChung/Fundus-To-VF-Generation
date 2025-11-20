#!/usr/bin/env python3
"""
Complete Training Script for Multi-Image Dataset
Target: Sub-3 MAE

Key improvements:
1. Handles multiple fundus images per eye
2. Uses all images during training (data augmentation)
3. Averages predictions from multiple images during validation
4. Optimized for larger effective dataset
"""

import os, sys, json, numpy as np
import argparse
from typing import List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============== Config ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32  # Can be larger now with more data
EPOCHS = 200
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 40
NUM_ENCODER_BLOCKS = 10
MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.35  # Less aggressive since we have more data

# Multi-image strategy
USE_MULTI_IMAGE_AVERAGE = True  # Average predictions from multiple images
SAMPLE_STRATEGY = 'all'  # 'all' or 'random' - use all images or sample one per epoch

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
PRETRAINED_DECODER = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")
BASE_DIR = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
BEST_SAVE = os.path.join(CURRENT_DIR, "best_multi_image_model.pth")
INFERENCE_SAVE = os.path.join(CURRENT_DIR, "inference_model.pth")

print(f"Using training data: {TRAIN_JSON}")
print(f"Using validation data: {VAL_JSON}")

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

# ============== Load RETFound ==============
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)
print(f"✓ Loaded RETFound")

# Load pre-trained decoder
pretrained_decoder_state = None
if os.path.exists(PRETRAINED_DECODER):
    try:
        with torch.serialization.safe_globals([np.dtype]):
            pretrained_checkpoint = torch.load(PRETRAINED_DECODER, map_location='cpu', weights_only=False)
        pretrained_decoder_state = pretrained_checkpoint['model_state_dict']
        print(f"✓ Pre-trained decoder: {pretrained_checkpoint['val_mae']:.2f} dB")
    except Exception as e:
        print(f"⚠️  Could not load pre-trained decoder: {e}")

# ============== Augmentation ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(8),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============== Multi-Image Dataset ==============
class MultiImageDataset(Dataset):
    """
    Dataset that handles multiple fundus images per eye
    
    For training: Returns one random image per eye per epoch
    For validation: Can return all images for averaging
    """
    def __init__(self, json_path: str, fundus_dir: str, transform, mode='train'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.fundus_dir = fundus_dir
        self.transform = transform
        self.mode = mode
        
        # Expand dataset - create one entry per image
        self.samples = []
        for item in self.data:
            images = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            hvf = item['hvf']
            laterality = item.get('Laterality', 'OD').strip().upper()
            patient_id = item.get('PatientID', 0)
            
            for img_path in images:
                self.samples.append({
                    'image': img_path,
                    'hvf': hvf,
                    'laterality': laterality,
                    'patient_id': patient_id,
                    'n_images': len(images)
                })
        
        print(f"  Loaded {len(self.data)} eyes with {len(self.samples)} total images")
        
        # Group by patient_id and laterality for validation averaging
        self.grouped_samples = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            key = (sample['patient_id'], sample['laterality'])
            self.grouped_samples[key].append(idx)
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.samples)
        else:
            return len(self.grouped_samples)  # One entry per eye
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            # Training: return single image
            sample = self.samples[idx]
            img_path = os.path.join(self.fundus_dir, sample['image'])
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            hvf = np.array(sample['hvf'], dtype=np.float32).flatten()
            
            return img_tensor, torch.tensor(hvf), sample['laterality']
        
        else:
            # Validation: return all images for this eye
            keys = list(self.grouped_samples.keys())
            key = keys[idx]
            indices = self.grouped_samples[key]
            
            images = []
            hvf = None
            laterality = None
            
            for i in indices:
                sample = self.samples[i]
                img_path = os.path.join(self.fundus_dir, sample['image'])
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                images.append(img_tensor)
                
                if hvf is None:
                    hvf = np.array(sample['hvf'], dtype=np.float32).flatten()
                    laterality = sample['laterality']
            
            # Stack images
            images = torch.stack(images)  # [N, 3, 224, 224]
            
            return images, torch.tensor(hvf), laterality

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

class MultiImageModel(nn.Module):
    def __init__(self, encoder, pretrained_state=None, num_blocks=10, dropout=0.35):
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
            print(f"✓ Unfrozen {num_blocks}/{total} encoder blocks")
        
        # Projection
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
        
        # Decoder
        self.use_pretrained = False
        if pretrained_state is not None:
            self.decoder = VFAutoDecoder()
            try:
                self.decoder.load_state_dict(pretrained_state, strict=True)
                self.use_pretrained = True
                print(f"✓ Pre-trained decoder loaded")
            except Exception as e:
                print(f"⚠️  Could not load decoder: {e}")
        
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
        """
        Args:
            x: Either [B, 3, 224, 224] for single images
               or [B, N, 3, 224, 224] for multiple images per eye
        """
        if x.dim() == 5:
            # Multiple images: [B, N, 3, 224, 224]
            B, N = x.shape[0], x.shape[1]
            x = x.view(B * N, *x.shape[2:])  # [B*N, 3, 224, 224]
            
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
            if latent.dim() == 3:
                latent = latent[:, 0, :]
            
            vf_features = self.projection(latent)
            pred = self.decoder(vf_features)
            
            # Reshape and average
            pred = pred.view(B, N, -1)  # [B, N, 52]
            pred = pred.mean(dim=1)  # [B, 52]
            
            return pred
        else:
            # Single image: [B, 3, 224, 224]
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
            if latent.dim() == 3:
                latent = latent[:, 0, :]
            
            vf_features = self.projection(latent)
            pred = self.decoder(vf_features)
            
            return pred

# ============== Loss & Metrics ==============
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

# ============== Training ==============
def train():
    print("="*60)
    print("Multi-Image Training for Sub-3 MAE")
    print("="*60)
    
    # Load datasets
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset = MultiImageDataset(VAL_JSON, FUNDUS_DIR, val_transform, mode='val')
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, 
                             num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Strategy: Use all images during training, average during validation")
    
    # Create model
    model = MultiImageModel(base_model, pretrained_decoder_state, NUM_ENCODER_BLOCKS, DROPOUT_RATE)
    model.to(DEVICE)
    
    enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in model.projection.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Trainable: Enc={enc_params:,}, Proj={proj_params:,}, Dec={dec_params:,}")
    
    # Optimizer
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
        pct_start=0.1
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                scheduler.step()
            
            pbar.set_postfix({'MAE': f'{mae:.2f}'})
        
        # Validation
        val_mae, val_corr = evaluate(model, val_loader)
        
        if epoch % 5 == 0:
            train_mae, train_corr = evaluate(model, train_loader)
            print(f"\n[Epoch {epoch}]")
            print(f"  Train: {train_mae:.2f} dB | Corr: {train_corr:.3f}")
            print(f"  Val:   {val_mae:.2f} dB | Corr: {val_corr:.3f}")
        
        if val_mae < best_mae:
            improvement = best_mae - val_mae
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
            
            if epoch % 5 == 0:
                print(f"  ✓ Best! Improved {improvement:.2f} dB")
            
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Val MAE: {best_mae:.2f} dB")
    
    if best_mae < 3.0:
        print(f"✓✓✓ SUB-3 MAE ACHIEVED!")
    elif best_mae < 3.5:
        print(f"✓✓ Close to sub-3!")
    elif best_mae < 4.0:
        print(f"✓ Sub-4 achieved")
    
    print(f"Model saved to: {INFERENCE_SAVE}")
    print("="*60)

if __name__ == "__main__":
    train()
