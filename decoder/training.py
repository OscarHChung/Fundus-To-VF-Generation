"""
Training - Build on pretraining baseline
 - TO ADD
"""

import os, sys, json, numpy as np
import argparse
from typing import List, Tuple
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
from scipy import stats

# ============== MPS Configuration ==============
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✓ Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print(f"⚠️  MPS not available, using CPU")

# ============== Config ==============
BATCH_SIZE = 32
EPOCHS = 120
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 40
NUM_ENCODER_BLOCKS = 3
MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.3
USE_TTA = True
NUM_TTA_AUGS = 4
LABEL_SMOOTH = 0.0
USE_OUTLIER_CLIPPING = True
OUTLIER_CLIP_RANGE = (0, 35)

# MILD class imbalance handling (not aggressive!)
LOW_DB_THRESHOLD = 10.0
LOW_VALUE_WEIGHT = 2.0

# Decoder unfreezing
DECODER_UNFREEZE_EPOCH = 1

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
mask_OS = np.fliplr(mask_OD)
valid_indices_os = [i for i, v in enumerate(mask_OS.flatten()) if v]

# ============== Load RETFound ==============
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)
print(f"✓ Loaded RETFound")

pretrained_decoder_state = None
if os.path.exists(PRETRAINED_DECODER):
    try:
        with torch.serialization.safe_globals([np.dtype]):
            pretrained_checkpoint = torch.load(PRETRAINED_DECODER, map_location='cpu', weights_only=False)
        pretrained_decoder_state = pretrained_checkpoint['model_state_dict']
        print(f"✓ Pre-trained decoder: {pretrained_checkpoint['val_mae']:.2f} dB")
    except Exception as e:
        print(f"⚠️  Could not load pre-trained decoder: {e}")

# ============== Augmentation (Keep Simple) ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_tta_transforms():
    return [
        val_transform,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: transforms.functional.hflip(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: transforms.functional.rotate(img, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: transforms.functional.rotate(img, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

# ============== Dataset ==============
class MultiImageDataset(Dataset):
    def __init__(self, json_path: str, fundus_dir: str, transform, mode='train', use_tta=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.fundus_dir = fundus_dir
        self.transform = transform
        self.mode = mode
        self.use_tta = use_tta
        
        self.samples = []
        for item in self.data:
            images = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            hvf = item['hvf']
            laterality = item.get('Laterality', 'OD').strip().upper()
            patient_id = item.get('PatientID', 0)
            
            if self.mode == 'train':
                for img_path in images:
                    self.samples.append({
                        'image': img_path,
                        'hvf': hvf,
                        'laterality': laterality,
                        'patient_id': patient_id
                    })
            else:
                self.samples.append({
                    'images': images,
                    'hvf': hvf,
                    'laterality': laterality,
                    'patient_id': patient_id
                })
        
        if self.mode == 'train':
            print(f"  Train: {len(self.data)} eyes → {len(self.samples)} images")
        else:
            print(f"  Val: {len(self.data)} eyes with {sum(len(s['images']) for s in self.samples)} images")
            if use_tta:
                print(f"  TTA: Enabled ({NUM_TTA_AUGS}x augmentations)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.mode == 'train':
            img_path = os.path.join(self.fundus_dir, sample['image'])
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            hvf = np.array(sample['hvf'], dtype=np.float32).flatten()
            
            return img_tensor, torch.tensor(hvf), sample['laterality']
        
        else:
            all_augmented_images = []
            tta_transforms_list = get_tta_transforms()
            
            for img_path in sample['images']:
                img_full_path = os.path.join(self.fundus_dir, img_path)
                img = Image.open(img_full_path).convert('RGB')
                
                if self.use_tta:
                    for tta_transform in tta_transforms_list:
                        img_aug = tta_transform(img)
                        all_augmented_images.append(img_aug)
                else:
                    img_tensor = self.transform(img)
                    all_augmented_images.append(img_tensor)
            
            hvf = np.array(sample['hvf'], dtype=np.float32).flatten()
            images = torch.stack(all_augmented_images)
            
            return images, torch.tensor(hvf), sample['laterality']

def val_collate_fn(batch):
    return batch[0]

# ============== Model ==============
class VFAutoDecoder(nn.Module):
    def __init__(self, input_dim: int = 52):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, input_dim)
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        output = self.network(x)
        output = output + self.residual_weight * x
        return output

class MultiImageModel(nn.Module):
    def __init__(self, encoder, pretrained_state=None, num_blocks=6, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if hasattr(self.encoder, 'blocks'):
            total = len(self.encoder.blocks)
            for i in range(max(0, total - num_blocks), total):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = True
            print(f"✓ Unfrozen {num_blocks}/{total} encoder blocks")
        
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, 52)
        )
        
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.decoder = VFAutoDecoder(input_dim=52)
        
        self.use_pretrained = False
        self.decoder_frozen = True
        
        if pretrained_state is not None:
            try:
                self.decoder.load_state_dict(pretrained_state, strict=True)
                self.use_pretrained = True
                for param in self.decoder.parameters():
                    param.requires_grad = False
                print(f"✓ Pre-trained decoder loaded (will unfreeze at epoch {DECODER_UNFREEZE_EPOCH})")
            except Exception as e:
                print(f"⚠️  Decoder: {e}")
                self.decoder_frozen = False
        else:
            self.decoder_frozen = False
            print(f"✓ Training decoder from scratch")
    
    def unfreeze_decoder(self):
        if self.use_pretrained and self.decoder_frozen:
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.decoder_frozen = False
            print(f"  ✓ Decoder UNFROZEN")
            return True
        return False
    
    def forward(self, x, average_multi=True):
        if x.dim() == 4:
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
            if latent.dim() == 3:
                latent = latent[:, 0, :]
            
            vf_features = self.projection(latent)
            pred = self.decoder(vf_features)
            
            # CRITICAL FIX: Lower threshold to 0.1 (was 0.3-0.5 in baseline)
            # This allows low predictions (0-3 dB) to pass through
            pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
            
            if USE_OUTLIER_CLIPPING:
                pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
            
            if average_multi and pred.shape[0] > 1:
                pred = pred.mean(dim=0, keepdim=True)
            
            return pred
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

# ============== Loss with MILD Reweighting ==============
def compute_loss(pred, target, laterality, smooth=0.0):
    device = pred.device
    target = target.to(device)
    
    if isinstance(laterality, str):
        laterality = [laterality]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    
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
        
        # Mild class reweighting
        weights = torch.ones_like(pred_clean)
        low_mask = (target_clean < LOW_DB_THRESHOLD)
        weights[low_mask] = LOW_VALUE_WEIGHT
        
        # Label smoothing
        if smooth > 0:
            mean_val = target_clean.mean()
            target_clean = (1 - smooth) * target_clean + smooth * mean_val
        
        # Huber loss
        loss = F.huber_loss(pred_clean, target_clean, reduction='none', delta=1.0)
        loss = (loss * weights).mean()
        
        mae = (pred_clean - target_clean).abs().mean()
        
        total_loss += loss * mask.sum().item()
        total_mae += mae.item() * mask.sum().item()
        n_valid += mask.sum().item()
    
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0
    
    return total_loss / n_valid, total_mae / n_valid, n_valid

def pearson_correlation(pred, target, laterality):
    all_pred, all_target = [], []
    
    if isinstance(laterality, str):
        laterality = [laterality]
    
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    
    batch_size = min(len(laterality), pred.shape[0], target.shape[0])
    
    for i in range(batch_size):
        lat = laterality[i] if i < len(laterality) else laterality[0]
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        
        target_i = target[min(i, target.shape[0]-1)][valid_idx]
        mask = target_i < MASKED_VALUE_THRESHOLD
        
        if mask.sum() > 0:
            pred_i = pred[min(i, pred.shape[0]-1)]
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
        for sample in loader:
            imgs, hvf, lat = sample
            imgs = imgs.to(DEVICE)
            pred = model(imgs, average_multi=True)
            
            _, mae, nv = compute_loss(pred, hvf, lat)
            total_mae += mae * nv
            n_valid += nv
            
            all_preds.append(pred.cpu().numpy().flatten())
            all_targets.append(hvf.cpu().numpy().flatten())
            all_lats.append(lat)
    
    if n_valid == 0:
        return float('inf'), 0.0
    
    mae = total_mae / n_valid
    all_preds = np.stack(all_preds)
    all_targets = np.stack(all_targets)
    corr = pearson_correlation(torch.tensor(all_preds), torch.tensor(all_targets), all_lats)
    
    return mae, corr

# ============== Training ==============
def train():
    print("="*60)
    print("CONSERVATIVE Training - Build on 3.74 dB Baseline")
    print("="*60)
    print(f"\nMinimal Changes:")
    print(f"  1. Unfreeze decoder at epoch {DECODER_UNFREEZE_EPOCH} (gentle LR)")
    print(f"  2. Lower threshold: 0.1 dB (was 0.3-0.5) - allows low predictions")
    print(f"  3. Mild 2x weight on low values (not 10x!)")
    print(f"  4. Keep everything else from baseline that was working")
    
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train', use_tta=False)
    val_dataset = MultiImageDataset(VAL_JSON, FUNDUS_DIR, val_transform, mode='val', use_tta=USE_TTA)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, 
                             num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=0, collate_fn=val_collate_fn)
    
    model = MultiImageModel(base_model, pretrained_decoder_state, NUM_ENCODER_BLOCKS, DROPOUT_RATE)
    model.to(DEVICE)

    # Unfreeze decoder immediately and include in optimizer from epoch 1
    model.unfreeze_decoder()
    decoder_unfrozen = True
    
    # Initial optimizer
    optimizer = optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad],
        'lr': BASE_LR * 0.05, 'weight_decay': WEIGHT_DECAY * 2},
        {'params': model.projection.parameters(),
        'lr': BASE_LR * 1.5, 'weight_decay': WEIGHT_DECAY},
        {'params': model.decoder.parameters(),
        'lr': BASE_LR * 0.08, 'weight_decay': WEIGHT_DECAY * 0.5},
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_mae = float('inf')
    patience = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            
            pred = model(imgs, average_multi=False)
            loss, mae, nv = compute_loss(pred, hvf, lat, smooth=LABEL_SMOOTH)
            
            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            pbar.set_postfix({'MAE': f'{mae:.2f}'})
        
        scheduler.step()
        
        val_mae, val_corr = evaluate(model, val_loader)
        
        if epoch % 5 == 0:
            train_mae, train_corr = evaluate(model, DataLoader(
                MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, val_transform, mode='val', use_tta=False),
                batch_size=1, shuffle=False, collate_fn=val_collate_fn
            ))
            gap = train_mae - val_mae
            status = "[FROZEN]" if not decoder_unfrozen else "[ACTIVE]"
            print(f"\n[Epoch {epoch}] {status}")
            print(f"  Train: {train_mae:.2f} dB | Corr: {train_corr:.3f}")
            print(f"  Val:   {val_mae:.2f} dB | Corr: {val_corr:.3f} | Gap: {gap:+.2f}")
        
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
            
            if epoch % 5 == 0:
                print(f"  ✓ New Best!")
            
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Val MAE: {best_mae:.2f} dB")
    
    baseline_gain = 3.74 - best_mae
    
    if best_mae < 3.0:
        print(f"\n SUB-3 MAE ACHIEVED! Beat baseline by {baseline_gain:.2f} dB")
    elif best_mae < 3.5:
        print(f"\n SUB-3.5 ACHIEVED! Beat baseline by {baseline_gain:.2f} dB")
    else:
        print(f"\n Gap to baseline: {best_mae - 3.74:+.2f} dB") # 3.74db previous baseline
    
    print(f"\nModel saved to: {INFERENCE_SAVE}")
    print("="*60)

if __name__ == "__main__":
    train()
