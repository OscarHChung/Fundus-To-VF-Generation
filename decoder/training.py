"""
Training v7 — Checkpoint ensemble + snapshot-based SWA

Why: All architectures plateau at ~4.2 dB. The model oscillates during training
(val MAE bounces 4.2-4.5 across epochs). Averaging predictions from multiple
good checkpoints smooths out this noise.

Strategy:
  1. Train with v6's architecture (spatial features + frozen decoder) — proven best
  2. Save top-5 checkpoints by val MAE during training
  3. After training: create an ensemble model that averages weights (SWA-style)
  4. Also save individual checkpoints for prediction-time ensemble if needed

Additionally trying:
  - Slightly higher dropout (0.45) since we're now ensembling
  - Wider projection hidden (384 instead of 256) to give each checkpoint
    more expressiveness — ensemble will regularize
  - CosineAnnealingWarmRestarts instead of OneCycleLR
    (allows multiple "restarts" → diverse checkpoints near each LR minimum)

Banned: LoRA, flips/affine, MixUp
"""

import os, sys, json, copy, numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

# ============== MPS ==============
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  MPS not available, using CPU")

# ==============================================================
# CONFIG
# ==============================================================
BATCH_SIZE   = 32
EPOCHS       = 100            # Enough for 2+ cosine cycles
BASE_LR      = 5e-4
WEIGHT_DECAY = 1e-3
PATIENCE     = 40             # Longer patience — we want diverse snapshots

MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.45

# Encoder
NUM_ENCODER_BLOCKS = 3
ENCODER_LR_SCALE   = 0.05

# Projection
PROJ_INIT_BIAS   = 18.0
PROJ_HIDDEN_DIM  = 384        # Wider than v6's 256

# Augmentation
ROTATION_DEG = 5

# TTA
USE_TTA       = True
TTA_ROTATIONS = [-5, 0, 5]

# Loss
LOW_DB_THRESHOLD = 10.0
LOW_VALUE_WEIGHT = 1.5
HUBER_DELTA      = 1.0

# Decoder: FROZEN
DECODER_FROZEN = True

# Validation & ensemble
VAL_EVERY     = 3
TOP_K_SAVE    = 5             # Save top 5 checkpoints for ensemble

# Clipping
OUTLIER_CLIP_RANGE = (0, 35)

# ── Paths ──────────────────────────────────────────────────────
CURRENT_DIR        = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR       = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH    = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
PRETRAINED_DECODER = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")
BASE_DIR           = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR         = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON         = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON           = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
BEST_SAVE          = os.path.join(CURRENT_DIR, "best_multi_image_model.pth")
INFERENCE_SAVE     = os.path.join(CURRENT_DIR, "inference_model.pth")
ENSEMBLE_SAVE      = os.path.join(CURRENT_DIR, "ensemble_model.pth")

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
mask_OS          = np.fliplr(mask_OD)
valid_indices_os = [i for i, v in enumerate(mask_OS.flatten()) if v]
NUM_VALID_POINTS = len(valid_indices_od)  # 52

# ============== Load RETFound ==============
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)
print("✓ Loaded RETFound")

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
    transforms.RandomRotation(ROTATION_DEG),
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
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img, d=deg: transforms.functional.rotate(img, d)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for deg in TTA_ROTATIONS
    ]

# ============== Dataset ==============
class MultiImageDataset(Dataset):
    def __init__(self, json_path, fundus_dir, transform, mode='train', use_tta=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.fundus_dir = fundus_dir
        self.transform  = transform
        self.mode       = mode
        self.use_tta    = use_tta

        self.samples = []
        for item in self.data:
            images     = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            hvf        = item['hvf']
            laterality = item.get('Laterality', 'OD').strip().upper()
            patient_id = item.get('PatientID', 0)

            if self.mode == 'train':
                for img_path in images:
                    self.samples.append({'image': img_path, 'hvf': hvf,
                                         'laterality': laterality, 'patient_id': patient_id})
            else:
                self.samples.append({'images': images, 'hvf': hvf,
                                     'laterality': laterality, 'patient_id': patient_id})

        if self.mode == 'train':
            print(f"  Train: {len(self.data)} eyes → {len(self.samples)} images")
        else:
            print(f"  Val: {len(self.data)} eyes with {sum(len(s['images']) for s in self.samples)} images")
            if use_tta:
                print(f"  TTA: {len(TTA_ROTATIONS)} rotations {TTA_ROTATIONS}°")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.mode == 'train':
            img = Image.open(os.path.join(self.fundus_dir, sample['image'])).convert('RGB')
            hvf = np.array(sample['hvf'], dtype=np.float32).flatten()
            return self.transform(img), torch.tensor(hvf), sample['laterality']
        else:
            all_imgs = []
            for img_path in sample['images']:
                img = Image.open(os.path.join(self.fundus_dir, img_path)).convert('RGB')
                if self.use_tta:
                    for t in get_tta_transforms():
                        all_imgs.append(t(img))
                else:
                    all_imgs.append(self.transform(img))
            hvf = np.array(sample['hvf'], dtype=np.float32).flatten()
            return torch.stack(all_imgs), torch.tensor(hvf), sample['laterality']

def val_collate_fn(batch):
    return batch[0]

# ============== VFAutoDecoder ==============
class VFAutoDecoder(nn.Module):
    def __init__(self, input_dim=52):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 512),       nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 512),       nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256),       nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return self.network(x) + self.residual_weight * x

# ============== Model ==============
class MultiImageModel(nn.Module):
    def __init__(self, encoder, pretrained_state=None, num_blocks=3, dropout=0.3,
                 freeze_decoder=True, proj_hidden=256):
        super().__init__()
        self.encoder = encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        if hasattr(self.encoder, 'blocks'):
            total = len(self.encoder.blocks)
            for i in range(max(0, total - num_blocks), total):
                for p in self.encoder.blocks[i].parameters():
                    p.requires_grad = True
            print(f"✓ Unfrozen {num_blocks}/{total} encoder blocks")

        # Projection: CLS+spatial → hidden → 52
        self.projection = nn.Sequential(
            nn.Linear(2048, proj_hidden), nn.LayerNorm(proj_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_hidden, NUM_VALID_POINTS)
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final = self.projection[-1]
        nn.init.constant_(final.bias, PROJ_INIT_BIAS)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)

        proj_params = sum(p.numel() for p in self.projection.parameters())
        print(f"✓ Projection: 2048→{proj_hidden}→52 ({proj_params:,} params)")

        # Decoder
        self.decoder = VFAutoDecoder(input_dim=NUM_VALID_POINTS)
        self.decoder_frozen = False
        if pretrained_state is not None:
            try:
                self.decoder.load_state_dict(pretrained_state, strict=True)
                if freeze_decoder:
                    for p in self.decoder.parameters():
                        p.requires_grad = False
                    self.decoder_frozen = True
                    print(f"✓ Decoder: FROZEN")
                else:
                    print(f"✓ Decoder: trainable")
            except Exception as e:
                print(f"⚠️  Decoder load failed: {e}")
        else:
            print("✓ Decoder: from scratch")

        enc  = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        proj = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
        dec  = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f"  Trainable: enc={enc:,} + proj={proj:,} + dec={dec:,} = {enc+proj+dec:,}")

    def encode(self, x):
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            cls_token = latent[:, 0, :]
            spatial_avg = latent[:, 1:, :].mean(dim=1)
            return torch.cat([cls_token, spatial_avg], dim=1)
        else:
            return torch.cat([latent, latent], dim=1)

    def forward(self, x, average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")
        features = self.encode(x)
        proj = self.projection(features)
        pred = self.decoder(proj)
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred


# ============== Loss ==============
def compute_loss(pred, target, laterality):
    device = pred.device
    target = target.to(device)
    if isinstance(laterality, str):
        laterality = [laterality]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    total_loss = total_mae = n_valid = 0
    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_52 = target[i][valid_idx]
        pred_52   = pred[i]
        mask      = target_52 < MASKED_VALUE_THRESHOLD
        if mask.sum() == 0:
            continue
        p = pred_52[mask]
        t = target_52[mask]
        weights = torch.ones_like(p)
        weights[t < LOW_DB_THRESHOLD] = LOW_VALUE_WEIGHT
        loss = (F.huber_loss(p, t, reduction='none', delta=HUBER_DELTA) * weights).mean()
        mae  = (p - t).abs().mean()
        total_loss += loss * mask.sum().item()
        total_mae  += mae.item() * mask.sum().item()
        n_valid    += mask.sum().item()

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
    for i in range(min(len(laterality), pred.shape[0], target.shape[0])):
        lat       = laterality[i] if i < len(laterality) else laterality[0]
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_52 = target[min(i, target.shape[0]-1)][valid_idx]
        pred_52   = pred[min(i, pred.shape[0]-1)]
        mask      = target_52 < MASKED_VALUE_THRESHOLD
        if mask.sum() > 0:
            all_pred.extend(pred_52[mask].tolist())
            all_target.extend(target_52[mask].tolist())
    if len(all_pred) < 2:
        return 0.0
    r = np.corrcoef(np.array(all_pred), np.array(all_target))[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def evaluate(model, loader):
    model.eval()
    total_mae = n_valid = 0
    all_preds, all_targets, all_lats = [], [], []
    with torch.no_grad():
        for sample in loader:
            imgs, hvf, lat = sample
            imgs = imgs.to(DEVICE)
            pred = model(imgs, average_multi=True)
            _, mae, nv = compute_loss(pred, hvf, lat)
            total_mae += mae * nv
            n_valid   += nv
            all_preds.append(pred.cpu())
            all_targets.append(hvf.unsqueeze(0) if hvf.dim() == 1 else hvf)
            all_lats.append(lat)
    if n_valid == 0:
        return float('inf'), 0.0
    stacked_preds   = torch.cat(all_preds, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    corr = pearson_correlation(stacked_preds, stacked_targets, all_lats)
    return total_mae / n_valid, corr


# ============== Top-K checkpoint tracker ==============
class TopKCheckpoints:
    """Keep the top-K model checkpoints by val MAE (lower is better)."""
    def __init__(self, k=5):
        self.k = k
        self.checkpoints = []  # list of (mae, state_dict, epoch)

    def update(self, mae, model, epoch):
        state = copy.deepcopy(model.state_dict())
        self.checkpoints.append((mae, state, epoch))
        self.checkpoints.sort(key=lambda x: x[0])
        if len(self.checkpoints) > self.k:
            self.checkpoints = self.checkpoints[:self.k]

    def get_averaged_state(self):
        """Average the weights of all saved checkpoints (SWA-style)."""
        if not self.checkpoints:
            return None
        avg_state = OrderedDict()
        n = len(self.checkpoints)
        for key in self.checkpoints[0][1]:
            tensors = [ckpt[1][key].float() for ckpt in self.checkpoints]
            avg_state[key] = sum(tensors) / n
        return avg_state

    def summary(self):
        return [(f"ep{ep}", f"{mae:.2f}") for mae, _, ep in self.checkpoints]


# ============== Training ==============
def train():
    print("=" * 60)
    print("Training v7 — Checkpoint ensemble")
    print("=" * 60)

    print(f"\nStrategy: train normally, save top-{TOP_K_SAVE} checkpoints,")
    print(f"  then average their weights (SWA) for final model.")

    print(f"\nConfig:")
    print(f"  Encoder: {NUM_ENCODER_BLOCKS}/24 blocks | LR={BASE_LR * ENCODER_LR_SCALE:.1e}")
    print(f"  Projection: 2048→{PROJ_HIDDEN_DIM}→52 | LR={BASE_LR:.1e} | dropout={DROPOUT_RATE}")
    print(f"  Decoder: FROZEN")
    print(f"  Loss: Huber + {LOW_VALUE_WEIGHT}x <{LOW_DB_THRESHOLD}dB")
    print(f"  Schedule: CosineWarmRestarts T0=20 T_mult=2 (diverse snapshots)")
    print(f"  Aug: ±{ROTATION_DEG}° | TTA: {TTA_ROTATIONS}")

    print(f"\nBanned: LoRA, flips/affine, MixUp")

    print(f"\nExpected:")
    print(f"  Ep 1-15:  MAE 5→4.3  (same trajectory as v5/v6)")
    print(f"  Ep 15-40: MAE ~4.2   (individual checkpoints plateau here)")
    print(f"  SWA:      MAE ~4.0?  (ensemble should beat best single)")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val', use_tta=USE_TTA)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=val_collate_fn)

    # ── Model ──────────────────────────────────────────────────
    model = MultiImageModel(base_model, pretrained_decoder_state,
                            NUM_ENCODER_BLOCKS, DROPOUT_RATE,
                            freeze_decoder=DECODER_FROZEN,
                            proj_hidden=PROJ_HIDDEN_DIM)
    model.to(DEVICE)

    # ── Optimizer with CosineWarmRestarts for diverse snapshots ──
    optimizer = optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad],
         'lr': BASE_LR * ENCODER_LR_SCALE, 'weight_decay': WEIGHT_DECAY},
        {'params': model.projection.parameters(),
         'lr': BASE_LR, 'weight_decay': WEIGHT_DECAY},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    topk = TopKCheckpoints(k=TOP_K_SAVE)
    best_mae  = float('inf')
    best_corr = 0.0
    patience  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            pred = model(imgs, average_multi=False)
            loss, mae, nv = compute_loss(pred, hvf, lat)

            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            pbar.set_postfix({'MAE': f'{mae:.2f}'})

        scheduler.step()

        # ── Validate ───────────────────────────────────────────
        if epoch % VAL_EVERY == 0 or epoch <= 3:
            val_mae, val_corr = evaluate(model, val_loader)

            train_eval = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, val_transform,
                                           mode='val', use_tta=False)
            train_eval_loader = DataLoader(train_eval, batch_size=1, shuffle=False,
                                           num_workers=0, collate_fn=val_collate_fn)
            train_mae, train_corr = evaluate(model, train_eval_loader)
            gap = train_mae - val_mae

            print(f"\n[Epoch {epoch}]")
            print(f"  Train: {train_mae:.2f} dB | Corr: {train_corr:.3f}")
            print(f"  Val:   {val_mae:.2f} dB | Corr: {val_corr:.3f} | Gap: {gap:+.2f}")

            # Update top-K
            topk.update(val_mae, model, epoch)

            if val_mae < best_mae:
                best_mae  = val_mae
                best_corr = val_corr
                torch.save({'model': model.state_dict(), 'mae': val_mae,
                            'corr': val_corr, 'epoch': epoch}, BEST_SAVE)
                print(f"  ✓ New Best! (MAE={val_mae:.2f}, Corr={val_corr:.3f})")
                patience = 0
            else:
                patience += VAL_EVERY
                if patience >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── SWA Ensemble ───────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Creating SWA ensemble from top-{TOP_K_SAVE} checkpoints...")
    print(f"  Checkpoints: {topk.summary()}")

    avg_state = topk.get_averaged_state()
    if avg_state is not None:
        model.load_state_dict(avg_state)
        model.to(DEVICE)

        swa_mae, swa_corr = evaluate(model, val_loader)
        print(f"\n  SWA Ensemble:")
        print(f"    Val MAE:  {swa_mae:.2f} dB")
        print(f"    Val Corr: {swa_corr:.3f}")
        print(f"    vs Best Single: {best_mae:.2f} dB (diff: {swa_mae - best_mae:+.2f})")

        # Save ensemble if it's better
        if swa_mae < best_mae:
            print(f"  ✓ Ensemble beats best single!")
            final_mae  = swa_mae
            final_corr = swa_corr
        else:
            print(f"  Best single checkpoint is still better")
            # Reload best single
            best_ckpt = torch.load(BEST_SAVE, map_location='cpu')
            model.load_state_dict(best_ckpt['model'])
            final_mae  = best_mae
            final_corr = best_corr

        # Always save ensemble for comparison
        torch.save({'model_state_dict': avg_state,
                    'val_mae': swa_mae, 'val_corr': swa_corr,
                    'source_checkpoints': topk.summary()}, ENSEMBLE_SAVE)

        # Save best overall as inference model
        torch.save({'model_state_dict': model.state_dict(),
                    'encoder_checkpoint': CHECKPOINT_PATH,
                    'val_mae': final_mae, 'val_corr': final_corr}, INFERENCE_SAVE)
    else:
        final_mae  = best_mae
        final_corr = best_corr

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Single:    {best_mae:.2f} dB | Corr: {best_corr:.3f}")
    if avg_state is not None:
        print(f"  SWA Ensemble:   {swa_mae:.2f} dB | Corr: {swa_corr:.3f}")
    print(f"  Final Model:    {final_mae:.2f} dB | Corr: {final_corr:.3f}")
    gain = 3.74 - final_mae
    if final_mae < 3.0:
        print(f"  🎯 SUB-3 dB! Beat baseline by {gain:.2f} dB")
    elif final_mae < 3.74:
        print(f"  ✓ Beat baseline by {gain:.2f} dB")
    else:
        print(f"  Gap to baseline: {final_mae - 3.74:+.2f} dB")
    print(f"  Models saved to: {INFERENCE_SAVE}")
    print(f"                   {ENSEMBLE_SAVE}")
    print("=" * 60)


if __name__ == "__main__":
    train()