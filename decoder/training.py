"""
Training - Build on pretraining baseline
Uses LoRA for parameter-efficient encoder adaptation (PEFT)
"""

import os, sys, json, numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import math

# ============== MPS Configuration ==============
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✓ Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print(f"⚠️  MPS not available, using CPU")

# ==============================================================
# CONFIG
# ==============================================================
BATCH_SIZE   = 32
EPOCHS       = 120
BASE_LR      = 5e-4
WEIGHT_DECAY = 5e-4
PATIENCE     = 40

MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.3

# 1. LoRA on encoder: parameter-efficient adaptation (rank-8)
#    Proven superior to full fine-tuning on small medical datasets
#    (Hu et al. 2022; MIDL 2024 PEFT study). Adds ~200k params vs
#    millions for full fine-tuning — right-sizes capacity for 211 eyes.
LORA_RANK  = 8
LORA_ALPHA = 16   # effective scale = alpha/rank = 2.0

# 2. Projection warm-start: bias final layer to mean VF (~18 dB)
#    Without this, random init outputs ~0 dB to pretrained decoder
#    for 20+ epochs before discovering the right output range.
PROJ_INIT_BIAS = 18.0

# 3. Rotation augmentation ±5° (training only, no flips — laterality-safe)
ROTATION_DEG = 5

# 4. TTA: average predictions across 3 rotations at inference
USE_TTA       = True
TTA_ROTATIONS = [-5, 0, 5]

# 5. Huber loss + low-dB upweighting (scotoma preservation)
LOW_DB_THRESHOLD = 10.0
LOW_VALUE_WEIGHT = 2.5

# 6. Label smoothing
LABEL_SMOOTH = 0.05

# Outlier clipping
OUTLIER_CLIP_RANGE = (0, 35)

# Paths
CURRENT_DIR      = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR     = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH  = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
PRETRAINED_DECODER = os.path.join(CURRENT_DIR, "pretrained_vf_decoder.pth")
BASE_DIR         = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR       = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON       = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON         = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
BEST_SAVE        = os.path.join(CURRENT_DIR, "best_multi_image_model.pth")
INFERENCE_SAVE   = os.path.join(CURRENT_DIR, "inference_model.pth")

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

# ============== LoRA ==============
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for a frozen Linear layer.
    Output = W*x + (alpha/rank) * B @ A @ x
    Only A and B are trained; W stays frozen.
    B is zero-initialised so LoRA contributes nothing at epoch 0.
    """
    def __init__(self, linear: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.linear = linear
        self.scale  = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def inject_lora(encoder, rank: int, alpha: int):
    """Inject LoRA into qkv projection of every attention block."""
    lora_params = []
    for block in encoder.blocks:
        attn = block.attn
        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
            lora_layer = LoRALinear(attn.qkv, rank, alpha)
            attn.qkv   = lora_layer
            lora_params += [lora_layer.lora_A, lora_layer.lora_B]
    n = len(lora_params) // 2
    print(f"✓ LoRA injected into {n} attention blocks (rank={rank}, alpha={alpha})")
    print(f"  LoRA trainable params: {sum(p.numel() for p in lora_params):,}")
    return lora_params

# ============== VFAutoDecoder (permanently frozen) ==============
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

# ============== Main Model ==============
class MultiImageModel(nn.Module):
    def __init__(self, encoder, pretrained_state=None, dropout=0.3):
        super().__init__()
        self.encoder = encoder

        # Freeze entire encoder; adaptation via LoRA only
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.lora_params = inject_lora(self.encoder, LORA_RANK, LORA_ALPHA)

        # Projection: 1024 → 512 → 256 → 52
        self.projection = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(256, 52)
        )
        for m in list(self.projection.modules())[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Warm-start final layer toward mean VF sensitivity
        final = self.projection[-1]
        nn.init.constant_(final.bias, PROJ_INIT_BIAS)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)

        # Refinement decoder — frozen permanently
        self.decoder = VFAutoDecoder(input_dim=52)
        if pretrained_state is not None:
            try:
                self.decoder.load_state_dict(pretrained_state, strict=True)
                print(f"✓ Pre-trained decoder loaded (frozen permanently)")
            except Exception as e:
                print(f"⚠️  Decoder load failed: {e}")
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, x, average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        pred = self.decoder(self.projection(latent))
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred

# ============== Loss ==============
def compute_loss(pred, target, laterality, smooth=0.0):
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
        valid_idx    = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_valid = target[i][valid_idx]
        mask         = target_valid < MASKED_VALUE_THRESHOLD
        if mask.sum() == 0:
            continue

        pred_clean   = pred[i][mask]
        target_clean = target_valid[mask]

        if smooth > 0:
            mean_val     = target_clean.mean()
            target_clean = (1 - smooth) * target_clean + smooth * mean_val

        weights = torch.ones_like(pred_clean)
        weights[target_clean < LOW_DB_THRESHOLD] = LOW_VALUE_WEIGHT

        loss = (F.huber_loss(pred_clean, target_clean, reduction='none', delta=1.0) * weights).mean()
        mae  = (pred_clean - target_clean).abs().mean()

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
        target_i  = target[min(i, target.shape[0]-1)][valid_idx]
        mask      = target_i < MASKED_VALUE_THRESHOLD
        if mask.sum() > 0:
            all_pred.extend(pred[min(i, pred.shape[0]-1)][mask].tolist())
            all_target.extend(target_i[mask].tolist())
    if len(all_pred) < 2:
        return 0.0
    return np.corrcoef(np.array(all_pred), np.array(all_target))[0, 1]


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
            all_preds.append(pred.cpu().numpy().flatten())
            all_targets.append(hvf.cpu().numpy().flatten())
            all_lats.append(lat)
    if n_valid == 0:
        return float('inf'), 0.0
    corr = pearson_correlation(torch.tensor(np.stack(all_preds)),
                               torch.tensor(np.stack(all_targets)), all_lats)
    return total_mae / n_valid, corr

# ============== Training ==============
def train():
    print("=" * 60)
    print("Training - Build on Pretraining Baseline")
    print("=" * 60)
    print(f"\nActive techniques (6):")
    print(f"  1. LoRA encoder adaptation  (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    print(f"  2. Projection warm-start    (bias={PROJ_INIT_BIAS} dB)")
    print(f"  3. Rotation augmentation    (±{ROTATION_DEG}°, no flips)")
    print(f"  4. TTA                      ({len(TTA_ROTATIONS)} rotations, averaged)")
    print(f"  5. Low-dB upweighting       ({LOW_VALUE_WEIGHT}x below {LOW_DB_THRESHOLD} dB)")
    print(f"  6. Label smoothing          (alpha={LABEL_SMOOTH})")
    print(f"\nBanned techniques:")
    print(f"  - Horizontal/affine augmentation (laterality-unsafe)")
    print(f"  - MixUp (destroys spatial VF structure)")
    print(f"  - Decoder fine-tuning (caused overfitting in every prior run)")
    print(f"  - Full/partial encoder block unfreezing (replaced by LoRA)")
    print(f"\nExpected MAE per epoch range:")
    print(f"  Ep  1- 5:  6-8 dB   (warm-start active; LoRA near-zero at init)")
    print(f"  Ep  5-15:  4-5 dB   (correlation climbing steadily; target >0.4 by ep15)")
    print(f"  Ep 15-30:  3.5-4.2  (approaching baseline; corr 0.5-0.65)")
    print(f"  Ep 30-60: <3.74     (target zone; corr 0.65+; gap within ±0.2 dB)")
    print(f"  Ep 60+  :  flat     (early stop likely triggers)")
    print(f"\nStop early if:")
    print(f"  - Corr < 0.35 at epoch 15        → LoRA rank too low or LR wrong")
    print(f"  - Val gap worse than -0.4 dB      → overfitting (raise WEIGHT_DECAY)")
    print(f"  - Val MAE rising 10+ epochs       → past peak, stop")

    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val', use_tta=USE_TTA)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=val_collate_fn)

    model = MultiImageModel(base_model, pretrained_decoder_state, DROPOUT_RATE)
    model.to(DEVICE)

    optimizer = optim.AdamW([
        {'params': model.lora_params,
         'lr': BASE_LR * 0.3, 'weight_decay': 0.0},          # LoRA: no weight decay (standard)
        {'params': model.projection.parameters(),
         'lr': BASE_LR,       'weight_decay': WEIGHT_DECAY},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

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
            print(f"\n[Epoch {epoch}]")
            print(f"  Train: {train_mae:.2f} dB | Corr: {train_corr:.3f}")
            print(f"  Val:   {val_mae:.2f} dB | Corr: {val_corr:.3f} | Gap: {gap:+.2f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({'model': model.state_dict(), 'mae': val_mae,
                        'corr': val_corr, 'epoch': epoch}, BEST_SAVE)
            torch.save({'model_state_dict': model.state_dict(),
                        'encoder_checkpoint': CHECKPOINT_PATH,
                        'val_mae': val_mae, 'val_corr': val_corr}, INFERENCE_SAVE)
            if epoch % 5 == 0:
                print(f"  ✓ New Best!")
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val MAE: {best_mae:.2f} dB")
    gain = 3.74 - best_mae
    if best_mae < 3.0:
        print(f" SUB-3 dB ACHIEVED! Beat baseline by {gain:.2f} dB")
    elif best_mae < 3.74:
        print(f" Beat baseline by {gain:.2f} dB")
    else:
        print(f" Gap to baseline: {best_mae - 3.74:+.2f} dB")
    print(f"Model saved to: {INFERENCE_SAVE}")
    print("=" * 60)

if __name__ == "__main__":
    train()