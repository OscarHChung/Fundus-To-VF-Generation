"""
Training v9.2 — Fix regression-to-mean

Diagnosis from v9 scatterplot (4.14 MAE, R²=0.314, slope=0.33):
  - Model collapses predictions toward population mean (~18-20 dB)
  - Low-sensitivity points (0-10 dB) overpredicted by ~10-15 dB
  - High-sensitivity points (25-30 dB) underpredicted by ~3-5 dB
  - Slope 0.33 means only 1/3 of true dynamic range captured

Root causes:
  1. Huber loss with weak 1.5x upweighting doesn't fight class imbalance
     (most VF points are 20-30 dB healthy, so predicting ~20 is "safe")
  2. Frozen pretrained decoder was trained as denoiser → pushes toward
     typical VF patterns → amplifies regression to mean
  3. No explicit incentive for the model to preserve dynamic range

Fixes in v9.2:
  1. CONTINUOUS INVERSE-SENSITIVITY WEIGHTING: weight = 1 + scale * (max_dB - target) / max_dB
     Low dB points get ~4-5x weight, not just 1.5x. This is the #1 fix.
  2. CONCORDANCE CORRELATION (CCC) LOSS COMPONENT: directly penalizes
     slope deviation from 1.0 and mean shift. CCC = 1 when pred=target exactly.
  3. QUANTILE-BALANCED SAMPLING: oversample eyes with low-sensitivity points
     so each batch has better representation of damaged fields
  4. LIGHTER DECODER: reduce decoder influence with lower residual weight init
     and add a learnable bypass that lets raw sector-head predictions through
  5. MONOTONIC COSINE SCHEDULE: CosineAnnealingLR (no restarts) per v9.1 plan
  6. POINTWISE CALIBRATION HEAD: tiny per-point scale+shift after decoder
     to correct systematic per-location biases

Banned: LoRA, flips/affine, MixUp
"""

import os, sys, json, numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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
EPOCHS       = 120
PATIENCE     = 40          # longer patience — monotonic schedule is slow

MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.3

# Heads
HEADS_LR = 1e-3
HEADS_WD = 5e-3

# Decoder: trainable but heavily regularized
DECODER_LR = 1e-4
DECODER_WD = 1e-2

# Calibration head (tiny)
CALIB_LR = 5e-4
CALIB_WD = 1e-3

# Projection
PROJ_INIT_BIAS = 18.0

# Augmentation
ROTATION_DEG = 5

# TTA
USE_TTA       = True
TTA_ROTATIONS = [-5, 0, 5]

# Loss — regression-to-mean fixes
HUBER_DELTA          = 1.0
# Continuous weighting: weight = 1 + WEIGHT_SCALE * (MAX_DB - target) / MAX_DB
# At 0 dB: weight = 1 + 4.0 = 5.0x
# At 10 dB: weight = 1 + 4.0 * 25/35 ≈ 3.86x
# At 20 dB: weight = 1 + 4.0 * 15/35 ≈ 2.71x
# At 30 dB: weight = 1 + 4.0 * 5/35 ≈ 1.57x
WEIGHT_SCALE = 2.5
MAX_DB       = 35.0

# CCC loss weight (concordance correlation coefficient)
CCC_LOSS_MAX    = 0.3      # max CCC weight after warmup
CCC_START_EPOCH = 10       # pure Huber before this
CCC_RAMP_EPOCHS = 20       # linearly ramp 0 → CCC_LOSS_MAX over this many epochs

# Validation
VAL_EVERY = 2

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
NUM_VALID_POINTS = len(valid_indices_od)

# ============== VF Sectors & Patch Quadrants ==============
def build_sector_indices():
    grid = mask_OD.copy()
    rows, cols = grid.shape
    sectors = {
        'sup_nasal': [], 'sup_temporal': [],
        'inf_nasal': [], 'inf_temporal': [],
    }
    valid_count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c]:
                is_superior = r < 4
                is_nasal = c <= 4
                if is_superior and is_nasal:
                    sectors['sup_nasal'].append(valid_count)
                elif is_superior and not is_nasal:
                    sectors['sup_temporal'].append(valid_count)
                elif not is_superior and is_nasal:
                    sectors['inf_nasal'].append(valid_count)
                else:
                    sectors['inf_temporal'].append(valid_count)
                valid_count += 1
    return sectors

VF_SECTORS = build_sector_indices()

def build_patch_quadrants():
    quadrants = {
        'sup_retina_nasal': [], 'sup_retina_temporal': [],
        'inf_retina_nasal': [], 'inf_retina_temporal': [],
    }
    for r in range(14):
        for c in range(14):
            patch_idx = r * 14 + c
            is_sup = r < 7
            is_right = c >= 7
            if is_sup and is_right:
                quadrants['sup_retina_nasal'].append(patch_idx)
            elif is_sup and not is_right:
                quadrants['sup_retina_temporal'].append(patch_idx)
            elif not is_sup and is_right:
                quadrants['inf_retina_nasal'].append(patch_idx)
            else:
                quadrants['inf_retina_temporal'].append(patch_idx)
    return quadrants

PATCH_QUADRANTS = build_patch_quadrants()

RETINA_TO_VF_OD = {
    'sup_retina_nasal': 'inf_temporal', 'sup_retina_temporal': 'inf_nasal',
    'inf_retina_nasal': 'sup_temporal', 'inf_retina_temporal': 'sup_nasal',
}
RETINA_TO_VF_OS = {
    'sup_retina_nasal': 'inf_nasal', 'sup_retina_temporal': 'inf_temporal',
    'inf_retina_nasal': 'sup_nasal', 'inf_retina_temporal': 'sup_temporal',
}

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

    def get_sample_severity(self):
        """Compute per-sample severity for weighted sampling.
        Returns weight for each training sample — eyes with more low-dB points get higher weight.
        """
        weights = []
        for sample in self.samples:
            hvf = np.array(sample['hvf'], dtype=np.float32).flatten()
            lat = sample['laterality']
            valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
            valid_vals = hvf[valid_idx]
            valid_vals = valid_vals[valid_vals < MASKED_VALUE_THRESHOLD]
            if len(valid_vals) == 0:
                weights.append(1.0)
                continue
            # Mean deviation from healthy (30 dB)
            mean_val = valid_vals.mean()
            # More weight for eyes with lower mean sensitivity
            # Healthy eye (~27 dB) → weight ~1.3
            # Moderate damage (~15 dB) → weight ~3.0
            # Severe damage (~5 dB) → weight ~4.7
            w = 1.0 + WEIGHT_SCALE * (MAX_DB - mean_val) / MAX_DB
            weights.append(max(w, 1.0))
        return weights

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

# ============== Sector Head ==============
class SectorHead(nn.Module):
    def __init__(self, input_dim, num_points, dropout=0.3, bias_init=18.0):
        super().__init__()
        hidden = max(64, num_points * 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_points)
        )
        nn.init.constant_(self.net[-1].bias, bias_init)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.net(x)

# ============== Pointwise Calibration ==============
class PointwiseCalibration(nn.Module):
    """Learnable per-point scale and shift to correct systematic biases.
    Initialized to identity (scale=1, shift=0).
    """
    def __init__(self, num_points=52):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_points))
        self.shift = nn.Parameter(torch.zeros(num_points))

    def forward(self, x):
        return x * self.scale + self.shift

# ============== Model ==============
class RegionalVFModel(nn.Module):
    def __init__(self, encoder, pretrained_decoder_state=None, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = 1024

        # Freeze entire encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("✓ Encoder: FROZEN")

        for quad_name, indices in PATCH_QUADRANTS.items():
            self.register_buffer(f'patch_idx_{quad_name}',
                                 torch.tensor(indices, dtype=torch.long))
        for sec_name, indices in VF_SECTORS.items():
            self.register_buffer(f'vf_idx_{sec_name}',
                                 torch.tensor(indices, dtype=torch.long))

        # Sector heads — wider hidden layers for more expressive per-sector prediction
        sector_input_dim = self.embed_dim * 2  # quadrant_pool + CLS
        self.sector_heads = nn.ModuleDict()
        for sec_name, sec_indices in VF_SECTORS.items():
            self.sector_heads[sec_name] = SectorHead(
                sector_input_dim, len(sec_indices), dropout, PROJ_INIT_BIAS
            )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(NUM_VALID_POINTS, NUM_VALID_POINTS * 2),
            nn.LayerNorm(NUM_VALID_POINTS * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(NUM_VALID_POINTS * 2, NUM_VALID_POINTS),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.3))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        # Decoder: TRAINABLE with bypass
        # bypass_alpha controls how much raw prediction passes through vs decoder
        # Start at 0.3 so decoder doesn't dominate early (decoder tends to regress to mean)
        self.decoder = VFAutoDecoder(input_dim=NUM_VALID_POINTS)
        self.decoder_bypass = nn.Parameter(torch.tensor(0.3))  # how much raw pred to keep

        if pretrained_decoder_state is not None:
            try:
                self.decoder.load_state_dict(pretrained_decoder_state, strict=True)
                print(f"✓ Decoder: TRAINABLE with bypass (init={self.decoder_bypass.item():.1f})")
            except Exception as e:
                print(f"⚠️  Decoder load failed: {e}")
        else:
            print("✓ Decoder: from scratch")

        # Pointwise calibration — corrects per-location systematic bias
        self.calibration = PointwiseCalibration(NUM_VALID_POINTS)
        print("✓ Pointwise calibration layer (scale+shift per VF point)")

        # Param summary
        heads = sum(p.numel() for p in self.sector_heads.parameters())
        fus   = sum(p.numel() for p in self.fusion.parameters()) + 1
        dec   = sum(p.numel() for p in self.decoder.parameters()) + 1  # +bypass
        cal   = sum(p.numel() for p in self.calibration.parameters())
        total = heads + fus + dec + cal
        print(f"  Trainable: heads={heads:,} + fusion={fus:,} + decoder={dec:,} + calib={cal:,} = {total:,}")

    def pool_quadrant(self, patches, quad_name):
        idx = getattr(self, f'patch_idx_{quad_name}')
        return patches[:, idx, :].mean(dim=1)

    def forward(self, x, laterality='OD', average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")

        with torch.no_grad():
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        cls_token = latent[:, 0, :]
        patches = latent[:, 1:, :]

        if isinstance(laterality, str):
            mapping = RETINA_TO_VF_OD if laterality.startswith('OD') else RETINA_TO_VF_OS
        else:
            mapping = RETINA_TO_VF_OD if laterality[0].startswith('OD') else RETINA_TO_VF_OS

        sector_preds = {}
        for retina_quad, vf_sector in mapping.items():
            quad_feat = self.pool_quadrant(patches, retina_quad)
            head_input = torch.cat([quad_feat, cls_token], dim=1)
            sector_preds[vf_sector] = self.sector_heads[vf_sector](head_input)

        pred = torch.zeros(x.shape[0], NUM_VALID_POINTS, device=x.device)
        for sec_name, sec_pred in sector_preds.items():
            idx = getattr(self, f'vf_idx_{sec_name}')
            pred[:, idx] = sec_pred

        # Fusion
        fused = self.fusion(pred)
        pred = pred + self.fusion_alpha * fused

        # Decoder with bypass — don't let decoder fully dominate
        decoded = self.decoder(pred)
        bypass = torch.sigmoid(self.decoder_bypass)  # keep in [0,1]
        pred = bypass * pred + (1 - bypass) * decoded

        # Pointwise calibration
        pred = self.calibration(pred)

        # Clamp
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])

        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred

# ============== Concordance Correlation Coefficient ==============
def ccc_loss(pred, target):
    """1 - CCC. CCC penalizes both correlation < 1 and mean/variance mismatch.
    Perfect when pred == target. Directly addresses slope != 1 problem.
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var()
    target_var = target.var()
    covar = ((pred - pred_mean) * (target - target_mean)).mean()

    ccc = (2 * covar) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
    return 1.0 - ccc

# ============== Loss ==============
def compute_loss(pred, target, laterality, epoch=0):
    device = pred.device
    target = target.to(device)
    if isinstance(laterality, str):
        laterality = [laterality]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    total_loss = total_mae = n_valid = 0
    all_p, all_t = [], []

    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_52 = target[i][valid_idx]
        pred_52   = pred[i]
        mask      = target_52 < MASKED_VALUE_THRESHOLD
        if mask.sum() == 0:
            continue
        p = pred_52[mask]
        t = target_52[mask]

        # Continuous inverse-sensitivity weighting
        # Higher weight for lower-sensitivity (more damaged) points
        weights = 1.0 + WEIGHT_SCALE * (MAX_DB - t).clamp(min=0) / MAX_DB

        huber = (F.huber_loss(p, t, reduction='none', delta=HUBER_DELTA) * weights).mean()
        mae   = (p - t).abs().mean()

        total_loss += huber * mask.sum().item()
        total_mae  += mae.item() * mask.sum().item()
        n_valid    += mask.sum().item()

        all_p.append(p)
        all_t.append(t)

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

    base_loss = total_loss / n_valid

    # CCC loss — phased in gradually to avoid destabilizing early training
    ccc_weight = 0.0
    if epoch >= CCC_START_EPOCH and len(all_p) > 0 and n_valid >= 10:
        ramp_progress = min(1.0, (epoch - CCC_START_EPOCH) / CCC_RAMP_EPOCHS)
        ccc_weight = CCC_LOSS_MAX * ramp_progress
        cat_p = torch.cat(all_p)
        cat_t = torch.cat(all_t)
        ccc_l = ccc_loss(cat_p, cat_t)
        combined = (1 - ccc_weight) * base_loss + ccc_weight * ccc_l
    else:
        combined = base_loss

    return combined, total_mae / n_valid, n_valid


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


def compute_r2_slope(pred, target, laterality):
    """Compute R² and regression slope for monitoring."""
    all_pred, all_target = [], []
    if isinstance(laterality, str):
        laterality = [laterality]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    for i in range(min(len(laterality), pred.shape[0], target.shape[0])):
        lat = laterality[i] if i < len(laterality) else laterality[0]
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_52 = target[min(i, target.shape[0]-1)][valid_idx]
        pred_52 = pred[min(i, pred.shape[0]-1)]
        mask = target_52 < MASKED_VALUE_THRESHOLD
        if mask.sum() > 0:
            all_pred.extend(pred_52[mask].tolist())
            all_target.extend(target_52[mask].tolist())
    if len(all_pred) < 2:
        return 0.0, 0.0
    p = np.array(all_pred)
    t = np.array(all_target)
    # Slope via least squares: pred = slope * target + intercept
    slope, intercept = np.polyfit(t, p, 1)
    ss_res = ((p - (slope * t + intercept)) ** 2).sum()
    ss_tot = ((p - p.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return float(r2), float(slope)


def evaluate(model, loader):
    model.eval()
    total_mae = n_valid = 0
    all_preds, all_targets, all_lats = [], [], []
    with torch.no_grad():
        for sample in loader:
            imgs, hvf, lat = sample
            imgs = imgs.to(DEVICE)
            pred = model(imgs, laterality=lat, average_multi=True)
            _, mae, nv = compute_loss(pred, hvf, lat)
            total_mae += mae * nv
            n_valid   += nv
            all_preds.append(pred.cpu())
            all_targets.append(hvf.unsqueeze(0) if hvf.dim() == 1 else hvf)
            all_lats.append(lat)
    if n_valid == 0:
        return float('inf'), 0.0, 0.0, 0.0
    stacked_preds   = torch.cat(all_preds, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    corr = pearson_correlation(stacked_preds, stacked_targets, all_lats)
    r2, slope = compute_r2_slope(stacked_preds, stacked_targets, all_lats)
    return total_mae / n_valid, corr, r2, slope


# ============== Training ==============
def train():
    print("=" * 60)
    print("Training v9.2 — Fix regression-to-mean")
    print("=" * 60)

    print(f"\nDiagnosis (v9 scatterplot):")
    print(f"  Slope = 0.33 (should be ~1.0)")
    print(f"  R² = 0.314")
    print(f"  Bias = +0.96 dB (overpredicts on average)")
    print(f"  Model collapses toward population mean")

    print(f"\nFixes:")
    print(f"  1. Continuous weighting: up to {1+WEIGHT_SCALE:.0f}x for 0 dB points")
    print(f"  2. CCC loss (max={CCC_LOSS_MAX}, ramp ep {CCC_START_EPOCH}-{CCC_START_EPOCH+CCC_RAMP_EPOCHS}) — penalizes slope≠1")
    print(f"  3. Quantile-balanced sampling — oversample damaged eyes")
    print(f"  4. Decoder bypass — reduce mean-regression from decoder")
    print(f"  5. Pointwise calibration — per-location scale+shift")
    print(f"  6. Monotonic cosine schedule (T_max={EPOCHS})")

    print(f"\nBanned: LoRA, flips/affine, MixUp")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val', use_tta=USE_TTA)

    # Weighted sampling — oversample damaged eyes
    sample_weights = train_dataset.get_sample_severity()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    print(f"  Sampler weight range: [{min(sample_weights):.1f}, {max(sample_weights):.1f}]")

    train_loader = DataLoader(train_dataset, BATCH_SIZE, sampler=sampler,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=val_collate_fn)

    # ── Model ──────────────────────────────────────────────────
    model = RegionalVFModel(base_model, pretrained_decoder_state, DROPOUT_RATE)
    model.to(DEVICE)

    # ── Optimizer: separate param groups ───────────────────────
    head_params = list(model.sector_heads.parameters()) + \
                  list(model.fusion.parameters()) + [model.fusion_alpha]
    decoder_params = list(model.decoder.parameters()) + [model.decoder_bypass]
    calib_params = list(model.calibration.parameters())

    optimizer = optim.AdamW([
        {'params': head_params,    'lr': HEADS_LR,   'weight_decay': HEADS_WD},
        {'params': decoder_params, 'lr': DECODER_LR, 'weight_decay': DECODER_WD},
        {'params': calib_params,   'lr': CALIB_LR,   'weight_decay': CALIB_WD},
    ])

    # Monotonic cosine decay — no restarts (v9 was killed by restart at ep 25)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_mae   = float('inf')
    best_corr  = 0.0
    best_slope = 0.0
    patience   = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = epoch_mae = epoch_n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            pred = model(imgs, laterality=lat, average_multi=False)
            loss, mae, nv = compute_loss(pred, hvf, lat, epoch=epoch)

            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                epoch_loss += loss.item() * nv
                epoch_mae  += mae * nv
                epoch_n    += nv

            pbar.set_postfix({'MAE': f'{mae:.2f}'})

        scheduler.step()

        # Log bypass parameter
        bypass_val = torch.sigmoid(model.decoder_bypass).item()
        if epoch <= 5 or epoch % 10 == 0:
            print(f"  Decoder bypass: {bypass_val:.3f} (raw={model.decoder_bypass.item():.3f})")
            print(f"  Fusion alpha: {model.fusion_alpha.item():.3f}")

        # ── Validate ───────────────────────────────────────────
        if epoch % VAL_EVERY == 0 or epoch <= 3:
            val_mae, val_corr, val_r2, val_slope = evaluate(model, val_loader)

            train_eval = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, val_transform,
                                           mode='val', use_tta=False)
            train_eval_loader = DataLoader(train_eval, batch_size=1, shuffle=False,
                                           num_workers=0, collate_fn=val_collate_fn)
            train_mae, train_corr, train_r2, train_slope = evaluate(model, train_eval_loader)
            gap = train_mae - val_mae

            print(f"\n[Epoch {epoch}]")
            print(f"  Train: MAE={train_mae:.2f} | Corr={train_corr:.3f} | R²={train_r2:.3f} | Slope={train_slope:.3f}")
            print(f"  Val:   MAE={val_mae:.2f} | Corr={val_corr:.3f} | R²={val_r2:.3f} | Slope={val_slope:.3f}")
            print(f"  Gap: {gap:+.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if gap < -0.4:
                print(f"  ⚠️  Gap < -0.4 — decoder may be overfitting")

            # Track slope improvement — this is the key metric for regression-to-mean
            if val_slope > best_slope + 0.02:
                print(f"  📈 Slope improved: {best_slope:.3f} → {val_slope:.3f}")

            if val_mae < best_mae:
                best_mae   = val_mae
                best_corr  = val_corr
                best_slope = max(best_slope, val_slope)
                torch.save({'model': model.state_dict(), 'mae': val_mae,
                            'corr': val_corr, 'r2': val_r2, 'slope': val_slope,
                            'epoch': epoch}, BEST_SAVE)
                torch.save({'model_state_dict': model.state_dict(),
                            'encoder_checkpoint': CHECKPOINT_PATH,
                            'val_mae': val_mae, 'val_corr': val_corr}, INFERENCE_SAVE)
                print(f"  ✓ New Best! (MAE={val_mae:.2f}, Corr={val_corr:.3f}, Slope={val_slope:.3f})")
                patience = 0
            else:
                patience += VAL_EVERY
                if patience >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete — v9.2")
    print(f"  Best Val MAE:   {best_mae:.2f} dB")
    print(f"  Best Val Corr:  {best_corr:.3f}")
    print(f"  Best Val Slope: {best_slope:.3f} (target: >0.6)")
    gain = 3.74 - best_mae
    if best_mae < 3.0:
        print(f"  🎯 SUB-3 dB! Beat baseline by {gain:.2f} dB")
    elif best_mae < 3.74:
        print(f"  ✓ Beat baseline by {gain:.2f} dB")
    elif best_mae < 4.0:
        print(f"  ✓ Sub-4! Gap to baseline: {best_mae - 3.74:+.2f} dB")
    else:
        print(f"  Gap to baseline: {best_mae - 3.74:+.2f} dB")
    print(f"  Model saved to: {INFERENCE_SAVE}")
    print("=" * 60)


if __name__ == "__main__":
    train()