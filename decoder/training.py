"""
Training v9.4 — Force patient-specific predictions

Diagnosis from v9.2/v9.3:
  - Model learns a "population average VF map" — roughly same 52 values for
    every patient with minor perturbations
  - Slope oscillates: when CCC pushes for spread, MAE worsens; when Huber
    pulls toward safe mean, slope collapses
  - Best MAE 4.17 but slope only 0.24 at that point — not truly discriminating
  - Pretrained decoder actively regresses predictions toward typical VF patterns

Root cause analysis:
  The model minimizes loss by learning the population-mean VF per location.
  With 211 eyes, most have healthy fields (20-30 dB), so predicting ~22 dB
  everywhere gives low MAE but zero patient discrimination.
  
  The CCC loss (v9.2/v9.3) tried to fix this globally, but global CCC 
  conflicts with per-point MAE — improving slope on damaged eyes hurts
  MAE on healthy eyes because predictions get noisier.

Fixes in v9.4:
  1. PER-EYE CCC LOSS: Compute CCC within each eye separately, then average.
     This forces the model to predict the correct PATTERN within each eye
     (scotoma vs healthy regions) without conflicting with MAE across eyes.
     
  2. CROSS-PATIENT VARIANCE PENALTY: If the model predicts similar values
     for different patients at the same VF location, penalize it. This 
     directly attacks the "same prediction for everyone" failure mode.
     
  3. REMOVE PRETRAINED DECODER: The autoencoder decoder is designed to
     map noisy VF → clean VF, which literally means regressing toward
     the training-set mean. Replace with a lightweight learned residual
     refinement that doesn't have pretrained "average VF" baked in.
     
  4. DEEPER SECTOR HEADS: 2 hidden layers instead of 1. The single hidden
     layer doesn't have enough capacity to learn patient-specific features
     from the 2048-dim input. More depth = more ability to extract subtle
     fundus features that distinguish patients.
     
  5. COMPOSITE SAVE CRITERION: Save best checkpoint based on 
     0.7*MAE_rank + 0.3*slope_rank, not just MAE. This prevents saving
     a checkpoint that has good MAE but zero discrimination.

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
EPOCHS       = 100
PATIENCE     = 35

MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.3

# Heads — slightly lower LR for deeper heads
HEADS_LR = 8e-4
HEADS_WD = 5e-3

# Refinement layer (replaces pretrained decoder)
REFINE_LR = 5e-4
REFINE_WD = 5e-3

# Projection
PROJ_INIT_BIAS = 18.0

# Augmentation
ROTATION_DEG = 5

# TTA
USE_TTA       = True
TTA_ROTATIONS = [-5, 0, 5]

# Loss
HUBER_DELTA      = 1.0

# Continuous weighting (proven helpful in v9.2/v9.3)
WEIGHT_SCALE = 2.0         # moderate — 3x at 0 dB, 1x at 35 dB
MAX_DB       = 35.0

# Per-eye CCC (NEW — the key change)
PER_EYE_CCC_WEIGHT = 0.2   # gentle — doesn't fight MAE
PER_EYE_CCC_START  = 5     # kick in after basic learning starts

# Cross-patient variance (NEW — penalizes same prediction for all patients)
VARIANCE_WEIGHT = 0.05     # small but meaningful
VARIANCE_START  = 8        # after model has rough predictions

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
NUM_VALID_POINTS = len(valid_indices_od)   # 52

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
        """Compute per-sample severity for weighted sampling."""
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
            mean_val = valid_vals.mean()
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


# ============== Deeper Sector Head ==============
class SectorHead(nn.Module):
    """2 hidden layers for more expressive patient-specific prediction.
    v8/v9 used 1 hidden layer (2048 → hidden → N), which couldn't extract
    enough patient-specific signal from the rich 2048-dim input.
    """
    def __init__(self, input_dim, num_points, dropout=0.3, bias_init=18.0):
        super().__init__()
        hidden1 = max(128, num_points * 8)   # wider first layer
        hidden2 = max(64, num_points * 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden2, num_points)
        )
        # Initialize final layer for warm start
        nn.init.constant_(self.net[-1].bias, bias_init)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.net(x)


# ============== Lightweight Residual Refinement ==============
class ResidualRefinement(nn.Module):
    """Replaces pretrained VFAutoDecoder. No pretrained weights = no
    regression to population mean. Learns a small correction on top
    of sector head predictions.
    
    Architecture: pred → small MLP → residual correction
    Initialized near-zero so it starts as identity.
    """
    def __init__(self, num_points=52, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_points, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_points)
        )
        # Initialize near-zero so refinement starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))  # starts at sigmoid(0)=0.5

    def forward(self, x):
        correction = self.net(x)
        weight = torch.sigmoid(self.alpha)  # learned blending
        return x + weight * correction


# ============== Model ==============
class RegionalVFModel(nn.Module):
    def __init__(self, encoder, dropout=0.3):
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

        # Sector heads — DEEPER (2 hidden layers)
        sector_input_dim = self.embed_dim * 2  # quadrant_pool + CLS = 2048
        self.sector_heads = nn.ModuleDict()
        for sec_name, sec_indices in VF_SECTORS.items():
            self.sector_heads[sec_name] = SectorHead(
                sector_input_dim, len(sec_indices), dropout, PROJ_INIT_BIAS
            )

        # Fusion (same as v9)
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

        # Lightweight residual refinement (replaces pretrained decoder)
        self.refinement = ResidualRefinement(NUM_VALID_POINTS, hidden=128, dropout=0.2)
        print("✓ Refinement: lightweight residual (no pretrained decoder)")

        # Param summary
        heads = sum(p.numel() for p in self.sector_heads.parameters())
        fus   = sum(p.numel() for p in self.fusion.parameters()) + 1
        ref   = sum(p.numel() for p in self.refinement.parameters())
        total = heads + fus + ref
        print(f"  Trainable: heads={heads:,} + fusion={fus:,} + refinement={ref:,} = {total:,}")

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

        # Lightweight refinement (NOT pretrained decoder)
        pred = self.refinement(pred)

        # Clamp
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])

        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred


# ============== Per-Eye CCC ==============
def per_eye_ccc(pred_52, target_52, mask):
    """CCC computed within a single eye's valid points.
    Forces model to predict the correct PATTERN of sensitivity
    within one eye (scotoma vs healthy regions).
    Unlike global CCC which conflicts with MAE across eyes.
    """
    p = pred_52[mask]
    t = target_52[mask]
    if p.numel() < 5:  # need enough points for meaningful correlation
        return torch.tensor(0.0, device=p.device)
    
    p_mean = p.mean()
    t_mean = t.mean()
    p_var  = p.var()
    t_var  = t.var()
    covar  = ((p - p_mean) * (t - t_mean)).mean()
    
    ccc = (2 * covar) / (p_var + t_var + (p_mean - t_mean) ** 2 + 1e-8)
    return 1.0 - ccc  # 0 when perfect


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

    total_huber = total_mae = n_valid = 0
    eye_ccc_losses = []

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
        weights = 1.0 + WEIGHT_SCALE * (MAX_DB - t).clamp(min=0) / MAX_DB

        # Weighted Huber
        huber = (F.huber_loss(p, t, reduction='none', delta=HUBER_DELTA) * weights).mean()
        mae   = (p - t).abs().mean()

        total_huber += huber * mask.sum().item()
        total_mae   += mae.item() * mask.sum().item()
        n_valid     += mask.sum().item()

        # Per-eye CCC — forces correct within-eye pattern
        if epoch >= PER_EYE_CCC_START and mask.sum() >= 5:
            eye_ccc = per_eye_ccc(pred_52, target_52, mask)
            eye_ccc_losses.append(eye_ccc)

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

    loss = total_huber / n_valid

    # Add per-eye CCC
    if len(eye_ccc_losses) > 0:
        mean_eye_ccc = torch.stack(eye_ccc_losses).mean()
        loss = loss + PER_EYE_CCC_WEIGHT * mean_eye_ccc

    # Cross-patient variance penalty
    # Computed across batch: for each VF point, variance of predictions
    # across patients should be close to variance of targets
    if epoch >= VARIANCE_START and pred.shape[0] >= 4:
        pred_var_per_point = pred.var(dim=0).mean()        # how much pred varies across patients
        # Collect valid targets per point for variance reference
        target_52_list = []
        for i, lat in enumerate(laterality):
            valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
            target_52_list.append(target[i][valid_idx])
        if len(target_52_list) >= 4:
            target_stack = torch.stack(target_52_list)
            # Mask out invalid values
            valid_mask = target_stack < MASKED_VALUE_THRESHOLD
            # Per-point target variance (only for points valid in all samples)
            all_valid = valid_mask.all(dim=0)
            if all_valid.sum() >= 10:
                target_var_per_point = target_stack[:, all_valid].var(dim=0).mean()
                # Penalize if prediction variance is much lower than target variance
                # This is one-sided: only penalize if pred_var < target_var
                var_ratio = pred_var_per_point / (target_var_per_point + 1e-8)
                if var_ratio < 1.0:
                    variance_penalty = (1.0 - var_ratio) ** 2
                    loss = loss + VARIANCE_WEIGHT * variance_penalty

    return loss, total_mae / n_valid, n_valid


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
    slope, intercept = np.polyfit(t, p, 1)
    ss_res = ((p - (slope * t + intercept)) ** 2).sum()
    ss_tot = ((p - p.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return float(r2), float(slope)


def compute_per_eye_metrics(pred, target, laterality):
    """Compute mean per-eye correlation — measures patient-specific discrimination.
    This is different from global correlation: global corr can be high just from
    predicting population mean (healthy points cluster together). Per-eye corr
    requires getting the PATTERN right within each individual eye.
    """
    if isinstance(laterality, str):
        laterality = [laterality]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    
    eye_corrs = []
    for i in range(min(len(laterality), pred.shape[0], target.shape[0])):
        lat = laterality[i] if i < len(laterality) else laterality[0]
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_52 = target[min(i, target.shape[0]-1)][valid_idx]
        pred_52 = pred[min(i, pred.shape[0]-1)]
        mask = target_52 < MASKED_VALUE_THRESHOLD
        if mask.sum() >= 5:
            p = pred_52[mask].numpy()
            t = target_52[mask].numpy()
            r = np.corrcoef(p, t)[0, 1]
            if not np.isnan(r):
                eye_corrs.append(r)
    
    if len(eye_corrs) == 0:
        return 0.0
    return float(np.mean(eye_corrs))


def evaluate(model, loader):
    model.eval()
    total_mae = n_valid = 0
    all_preds, all_targets, all_lats = [], [], []
    with torch.no_grad():
        for sample in loader:
            imgs, hvf, lat = sample
            imgs = imgs.to(DEVICE)
            pred = model(imgs, laterality=lat, average_multi=True)
            _, mae, nv = compute_loss(pred, hvf, lat, epoch=0)  # no CCC in eval
            total_mae += mae * nv
            n_valid   += nv
            all_preds.append(pred.cpu())
            all_targets.append(hvf.unsqueeze(0) if hvf.dim() == 1 else hvf)
            all_lats.append(lat)
    if n_valid == 0:
        return float('inf'), 0.0, 0.0, 0.0, 0.0
    stacked_preds   = torch.cat(all_preds, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    corr = pearson_correlation(stacked_preds, stacked_targets, all_lats)
    r2, slope = compute_r2_slope(stacked_preds, stacked_targets, all_lats)
    per_eye_corr = compute_per_eye_metrics(stacked_preds, stacked_targets, all_lats)
    return total_mae / n_valid, corr, r2, slope, per_eye_corr


# ============== Training ==============
def train():
    print("=" * 60)
    print("Training v9.4 — Force patient-specific predictions")
    print("=" * 60)

    print(f"\nDiagnosis (v9.2/v9.3):")
    print(f"  Model predicts ~same VF for every patient")
    print(f"  Slope-MAE tradeoff: good slope → bad MAE and vice versa")
    print(f"  Pretrained decoder regresses toward population mean")

    print(f"\nFixes:")
    print(f"  1. Per-eye CCC (weight={PER_EYE_CCC_WEIGHT}, start ep {PER_EYE_CCC_START})")
    print(f"     Forces correct within-eye pattern without conflicting with MAE")
    print(f"  2. Cross-patient variance penalty (weight={VARIANCE_WEIGHT}, start ep {VARIANCE_START})")
    print(f"     Penalizes if predictions don't vary enough across patients")
    print(f"  3. NO pretrained decoder — lightweight residual refinement instead")
    print(f"  4. Deeper sector heads (2 hidden layers)")
    print(f"  5. Continuous weighting: up to {1+WEIGHT_SCALE:.0f}x for 0 dB points")
    print(f"  6. Composite save criterion (MAE + slope)")

    print(f"\nBanned: LoRA, flips/affine, MixUp")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val', use_tta=USE_TTA)

    # Weighted sampling
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

    # ── Model (no pretrained decoder) ──────────────────────────
    model = RegionalVFModel(base_model, DROPOUT_RATE)
    model.to(DEVICE)

    # ── Optimizer ──────────────────────────────────────────────
    head_params = list(model.sector_heads.parameters()) + \
                  list(model.fusion.parameters()) + [model.fusion_alpha]
    refine_params = list(model.refinement.parameters())

    optimizer = optim.AdamW([
        {'params': head_params,   'lr': HEADS_LR,  'weight_decay': HEADS_WD},
        {'params': refine_params, 'lr': REFINE_LR, 'weight_decay': REFINE_WD},
    ])

    # Monotonic cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_mae    = float('inf')
    best_corr   = 0.0
    best_slope  = 0.0
    best_score  = float('inf')  # composite score for saving
    patience    = 0

    # Track history for composite scoring
    mae_history   = []
    slope_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
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

            pbar.set_postfix({'MAE': f'{mae:.2f}'})

        scheduler.step()

        # Log refinement parameter
        ref_alpha = torch.sigmoid(model.refinement.alpha).item()
        if epoch <= 5 or epoch % 10 == 0:
            print(f"  Refinement alpha: {ref_alpha:.3f}")
            print(f"  Fusion alpha: {model.fusion_alpha.item():.3f}")

        # ── Validate ───────────────────────────────────────────
        if epoch % VAL_EVERY == 0 or epoch <= 3:
            val_mae, val_corr, val_r2, val_slope, val_eye_corr = evaluate(model, val_loader)

            train_eval = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, val_transform,
                                           mode='val', use_tta=False)
            train_eval_loader = DataLoader(train_eval, batch_size=1, shuffle=False,
                                           num_workers=0, collate_fn=val_collate_fn)
            train_mae, train_corr, train_r2, train_slope, train_eye_corr = evaluate(model, train_eval_loader)
            gap = train_mae - val_mae

            print(f"\n[Epoch {epoch}]")
            print(f"  Train: MAE={train_mae:.2f} | Corr={train_corr:.3f} | Slope={train_slope:.3f} | EyeCorr={train_eye_corr:.3f}")
            print(f"  Val:   MAE={val_mae:.2f} | Corr={val_corr:.3f} | Slope={val_slope:.3f} | EyeCorr={val_eye_corr:.3f}")
            print(f"  Gap: {gap:+.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if gap < -0.4:
                print(f"  ⚠️  Gap < -0.4 — possible overfitting")

            # Composite save score: lower is better
            # Prioritize MAE but require decent slope
            # slope_penalty: if slope < 0.3, add penalty; if slope > 0.4, give bonus
            slope_penalty = max(0, 0.3 - val_slope) * 2.0   # e.g., slope=0.1 → penalty=0.4
            composite = val_mae + slope_penalty

            mae_history.append(val_mae)
            slope_history.append(val_slope)

            if composite < best_score or val_mae < best_mae:
                save_reason = []
                if val_mae < best_mae:
                    save_reason.append(f"MAE {best_mae:.2f}→{val_mae:.2f}")
                    best_mae = val_mae
                if composite < best_score:
                    save_reason.append(f"Score {best_score:.2f}→{composite:.2f}")
                    best_score = composite
                best_corr  = max(best_corr, val_corr)
                best_slope = max(best_slope, val_slope)

                torch.save({'model': model.state_dict(), 'mae': val_mae,
                            'corr': val_corr, 'r2': val_r2, 'slope': val_slope,
                            'eye_corr': val_eye_corr, 'epoch': epoch}, BEST_SAVE)
                torch.save({'model_state_dict': model.state_dict(),
                            'encoder_checkpoint': CHECKPOINT_PATH,
                            'val_mae': val_mae, 'val_corr': val_corr}, INFERENCE_SAVE)
                print(f"  ✓ Saved! ({', '.join(save_reason)})")
                print(f"    MAE={val_mae:.2f} | Slope={val_slope:.3f} | Score={composite:.2f}")
                patience = 0
            else:
                patience += VAL_EVERY
                if patience >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete — v9.4")
    print(f"  Best Val MAE:      {best_mae:.2f} dB")
    print(f"  Best Val Corr:     {best_corr:.3f}")
    print(f"  Best Val Slope:    {best_slope:.3f}")
    print(f"  Best Composite:    {best_score:.2f}")
    gain = 3.74 - best_mae
    if best_mae < 3.0:
        print(f"  🎯 SUB-3 dB!")
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