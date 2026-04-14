"""
Training v10.2 — Per-point attention + heavy regularization

v10.1 results:
  Train: MAE=3.57, Slope=0.78 — architecture WORKS
  Val:   MAE=4.02, Slope=0.38 — but OVERFITS (gap=-0.60)
  
  The per-point attention learns patient-specific patterns beautifully
  on train, but 52 attention patterns × 196 patches = too many degrees
  of freedom for 211 training eyes. Attention weights memorize individual
  patients instead of learning generalizable anatomy.

Fixes (architecture unchanged, only regularization):
  1. ATTENTION DROPOUT 0.2 → 0.4 — main overfitting site
  2. POINT HEAD DROPOUT 0.25 → 0.4
  3. WEIGHT DECAY 5e-3 → 1.5e-2 — much stronger
  4. ATTENTION TEMPERATURE — learnable, initialized at 2.0 (softer attention)
     Prevents attention from collapsing to 1-2 patches per VF point
  5. LR 8e-4 → 5e-4 — slower learning = less memorization  
  6. ATTENTION ENTROPY BONUS — small penalty if attention becomes too peaked
     Encourages each VF point to use a broader patch neighborhood
  7. LABEL NOISE — tiny gaussian noise (σ=0.3 dB) on training targets
     Standard regularization for regression with small datasets

Target: close the gap from -0.60 to ~-0.15, yielding val MAE ~3.7-3.9

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
import math

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

# ── Dropout (INCREASED from v10.1) ────────────────────────────
ATTN_DROPOUT     = 0.4     # was 0.2 — attention is where overfitting happens
HEAD_DROPOUT     = 0.4     # was 0.25
REFINE_DROPOUT   = 0.15    # unchanged — refinement is tiny

# ── Learning rates (DECREASED) ────────────────────────────────
ATTN_LR = 5e-4             # was 8e-4
ATTN_WD = 1.5e-2           # was 5e-3 — 3x stronger

REFINE_LR = 3e-4           # was 5e-4
REFINE_WD = 1e-2            # was 5e-3

# Projection warm start
PROJ_INIT_BIAS = 18.0

# Augmentation
ROTATION_DEG = 5

# TTA
USE_TTA       = True
TTA_ROTATIONS = [-5, 0, 5]

# Loss
HUBER_DELTA    = 1.0
WEIGHT_SCALE   = 2.0
MAX_DB         = 35.0

# Per-eye CCC
PER_EYE_CCC_WEIGHT = 0.15
PER_EYE_CCC_START  = 5

# Variance penalty
VARIANCE_WEIGHT = 0.05
VARIANCE_START  = 8

# ── NEW: Attention regularization ─────────────────────────────
ATTN_ENTROPY_WEIGHT = 0.01   # small bonus for spread-out attention
ATTN_TEMP_INIT      = 2.0    # initial temperature (>1 = softer attention)

# ── NEW: Label noise ──────────────────────────────────────────
LABEL_NOISE_STD = 0.3        # σ=0.3 dB gaussian noise on train targets

# Validation
VAL_EVERY = 2

# Clipping
OUTLIER_CLIP_RANGE = (0, 35)

# ── Paths ──────────────────────────────────────────────────────
CURRENT_DIR        = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR       = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH    = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
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

# ============== Anatomical priors ==============
def build_vf_to_patch_prior():
    vf_to_patch = {}
    valid_count = 0
    for r in range(8):
        for c in range(9):
            if mask_OD[r, c]:
                patch_r = 13 - int(r * 13 / 7)
                patch_c = 13 - int(c * 13 / 8)
                vf_to_patch[valid_count] = (patch_r, patch_c)
                valid_count += 1
    return vf_to_patch

VF_TO_PATCH_PRIOR = build_vf_to_patch_prior()

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
            hvf_tensor = torch.tensor(hvf)
            # Label noise — small gaussian perturbation on train targets
            if LABEL_NOISE_STD > 0:
                noise = torch.randn_like(hvf_tensor) * LABEL_NOISE_STD
                # Only add noise to valid (non-masked) values
                valid_mask = hvf_tensor < MASKED_VALUE_THRESHOLD
                hvf_tensor = hvf_tensor + noise * valid_mask.float()
                # Clamp to valid range
                hvf_tensor = torch.clamp(hvf_tensor, 0.0, 35.0) * valid_mask.float() + \
                             hvf_tensor * (~valid_mask).float()
            return self.transform(img), hvf_tensor, sample['laterality']
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


# ============== Per-Point Cross-Attention ==============
class PerPointAttention(nn.Module):
    """Same as v10.1 but with:
    - Learnable temperature (softer attention)
    - Higher dropout
    - Returns attention weights for entropy regularization
    """
    def __init__(self, embed_dim=1024, num_points=52, attn_dim=128, dropout=0.4):
        super().__init__()
        self.num_points = num_points
        self.attn_dim = attn_dim
        
        self.queries = nn.Parameter(torch.randn(num_points, attn_dim) * 0.02)
        self.key_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attn_dim)
        
        # Learnable temperature — initialized > 1 for softer attention
        self.temperature = nn.Parameter(torch.tensor(ATTN_TEMP_INIT))
        
        self.patch_pos = nn.Parameter(torch.randn(196, attn_dim) * 0.02)
        
        self._init_anatomical_prior()
    
    def _init_anatomical_prior(self):
        with torch.no_grad():
            prior_bias = torch.zeros(self.num_points, 196)
            for vf_idx, (pr, pc) in VF_TO_PATCH_PRIOR.items():
                if vf_idx >= self.num_points:
                    continue
                for patch_idx in range(196):
                    patch_r = patch_idx // 14
                    patch_c = patch_idx % 14
                    dist_sq = (patch_r - pr) ** 2 + (patch_c - pc) ** 2
                    prior_bias[vf_idx, patch_idx] = -dist_sq / (2 * 3.0 ** 2)
            self.register_buffer('attn_prior', prior_bias)
    
    def forward(self, patches, laterality='OD'):
        B = patches.shape[0]
        
        keys = self.key_proj(patches) + self.patch_pos.unsqueeze(0)
        vals = self.val_proj(patches)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        logits = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        
        # Anatomical prior
        if isinstance(laterality, str):
            is_os = not laterality.startswith('OD')
        else:
            is_os = not laterality[0].startswith('OD')
        
        if is_os:
            flipped_prior = self.attn_prior.clone()
            flipped_prior_reshaped = flipped_prior.view(self.num_points, 14, 14)
            flipped_prior_reshaped = flipped_prior_reshaped.flip(2)
            flipped_prior = flipped_prior_reshaped.view(self.num_points, 196)
            logits = logits + flipped_prior.unsqueeze(0)
        else:
            logits = logits + self.attn_prior.unsqueeze(0)
        
        # Temperature scaling — higher temp = softer attention = less overfitting
        temp = F.softplus(self.temperature) + 0.5  # ensure temp >= 0.5
        logits = logits / temp
        
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)
        
        attended = torch.bmm(attn_weights_dropped, vals)
        out = self.out_proj(attended)
        
        # Return attention weights for entropy regularization
        return out, attn_weights


# ============== Point-wise Prediction Head ==============
class PointHead(nn.Module):
    def __init__(self, input_dim=2048, hidden=256, dropout=0.4, bias_init=18.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )
        nn.init.constant_(self.net[-1].bias, bias_init)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============== Cross-Point Refinement ==============
class CrossPointRefinement(nn.Module):
    def __init__(self, num_points=52, hidden=104, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_points, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_points)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        correction = self.net(x)
        return x + torch.sigmoid(self.alpha) * correction


# ============== Full Model ==============
class PerPointVFModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = 1024
        
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("✓ Encoder: FROZEN")
        
        self.attention = PerPointAttention(
            embed_dim=self.embed_dim,
            num_points=NUM_VALID_POINTS,
            attn_dim=128,
            dropout=ATTN_DROPOUT
        )
        print(f"✓ Per-point attention (dropout={ATTN_DROPOUT}, temp_init={ATTN_TEMP_INIT})")
        
        self.point_head = PointHead(
            input_dim=self.embed_dim * 2,
            hidden=256,
            dropout=HEAD_DROPOUT,
            bias_init=PROJ_INIT_BIAS
        )
        print(f"✓ Shared point head (dropout={HEAD_DROPOUT})")
        
        self.refinement = CrossPointRefinement(NUM_VALID_POINTS, hidden=104, dropout=REFINE_DROPOUT)
        print("✓ Cross-point refinement")
        
        attn_p = sum(p.numel() for p in self.attention.parameters())
        head_p = sum(p.numel() for p in self.point_head.parameters())
        ref_p  = sum(p.numel() for p in self.refinement.parameters())
        total  = attn_p + head_p + ref_p
        print(f"  Trainable: attention={attn_p:,} + head={head_p:,} + refinement={ref_p:,} = {total:,}")
    
    def forward(self, x, laterality='OD', average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")
        
        with torch.no_grad():
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        cls_token = latent[:, 0, :]
        patches   = latent[:, 1:, :]
        
        point_feats, attn_weights = self.attention(patches, laterality)
        
        B = x.shape[0]
        cls_expanded = cls_token.unsqueeze(1).expand(B, NUM_VALID_POINTS, self.embed_dim)
        combined = torch.cat([point_feats, cls_expanded], dim=2)
        
        pred = self.point_head(combined)
        pred = self.refinement(pred)
        
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        
        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        
        # Store attention weights for regularization (only during training)
        self._last_attn_weights = attn_weights
        
        return pred


# ============== Per-Eye CCC ==============
def per_eye_ccc(pred_52, target_52, mask):
    p = pred_52[mask]
    t = target_52[mask]
    if p.numel() < 5:
        return torch.tensor(0.0, device=p.device)
    p_mean = p.mean()
    t_mean = t.mean()
    p_var  = p.var()
    t_var  = t.var()
    covar  = ((p - p_mean) * (t - t_mean)).mean()
    ccc = (2 * covar) / (p_var + t_var + (p_mean - t_mean) ** 2 + 1e-8)
    return 1.0 - ccc


# ============== Attention Entropy ==============
def attention_entropy_loss(attn_weights):
    """Penalize if attention is too peaked (low entropy).
    Encourages each VF point to attend to a neighborhood of patches,
    not just 1-2 specific ones (which leads to memorization).
    
    attn_weights: (B, 52, 196) — softmax probabilities
    Returns negative mean entropy (minimize this = maximize entropy)
    """
    # Entropy per query: -sum(p * log(p))
    entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1)  # (B, 52)
    # Max possible entropy = log(196) ≈ 5.28
    # We want entropy to stay above ~3.0 (attending to ~20 patches)
    max_entropy = math.log(196)
    # Normalized: 0 = max entropy, 1 = completely peaked
    normalized = 1.0 - entropy / max_entropy  # (B, 52)
    return normalized.mean()


# ============== Loss ==============
def compute_loss(pred, target, laterality, epoch=0, attn_weights=None):
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

        weights = 1.0 + WEIGHT_SCALE * (MAX_DB - t).clamp(min=0) / MAX_DB
        huber = (F.huber_loss(p, t, reduction='none', delta=HUBER_DELTA) * weights).mean()
        mae   = (p - t).abs().mean()

        total_huber += huber * mask.sum().item()
        total_mae   += mae.item() * mask.sum().item()
        n_valid     += mask.sum().item()

        if epoch >= PER_EYE_CCC_START and mask.sum() >= 5:
            eye_ccc = per_eye_ccc(pred_52, target_52, mask)
            eye_ccc_losses.append(eye_ccc)

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

    loss = total_huber / n_valid

    # Per-eye CCC
    if len(eye_ccc_losses) > 0:
        mean_eye_ccc = torch.stack(eye_ccc_losses).mean()
        loss = loss + PER_EYE_CCC_WEIGHT * mean_eye_ccc

    # Variance penalty
    if epoch >= VARIANCE_START and pred.shape[0] >= 4:
        pred_var_per_point = pred.var(dim=0).mean()
        target_52_list = []
        for i, lat in enumerate(laterality):
            valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
            target_52_list.append(target[i][valid_idx])
        if len(target_52_list) >= 4:
            target_stack = torch.stack(target_52_list)
            valid_mask = target_stack < MASKED_VALUE_THRESHOLD
            all_valid = valid_mask.all(dim=0)
            if all_valid.sum() >= 10:
                target_var_per_point = target_stack[:, all_valid].var(dim=0).mean()
                var_ratio = pred_var_per_point / (target_var_per_point + 1e-8)
                if var_ratio < 1.0:
                    variance_penalty = (1.0 - var_ratio) ** 2
                    loss = loss + VARIANCE_WEIGHT * variance_penalty

    # Attention entropy regularization
    if attn_weights is not None and ATTN_ENTROPY_WEIGHT > 0:
        entropy_loss = attention_entropy_loss(attn_weights)
        loss = loss + ATTN_ENTROPY_WEIGHT * entropy_loss

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
            _, mae, nv = compute_loss(pred, hvf, lat, epoch=0)
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
    print("Training v10.2 — Per-point attention + heavy regularization")
    print("=" * 60)

    print(f"\nv10.1 showed: train MAE=3.57 (works!), val MAE=4.02 (overfits)")
    print(f"Same architecture, more regularization to close the gap")

    print(f"\nRegularization:")
    print(f"  Attention dropout: {ATTN_DROPOUT} (was 0.2)")
    print(f"  Head dropout:      {HEAD_DROPOUT} (was 0.25)")
    print(f"  Weight decay:      {ATTN_WD} (was 0.005)")
    print(f"  Attention temp:    {ATTN_TEMP_INIT} (learnable, softens attention)")
    print(f"  Entropy penalty:   {ATTN_ENTROPY_WEIGHT} (prevents peaked attention)")
    print(f"  Label noise:       σ={LABEL_NOISE_STD} dB")
    print(f"  LR:                {ATTN_LR} (was 8e-4)")

    print(f"\nBanned: LoRA, flips/affine, MixUp")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val', use_tta=USE_TTA)

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
    model = PerPointVFModel(base_model)
    model.to(DEVICE)

    # ── Optimizer ──────────────────────────────────────────────
    attn_head_params = list(model.attention.parameters()) + \
                       list(model.point_head.parameters())
    refine_params = list(model.refinement.parameters())

    optimizer = optim.AdamW([
        {'params': attn_head_params, 'lr': ATTN_LR,   'weight_decay': ATTN_WD},
        {'params': refine_params,    'lr': REFINE_LR,  'weight_decay': REFINE_WD},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_mae    = float('inf')
    best_corr   = 0.0
    best_slope  = 0.0
    best_score  = float('inf')
    patience    = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            pred = model(imgs, laterality=lat, average_multi=False)
            
            # Get stored attention weights for entropy regularization
            attn_w = model._last_attn_weights if hasattr(model, '_last_attn_weights') else None
            
            loss, mae, nv = compute_loss(pred, hvf, lat, epoch=epoch, attn_weights=attn_w)

            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

            pbar.set_postfix({'MAE': f'{mae:.2f}'})

        scheduler.step()

        if epoch <= 5 or epoch % 10 == 0:
            ref_alpha = torch.sigmoid(model.refinement.alpha).item()
            temp = F.softplus(model.attention.temperature).item() + 0.5
            print(f"  Refinement alpha: {ref_alpha:.3f} | Attn temp: {temp:.3f}")

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

            slope_penalty = max(0, 0.3 - val_slope) * 2.0
            composite = val_mae + slope_penalty

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
                print(f"    MAE={val_mae:.2f} | Slope={val_slope:.3f} | EyeCorr={val_eye_corr:.3f} | Score={composite:.2f}")
                patience = 0
            else:
                patience += VAL_EVERY
                if patience >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete — v10.2")
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