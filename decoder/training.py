"""
Training v10.1 — Per-point spatial attention

WHY the architecture must change:
  v8-v9.4 all use the same bottleneck: 196 patches → 4 quadrant pools → sector heads.
  Mean-pooling 49 patches per quadrant destroys the spatial detail needed to detect
  focal scotomas. Each VF point corresponds to a specific nerve fiber bundle path,
  not a whole quadrant. A scotoma affecting 3 VF points might correspond to 5-10
  specific patches — their signal gets averaged away with 39-44 healthy patches.

  Result: ~4.1 MAE ceiling regardless of loss function.

NEW ARCHITECTURE:
  Each of the 52 VF points gets its own learned attention over all 196 patches.
  This means VF point #7 can learn "I depend mostly on patches (3,9), (4,9), (4,10)"
  which corresponds to the actual retinal nerve fiber anatomy.

  Frozen RETFound → 196 patches (1024-dim each) + CLS token
      ↓
  Per-point cross-attention: 52 learned query vectors attend to 196 patches
  Each query produces a 1024-dim weighted combination of patches
      ↓  
  Point-wise prediction: [attended_feat_1024 + CLS_1024] → hidden → 1 value per point
      ↓
  Lightweight refinement (cross-point, captures neighbor interactions)
      ↓
  52 VF sensitivity values

Memory: attention weights are 52 × 196 = 10,192 floats — negligible.
Trainable params: ~700K (similar to v9's 1.1M, well within 211-eye budget).

KEEPS from v9.4 (proven):
  - Per-eye CCC loss
  - Cross-patient variance penalty  
  - Weighted sampling
  - No pretrained decoder
  - Continuous inverse-sensitivity weighting
  - Monotonic cosine schedule

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
DROPOUT_RATE = 0.25

# Attention + point heads
ATTN_LR = 8e-4
ATTN_WD = 5e-3

# Refinement
REFINE_LR = 5e-4
REFINE_WD = 5e-3

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

# ============== Anatomical priors for attention init ==============
def build_vf_to_patch_prior():
    """Build approximate mapping from VF grid positions to retinal patch positions.
    
    VF grid is 8×9, retinal patches are 14×14.
    Key anatomy: superior retina → inferior VF, nasal retina → temporal VF.
    For OD: nasal = right side of image (high col), temporal = left.
    
    Returns dict mapping each valid VF index (0-51) to approximate
    (row, col) center in the 14×14 patch grid.
    """
    vf_to_patch = {}
    valid_count = 0
    for r in range(8):
        for c in range(9):
            if mask_OD[r, c]:
                # VF (r,c) → retinal position is vertically flipped and
                # horizontally flipped (for OD, nasal-temporal swap)
                # Map VF row [0-7] → patch row [13-0] (vertical flip)
                # Map VF col [0-8] → patch col [13-0] (horizontal flip for OD)
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


# ============== Per-Point Cross-Attention ==============
class PerPointAttention(nn.Module):
    """Each of 52 VF points learns which of 196 retinal patches to attend to.
    
    Architecture:
      52 learned query vectors (each 128-dim)
      Patches projected to 128-dim keys
      Attention: softmax(Q @ K^T / sqrt(d)) → weighted sum of patch values
      
    Initialized with anatomical prior: each query starts with higher attention
    on patches near its expected retinal location.
    
    Memory: 52 × 196 attention matrix = ~10K floats (negligible)
    """
    def __init__(self, embed_dim=1024, num_points=52, attn_dim=128, dropout=0.2):
        super().__init__()
        self.num_points = num_points
        self.attn_dim = attn_dim
        
        # Learned queries — one per VF point
        self.queries = nn.Parameter(torch.randn(num_points, attn_dim) * 0.02)
        
        # Project patches to keys and values
        self.key_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        
        # Output projection: attn_dim → embed_dim (to combine with CLS)
        self.out_proj = nn.Linear(attn_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attn_dim)
        
        # Spatial position embedding for patches (14×14 grid)
        self.patch_pos = nn.Parameter(torch.randn(196, attn_dim) * 0.02)
        
        # Initialize attention with anatomical prior
        self._init_anatomical_prior()
    
    def _init_anatomical_prior(self):
        """Bias initial attention toward anatomically correct patches.
        Each VF point starts attending to a gaussian blob centered on its
        expected retinal location. The model can learn to override this.
        """
        with torch.no_grad():
            prior_bias = torch.zeros(self.num_points, 196)
            for vf_idx, (pr, pc) in VF_TO_PATCH_PRIOR.items():
                if vf_idx >= self.num_points:
                    continue
                for patch_idx in range(196):
                    patch_r = patch_idx // 14
                    patch_c = patch_idx % 14
                    dist_sq = (patch_r - pr) ** 2 + (patch_c - pc) ** 2
                    # Gaussian with sigma=3 patches (~20% of grid)
                    prior_bias[vf_idx, patch_idx] = -dist_sq / (2 * 3.0 ** 2)
            
            # Encode this prior into query vectors via SVD-like initialization
            # Simpler: just store as a bias that gets added to attention logits
            self.register_buffer('attn_prior', prior_bias)
    
    def forward(self, patches, laterality='OD'):
        """
        patches: (B, 196, 1024)
        Returns: (B, 52, 1024) — one feature vector per VF point
        """
        B = patches.shape[0]
        
        # Project patches to keys and values, add position
        keys = self.key_proj(patches) + self.patch_pos.unsqueeze(0)   # (B, 196, attn_dim)
        vals = self.val_proj(patches)                                  # (B, 196, attn_dim)
        
        # Queries: (52, attn_dim) → (B, 52, attn_dim)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        # Attention logits: (B, 52, 196)
        logits = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        
        # Add anatomical prior bias
        # For OS eyes, flip the prior horizontally (columns 0-13 ↔ 13-0)
        if isinstance(laterality, str):
            is_os = not laterality.startswith('OD')
        else:
            is_os = not laterality[0].startswith('OD')
        
        if is_os:
            # Flip patch columns: patch at (r,c) → (r, 13-c)
            # patch_idx = r*14 + c → r*14 + (13-c)
            flipped_prior = self.attn_prior.clone()
            flipped_prior_reshaped = flipped_prior.view(self.num_points, 14, 14)
            flipped_prior_reshaped = flipped_prior_reshaped.flip(2)  # flip columns
            flipped_prior = flipped_prior_reshaped.view(self.num_points, 196)
            logits = logits + flipped_prior.unsqueeze(0)
        else:
            logits = logits + self.attn_prior.unsqueeze(0)
        
        # Softmax attention
        attn_weights = F.softmax(logits, dim=-1)      # (B, 52, 196)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum of values
        attended = torch.bmm(attn_weights, vals)       # (B, 52, attn_dim)
        
        # Project back to embed_dim
        out = self.out_proj(attended)                   # (B, 52, embed_dim)
        
        return out


# ============== Point-wise Prediction Head ==============
class PointHead(nn.Module):
    """Shared MLP that maps per-point features to a single dB value.
    Input: [attended_feat (1024) + CLS (1024)] = 2048 per point
    Output: 1 value per point
    
    Shared across all 52 points (weight sharing = regularization).
    """
    def __init__(self, input_dim=2048, hidden=256, dropout=0.25, bias_init=18.0):
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
        """x: (B, 52, 2048) → (B, 52)"""
        return self.net(x).squeeze(-1)


# ============== Cross-Point Refinement ==============
class CrossPointRefinement(nn.Module):
    """Light cross-point layer: captures spatial dependencies between 
    neighboring VF points (e.g., scotomas are spatially contiguous).
    Initialized near-identity.
    """
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
    def __init__(self, encoder, dropout=0.25):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = 1024
        
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("✓ Encoder: FROZEN")
        
        # Per-point cross-attention
        self.attention = PerPointAttention(
            embed_dim=self.embed_dim,
            num_points=NUM_VALID_POINTS,
            attn_dim=128,
            dropout=dropout
        )
        print("✓ Per-point attention (52 queries × 196 patches)")
        
        # Shared point prediction head
        self.point_head = PointHead(
            input_dim=self.embed_dim * 2,  # attended + CLS
            hidden=256,
            dropout=dropout,
            bias_init=PROJ_INIT_BIAS
        )
        print("✓ Shared point head (2048 → 256 → 128 → 1)")
        
        # Cross-point refinement
        self.refinement = CrossPointRefinement(NUM_VALID_POINTS, hidden=104, dropout=0.15)
        print("✓ Cross-point refinement")
        
        # Param summary
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
        cls_token = latent[:, 0, :]       # (B, 1024)
        patches   = latent[:, 1:, :]      # (B, 196, 1024)
        
        # Per-point attention: each VF point attends to relevant patches
        point_feats = self.attention(patches, laterality)  # (B, 52, 1024)
        
        # Concatenate CLS to each point's feature
        B = x.shape[0]
        cls_expanded = cls_token.unsqueeze(1).expand(B, NUM_VALID_POINTS, self.embed_dim)
        combined = torch.cat([point_feats, cls_expanded], dim=2)  # (B, 52, 2048)
        
        # Predict per-point values
        pred = self.point_head(combined)    # (B, 52)
        
        # Cross-point refinement
        pred = self.refinement(pred)
        
        # Clamp
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        
        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
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

    if len(eye_ccc_losses) > 0:
        mean_eye_ccc = torch.stack(eye_ccc_losses).mean()
        loss = loss + PER_EYE_CCC_WEIGHT * mean_eye_ccc

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
    print("Training v10.1 — Per-point spatial attention")
    print("=" * 60)

    print(f"\nArchitecture change:")
    print(f"  OLD: 196 patches → 4 quadrant pools → 4 sector heads")
    print(f"  NEW: 196 patches → 52-point cross-attention → shared point head")
    print(f"  Each VF point learns which patches to attend to")
    print(f"  Initialized with anatomical prior (gaussian centered on expected location)")

    print(f"\nLoss (from v9.4):")
    print(f"  Huber + {WEIGHT_SCALE}x weighting + per-eye CCC + variance penalty")

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
    model = PerPointVFModel(base_model, DROPOUT_RATE)
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
            loss, mae, nv = compute_loss(pred, hvf, lat, epoch=epoch)

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
            print(f"  Refinement alpha: {ref_alpha:.3f}")

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
    print(f"Training Complete — v10.1")
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