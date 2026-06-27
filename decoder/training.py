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

# Encoder partial fine-tune (opt-in via --unfreeze-blocks). TINY LR so the
# pretrained RETFound features adapt to VF without catastrophic drift on 211 eyes.
ENC_LR = 1e-5
ENC_WD = 1e-4

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

# Variance penalty (iter-5 showed removing it overfits → restored)
VARIANCE_WEIGHT = 0.05
VARIANCE_START  = 8

# ── Session-3: within-eye dispersion match (anti-shrinkage) ──────
# Iter-A probes proved the model is excessively shrunk: pooled σ_pred/σ_true ≈ 0.53 while
# the SAME frozen features support ~0.82 (per-point ridge). Shrinkage flattens the scatterplot
# (low slope) and under-deepens scotomata (high floor MAE) — the severe-point bottleneck.
# This term matches each eye's WITHIN-field prediction spread to its target spread,
# σ_pred(52pts) ≈ σ_true(52pts), pushing dispersion up from real per-point structure rather
# than the global down-shift that loss-reweighting caused. 0 disables (default).
DISPERSION_WEIGHT = 0.0
DISPERSION_START  = 3

# ── NEW: Attention regularization ─────────────────────────────
ATTN_ENTROPY_WEIGHT = 0.01   # small bonus for spread-out attention
ATTN_TEMP_INIT      = 2.0    # initial temperature (>1 = softer attention)

# ── NEW: Label noise ──────────────────────────────────────────
LABEL_NOISE_STD = 0.3        # iter-5 showed removing it overfits → restored

# Validation
VAL_EVERY       = 2
TRAIN_MAE_GATE  = 5.5   # skip val when epoch train MAE is above this (still converging)
FORCE_VAL_EVERY = 10    # validate every N epochs regardless of gate (sanity / monitor)

# Clipping
OUTLIER_CLIP_RANGE = (0, 35)

# ══════════════════════════════════════════════════════════════
# TIER-1 IMPROVEMENTS (opt-in; see decoder/specs/severe_point_improvement_plan.md)
# ══════════════════════════════════════════════════════════════
# M1 — Distributional / ordinal per-point head.
#   The scalar Huber head structurally regresses rare deep points toward the
#   mean (RC-1). A parallel head predicts a DISTRIBUTION over dB bins trained
#   with soft cross-entropy, whose gradient pulls mass to the correct deep bin
#   regardless of rarity. Final pred = (1-blend)*scalar + blend*E[dist].
DIST_BIN_CENTERS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0,
                    20.0, 24.0, 28.0, 32.0, 35.0]   # 13 bins, finer near the floor
DIST_LABEL_SIGMA = 3.0     # dB; width of the Gaussian soft label over bin centers
DIST_HIDDEN      = 128     # small head to limit added params on 211 eyes
DIST_BLEND       = 0.5     # default weight on E[dist] in the blended prediction
DIST_LOSS_WEIGHT = 0.5     # default λ on the soft-CE term in compute_loss

# M2 — Label-Distribution-Smoothing (LDS) reweighting.
#   Replaces the hand-tuned value weight + floor_boost with a principled
#   inverse-(smoothed)-density weight derived from the train label distribution.
LDS_SIGMA_DB = 2.0         # dB; Gaussian kernel σ for smoothing the label density
LDS_MAX_WEIGHT = 4.0       # cap on any single bin's weight (run-4: 6 over-deepened)

# Weight EMA — exponential moving average of decoder weights, evaluated/saved
# instead of the raw weights. Low-risk variance reduction on the plateau (the
# 211-eye generalization ceiling). 0 disables.
EMA_DECAY = 0.998

# Bias control — per-eye mean-error penalty. Decouples "deep points should be
# deep" (goal 1) from "the whole field is shifted down" (the −4 dB bias runaway
# in run-4). Pins each eye's mean sensitivity so deep emphasis shapes the
# distribution without shifting the level. Set 0 to disable.
BIAS_PENALTY_WEIGHT = 0.1

# Goal-aware self-stop — give up when the run is clearly not going to hit target.
SUBGOAL_TARGET_MAE     = 4.0    # the MAE we are trying to beat (sub-4)
GOAL_PLATEAU_CHECKS    = 8      # val checks of no MEANINGFUL gain before stopping
GOAL_PLATEAU_MIN_DELTA = 0.05   # dB; improvement smaller than this doesn't count

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
    # (iter-7 ColorJitter REVERTED: val MAE 4.36 > champion 4.15 — photometric aug
    #  adds no VF-relevant signal. Clean champion baseline restored.)
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

# ============== Disc-crop view (iter-11 signal probe) ==============
# GRAPE fundus photos are macula-centered with the optic disc at a stereotyped,
# laterality-mirrored location (OD ~x=0.78, OS ~x=0.22; y~0.49). A FIXED
# laterality-aware crop zooms the optic-nerve-head / peripapillary RNFL — the
# structural basis of glaucomatous VF loss — with NO learned detector. Added as an
# extra per-eye view; the model's existing multi-image averaging fuses it with the
# full image. Probes whether higher-res disc detail raises corr (the severe ceiling).
DISC_CX_OD = 0.78
DISC_CX_OS = 0.22
DISC_CY    = 0.49
DISC_HALF  = 0.27   # half box size (fraction of W/H) → ~54% crop, resized to 224

def disc_crop_pil(img, laterality):
    w, h = img.size
    cx = DISC_CX_OD if str(laterality).startswith('OD') else DISC_CX_OS
    left   = int(max(0, (cx - DISC_HALF) * w))
    right  = int(min(w, (cx + DISC_HALF) * w))
    top    = int(max(0, (DISC_CY - DISC_HALF) * h))
    bottom = int(min(h, (DISC_CY + DISC_HALF) * h))
    if right - left < 8 or bottom - top < 8:   # degenerate guard
        return img
    return img.crop((left, top, right, bottom))


# ============== Dataset ==============
class MultiImageDataset(Dataset):
    def __init__(self, json_path, fundus_dir, transform, mode='train', use_tta=False,
                 disc_crop=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.fundus_dir = fundus_dir
        self.transform  = transform
        self.mode       = mode
        self.use_tta    = use_tta
        self.disc_crop  = disc_crop
        # Views per image: always 'full'; + 'disc' (a laterality-aware disc zoom)
        # when disc_crop is on. Train treats each (image,view) as its own sample;
        # val stacks all views of an eye and averages their predictions.
        views = ['full', 'disc'] if disc_crop else ['full']
        self.samples = []
        for item in self.data:
            images     = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            hvf        = item['hvf']
            laterality = item.get('Laterality', 'OD').strip().upper()
            patient_id = item.get('PatientID', 0)
            if self.mode == 'train':
                for img_path in images:
                    for v in views:
                        self.samples.append({'image': img_path, 'view': v, 'hvf': hvf,
                                             'laterality': laterality, 'patient_id': patient_id})
            else:
                image_views = [(p, v) for p in images for v in views]
                self.samples.append({'images': images, 'image_views': image_views,
                                     'hvf': hvf, 'laterality': laterality,
                                     'patient_id': patient_id})
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
            if sample.get('view') == 'disc':
                img = disc_crop_pil(img, sample['laterality'])
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
            image_views = sample.get('image_views', [(p, 'full') for p in sample['images']])
            for img_path, view in image_views:
                img = Image.open(os.path.join(self.fundus_dir, img_path)).convert('RGB')
                if view == 'disc':
                    img = disc_crop_pil(img, sample['laterality'])
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
# ══════════════════════════════════════════════════════════════
# TIER-1 HELPERS — distributional head + LDS reweighting (pure functions)
# ══════════════════════════════════════════════════════════════
def dist_bin_centers(device=None):
    """Fixed dB bin centers for the distributional head, as a tensor."""
    t = torch.tensor(DIST_BIN_CENTERS, dtype=torch.float32)
    return t.to(device) if device is not None else t


def build_soft_targets(values, centers, sigma=DIST_LABEL_SIGMA):
    """Gaussian soft labels over `centers` for each true value (DLDL-style).

    values (...,) true dB → (..., K) row-normalized distribution peaked at the
    nearest bin. This is label-smoothing in value space: deep bins receive
    gradient without the mean-collapse that L1/Huber imposes on rare values."""
    values = values.to(centers.device).unsqueeze(-1)             # (...,1)
    logits = -((centers - values) ** 2) / (2.0 * sigma * sigma)  # (...,K)
    return torch.softmax(logits, dim=-1)


def dist_expectation(logits, centers):
    """E[value] under softmax(logits) over `centers`. logits (...,K) → (...)."""
    centers = centers.to(logits.device)
    return (torch.softmax(logits, dim=-1) * centers).sum(dim=-1)


def dist_soft_ce(logits, values, centers, sigma=DIST_LABEL_SIGMA, weight=None):
    """Soft cross-entropy between the predicted bin distribution and the Gaussian
    soft target for each true value. logits (N,K), values (N,), optional
    per-point `weight` (N,). Returns a scalar."""
    centers = centers.to(logits.device)
    soft = build_soft_targets(values, centers, sigma)           # (N,K)
    logp = torch.log_softmax(logits, dim=-1)                    # (N,K)
    ce = -(soft * logp).sum(dim=-1)                             # (N,)
    if weight is not None:
        weight = weight.to(ce.device)
        return (ce * weight).sum() / (weight.sum() + 1e-8)
    return ce.mean()


def compute_lds_weights(values, sigma_db=LDS_SIGMA_DB, vmin=0.0, vmax=35.0,
                        max_weight=LDS_MAX_WEIGHT):
    """Label-Distribution-Smoothing per-bin loss weights (Yang et al. 2021).

    1-dB bins over [vmin,vmax); empirical density → Gaussian-smoothed density →
    inverse → normalized so the empirical-weighted mean is 1 (keeps loss on the
    dB scale). Weights are CAPPED at `max_weight` (then re-normalized) so a rare
    deep/near-empty bin can't blow up the batch loss — the bias-oscillation risk
    on a 211-eye set. The principled replacement for value weight + floor_boost."""
    values = np.asarray(values, dtype=np.float64)
    edges  = np.arange(vmin, vmax + 1e-6, 1.0)
    counts, _ = np.histogram(np.clip(values, vmin, vmax - 1e-6), bins=edges)
    counts = counts.astype(np.float64)
    radius = max(1, int(round(3 * sigma_db)))
    kx = np.arange(-radius, radius + 1)
    kernel = np.exp(-(kx ** 2) / (2.0 * sigma_db * sigma_db))
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode='same')
    raw = 1.0 / (smooth + 1e-6)
    N = counts.sum()
    emp_mean = (counts * raw).sum() / (N + 1e-8)                # weighted mean
    w = raw / (emp_mean + 1e-8)                                 # → mean 1
    if max_weight is not None:
        w = np.minimum(w, max_weight)                          # cap extremes
        emp2 = (counts * w).sum() / (N + 1e-8)
        w = w / (emp2 + 1e-8)                                  # restore mean 1
    return torch.tensor(w, dtype=torch.float32)


def lds_lookup(weights, values):
    """Per-point LDS weight: bin each value (1-dB bins) and gather. values any
    shape → same shape. Clamped to the valid bin range."""
    idx = torch.clamp(values.long(), 0, weights.numel() - 1)
    return weights.to(values.device)[idx]


def load_train_db_values(json_path):
    """Flatten all valid (unmasked) dB sensitivities from a GRAPE-format JSON.
    Used once at startup to build the LDS weights from the train distribution."""
    with open(json_path) as f:
        data = json.load(f)
    records = data.values() if isinstance(data, dict) else data
    vals = []
    for rec in records:
        arr = np.array(rec['hvf'], dtype=np.float32).flatten()
        vals.extend(arr[arr < MASKED_VALUE_THRESHOLD].tolist())
    return np.array(vals, dtype=np.float32)


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


# ============== Distributional Point Head (M1) ==============
class DistPointHead(nn.Module):
    """Per-point distribution over dB bins, shared across all 52 points.

    Trained with soft cross-entropy (dist_soft_ce); its gradient pulls mass to
    the correct deep bin regardless of rarity, so deep points are not collapsed
    toward the mean the way the scalar Huber head collapses them (RC-1)."""
    def __init__(self, input_dim=2048, hidden=DIST_HIDDEN,
                 n_bins=len(DIST_BIN_CENTERS), dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_bins),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)   # (...,K) logits


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
    def __init__(self, encoder, use_dist=False, dist_blend=DIST_BLEND, unfreeze_blocks=0,
                 mean_residual=False, global_head=False, finetune_norm=False):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = 1024
        self.unfreeze_blocks = unfreeze_blocks
        self.finetune_norm = finetune_norm

        for p in self.encoder.parameters():
            p.requires_grad = False
        if finetune_norm:
            # BitFit: tune ONLY LayerNorm scale/shift + biases throughout the encoder (~tens of k
            # params on 24 blocks). Adapts RETFound's feature STATISTICS to the GRAPE/glaucoma domain
            # with far less overfit risk than a full-block unfreeze (the prior session's 12.6M-param
            # unfreeze overfit). Trainable params are spread across all blocks → no frozen prefix.
            n_enc = 0
            for name, p in self.encoder.named_parameters():
                if 'norm' in name or name.endswith('.bias'):
                    p.requires_grad = True; n_enc += p.numel()
            print(f"✓ Encoder: BitFit (norm+bias only) UNFROZEN ({n_enc:,} params; rest frozen) "
                  f"— regularized domain adaptation, attacks the eyeCorr≈0.51 frozen cap")
        elif unfreeze_blocks > 0:
            for blk in self.encoder.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.encoder.norm.parameters():
                p.requires_grad = True
            n_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            print(f"✓ Encoder: last {unfreeze_blocks} block(s) + norm UNFROZEN "
                  f"({n_enc:,} trainable enc params; rest frozen) — attacks the corr ceiling")
        else:
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

        # M1 — distributional head. Always built (so checkpoints load with
        # strict=False either way); only USED when use_dist=True.
        self.use_dist   = use_dist
        self.dist_blend = dist_blend
        self.dist_head  = DistPointHead(input_dim=self.embed_dim * 2,
                                        hidden=DIST_HIDDEN,
                                        n_bins=len(DIST_BIN_CENTERS),
                                        dropout=HEAD_DROPOUT)
        self.register_buffer('dist_centers',
                             torch.tensor(DIST_BIN_CENTERS, dtype=torch.float32))
        self._last_dist_logits = None

        # Session-3 mean+residual decomposition (opt-in; always built so checkpoints load with
        # strict=False either way). CLS→eye-mean (severity = the SOLVED part); point_head output
        # is re-interpreted as a zero-mean within-eye RESIDUAL so the spatial head is no longer
        # drowned/over-regularized by the dominant severity signal (MLP probe: frozen features
        # support eyeCorr ~0.51 but the lumped head only reaches ~0.42).
        self.mean_residual = mean_residual
        self.mean_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(HEAD_DROPOUT), nn.Linear(128, 1))
        nn.init.constant_(self.mean_head[-1].bias, PROJ_INIT_BIAS)
        if mean_residual:
            print("✓ Mean+residual head ON (CLS→eye-mean ; point_head→zero-mean spatial residual)")

        # Session-3 JOINT global-spatial head (opt-in). Probes: the within-eye PATTERN is best
        # predicted from GLOBAL mean-pooled patches+CLS via a joint (global→52) map (eyeCorr 0.51)
        # — a SHARED per-point head is capped at the local-feature eyeCorr (~0.41 = champion).
        # Pairs with the mean head: pred = mean_head(CLS) + zero-mean(global_spatial([pool‖CLS])).
        self.use_global_head = global_head
        self.global_spatial = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(HEAD_DROPOUT),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(HEAD_DROPOUT),
            nn.Linear(256, NUM_VALID_POINTS))
        nn.init.normal_(self.global_spatial[-1].weight, std=0.01)
        nn.init.zeros_(self.global_spatial[-1].bias)
        if global_head:
            print("✓ Joint global-spatial head ON (ADDITIVE: per-point pred + zero-mean global "
                  "pattern from mean-pool patches‖CLS; zero-init=no-op start) — targets eyeCorr 0.51")

        if use_dist:
            print(f"✓ Distributional head ON (bins={len(DIST_BIN_CENTERS)}, "
                  f"blend={dist_blend})")

        self.refinement = CrossPointRefinement(NUM_VALID_POINTS, hidden=104, dropout=REFINE_DROPOUT)
        print("✓ Cross-point refinement")

        attn_p = sum(p.numel() for p in self.attention.parameters())
        head_p = sum(p.numel() for p in self.point_head.parameters())
        ref_p  = sum(p.numel() for p in self.refinement.parameters())
        total  = attn_p + head_p + ref_p
        if use_dist:
            dist_p = sum(p.numel() for p in self.dist_head.parameters())
            total += dist_p
            print(f"  Trainable: attention={attn_p:,} + head={head_p:,} + "
                  f"dist={dist_p:,} + refinement={ref_p:,} = {total:,}")
        else:
            print(f"  Trainable: attention={attn_p:,} + head={head_p:,} + refinement={ref_p:,} = {total:,}")

    def _apply_heads(self, point_feats, cls_token, B):
        """Scalar (+ optional distributional) heads → blended per-point pred.
        Stores the per-image dist logits on self._last_dist_logits."""
        cls_expanded = cls_token.unsqueeze(1).expand(B, NUM_VALID_POINTS, self.embed_dim)
        combined = torch.cat([point_feats, cls_expanded], dim=2)
        scalar = self.point_head(combined)                          # (B,52)
        if self.mean_residual:
            m = self.mean_head(cls_token)                           # (B,1) eye-mean (severity)
            resid = scalar - scalar.mean(dim=1, keepdim=True)       # zero-mean spatial pattern
            pred = m + resid
            self._last_dist_logits = None
        elif self.use_dist:
            logits = self.dist_head(combined)                       # (B,52,K)
            e = dist_expectation(logits, self.dist_centers)         # (B,52)
            pred = (1.0 - self.dist_blend) * scalar + self.dist_blend * e
            self._last_dist_logits = logits
        else:
            pred = scalar
            self._last_dist_logits = None
        return pred

    def _finish(self, pred, average_multi):
        pred = self.refinement(pred)
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred

    def _encode(self, x):
        """Frozen RETFound encoder WITHOUT the MAE random patch shuffle.

        encoder.forward_encoder(x, mask_ratio=0) routes through random_masking,
        which — even at mask_ratio=0 — returns the 196 patch tokens in a RANDOM
        per-call order (ids_shuffle=argsort(noise), keep all) and drops ids_restore.
        That scrambles the spatial layout, so PerPointAttention's fixed row-major
        patch_pos and the Garway–Heath anatomical prior point at random patches →
        the per-point/spatial machinery is effectively dead and attention collapses
        to order-invariant content pooling (the real cause of the severe-point /
        slope ceiling). Reproducing the encoder forward without random_masking keeps
        patches in proper row-major order so the prior + patch_pos are meaningful."""
        enc = self.encoder
        # BitFit: trainable norm/bias params are spread through ALL blocks → the whole forward
        # must build a graph (no frozen prefix to skip). Slightly slower / more memory, but the
        # only way gradients reach the throughout norms.
        if getattr(self, 'finetune_norm', False):
            h = enc.patch_embed(x)
            h = h + enc.pos_embed[:, 1:, :]
            cls = (enc.cls_token + enc.pos_embed[:, :1, :]).expand(h.shape[0], -1, -1)
            h = torch.cat((cls, h), dim=1)
            for blk in enc.blocks:
                h = blk(h)
            return enc.norm(h)
        n_unf = getattr(self, 'unfreeze_blocks', 0)
        n_frozen = len(enc.blocks) - n_unf
        # Frozen prefix always runs under no_grad (no graph, no memory cost).
        with torch.no_grad():
            h = enc.patch_embed(x)
            h = h + enc.pos_embed[:, 1:, :]
            cls = (enc.cls_token + enc.pos_embed[:, :1, :]).expand(h.shape[0], -1, -1)
            h = torch.cat((cls, h), dim=1)
            for blk in enc.blocks[:n_frozen]:
                h = blk(h)
        # Trainable suffix (if any) runs WITH grad so backprop reaches its params.
        if n_unf > 0:
            for blk in enc.blocks[n_frozen:]:
                h = blk(h)
            h = enc.norm(h)
        else:
            with torch.no_grad():
                h = enc.norm(h)
        return h

    def _global_residual(self, cls_token, patches):
        """Zero-mean within-eye PATTERN from a JOINT global→52 map on [mean-pool patches ‖ CLS]
        (eyeCorr 0.51 in probes — a shared per-point head is capped at ~0.41). Added on top of
        the per-point path so it only contributes the pattern the local head misses; zero-init
        ⇒ starts as a no-op (degrades gracefully to the proven control)."""
        pooled = patches.mean(dim=1)                                       # (B,1024)
        g = self.global_spatial(torch.cat([pooled, cls_token], dim=1))     # (B,52)
        return g - g.mean(dim=1, keepdim=True)                             # zero-mean

    def forward(self, x, laterality='OD', average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")

        # _encode manages no_grad internally (all-frozen → no graph; partial unfreeze
        # → graph only through the trainable suffix blocks).
        latent = self._encode(x)
        cls_token = latent[:, 0, :]
        patches   = latent[:, 1:, :]

        point_feats, attn_weights = self.attention(patches, laterality)
        pred = self._apply_heads(point_feats, cls_token, x.shape[0])
        if self.use_global_head:
            pred = pred + self._global_residual(cls_token, patches)
        pred = self._finish(pred, average_multi)
        self._last_attn_weights = attn_weights   # for entropy regularization (training)
        return pred

    def decode_latent(self, latent, laterality='OD', average_multi=True):
        """Trainable-decoder-only forward on a pre-cached encoder latent (skips the frozen
        encoder; ~50× faster val). Mirrors forward()'s head routing."""
        cls_token = latent[:, 0, :]
        patches   = latent[:, 1:, :]
        point_feats, attn_weights = self.attention(patches, laterality)
        pred = self._apply_heads(point_feats, cls_token, latent.shape[0])
        if self.use_global_head:
            pred = pred + self._global_residual(cls_token, patches)
        pred = self._finish(pred, average_multi)
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
def compute_loss(pred, target, laterality, epoch=0, attn_weights=None,
                 sector_weights=None, sector_combine='both', deep_cfg=None,
                 dist_logits=None, dist_cfg=None, lds_weights=None,
                 bias_penalty=BIAS_PENALTY_WEIGHT, dispersion_weight=DISPERSION_WEIGHT):
    """Weighted Huber loss (+ CCC / variance / attention-entropy terms).

    Garway–Heath sector weighting is opt-in and fully backward-compatible:
      * sector_weights=None (default)  → IDENTICAL to the original baseline.
      * sector_weights={'OD':t52,'OS':t52} (query-order, e.g. from
        garway_heath_weighting.sector_weight_tensors) layers a spatial weight on
        top of the existing value-based weight. `sector_combine` controls the
        interaction: 'both' (value × sector), 'sector_only', or 'value_only'.

    Deep-floor shaping is also opt-in (deep_cfg=None → baseline). When given
    (from garway_heath_weighting.deep_loss_config) it (1) adds an additive
    'floor boost' weight to the deepest GT points and (2) applies an asymmetric
    extra Huber penalty when the model OVER-predicts a deep point — together
    targeting the 0-10 dB underestimation / positive-bias seen in the first run.

    Only the per-point Huber term is affected; the returned MAE stays unweighted
    so validation MAE remains apples-to-apples with the baseline.
    """
    device = pred.device
    target = target.to(device)
    if isinstance(laterality, str):
        laterality = [laterality]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    total_huber = total_mae = n_valid = 0
    total_dist_ce = 0.0
    eye_ccc_losses = []
    bias_sq_terms  = []
    disp_terms     = []

    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        target_52 = target[i][valid_idx]
        pred_52   = pred[i]
        mask      = target_52 < MASKED_VALUE_THRESHOLD
        if mask.sum() == 0:
            continue
        p = pred_52[mask]
        t = target_52[mask]

        if lds_weights is not None:
            # M2 — LDS inverse-density weight (replaces value weight + floor_boost).
            value_w = lds_lookup(lds_weights, t)
        else:
            # Existing value-based (severity) weight.
            value_w = 1.0 + WEIGHT_SCALE * (MAX_DB - t).clamp(min=0) / MAX_DB
            # Deep-floor boost (goal-1 rescue) — additive extra weight on the
            # deepest GT points, layered onto the severity weight. deep_cfg=None
            # (baseline) leaves value_w untouched.
            if deep_cfg is not None and deep_cfg.get('floor_boost', 0.0) > 0:
                floor_db = deep_cfg['floor_db']
                value_w = value_w + deep_cfg['floor_boost'] * \
                          (floor_db - t).clamp(min=0) / floor_db
        # New spatial (Garway–Heath) weight, layered in behind the flag.
        if sector_weights is not None:
            eye = 'OD' if lat.startswith('OD') else 'OS'
            sector_w = sector_weights[eye].to(device)[mask]   # query-order → masked subset
            if sector_combine == 'sector_only':
                weights = sector_w
            elif sector_combine == 'value_only':
                weights = value_w
            else:  # 'both'
                weights = value_w * sector_w
        else:
            weights = value_w
        # Per-point Huber, with an asymmetric penalty on OVER-predicting deep
        # points (pred>gt AND gt<floor) — directly counters the positive bias /
        # scotoma-depth underestimation seen in the first GH run.
        huber_pp = F.huber_loss(p, t, reduction='none', delta=HUBER_DELTA)
        if deep_cfg is not None and deep_cfg.get('overpred_penalty', 0.0) > 0:
            over_deep = (p > t) & (t < deep_cfg['floor_db'])
            huber_pp = huber_pp * torch.where(
                over_deep,
                torch.full_like(p, 1.0 + deep_cfg['overpred_penalty']),
                torch.ones_like(p))
        huber = (huber_pp * weights).mean()
        mae   = (p - t).abs().mean()

        total_huber += huber * mask.sum().item()
        total_mae   += mae.item() * mask.sum().item()
        n_valid     += mask.sum().item()

        # M1 — distributional soft cross-entropy. dist_logits are in query order
        # (same as pred_52), gathered by the same mask. UNWEIGHTED: the soft-CE
        # already gives every deep point a full-strength per-point gradient (its
        # whole advantage); weighting it by the deep-heavy LDS weights triple-
        # counted the deep emphasis and drove the −4 dB bias runaway in run-4.
        if dist_logits is not None and dist_cfg is not None:
            dl = dist_logits[i][mask]                            # (nv, K)
            ce = dist_soft_ce(dl, t, dist_cfg['centers'],
                              sigma=dist_cfg.get('sigma', DIST_LABEL_SIGMA),
                              weight=None)
            total_dist_ce = total_dist_ce + ce * mask.sum().item()

        # Bias control — per-eye squared mean error (pins the eye's level).
        if bias_penalty > 0:
            bias_sq_terms.append((p - t).mean() ** 2)

        # Within-eye dispersion match (anti-shrinkage) — pull σ_pred(field) up to σ_true(field).
        if dispersion_weight > 0 and epoch >= DISPERSION_START and mask.sum() >= 5:
            disp_terms.append((p.std(unbiased=False) - t.std(unbiased=False)) ** 2)

        if epoch >= PER_EYE_CCC_START and mask.sum() >= 5:
            eye_ccc = per_eye_ccc(pred_52, target_52, mask)
            eye_ccc_losses.append(eye_ccc)

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0

    loss = total_huber / n_valid

    # M1 — distributional soft-CE term (the geometry-preserving deep-point loss).
    if dist_logits is not None and dist_cfg is not None and \
       isinstance(total_dist_ce, torch.Tensor):
        loss = loss + dist_cfg.get('weight', DIST_LOSS_WEIGHT) * (total_dist_ce / n_valid)

    # Bias-control term — drives each eye's mean error toward 0 (decouples the
    # global level from the deep-point emphasis; the run-4 negative-bias fix).
    if bias_penalty > 0 and len(bias_sq_terms) > 0:
        loss = loss + bias_penalty * torch.stack(bias_sq_terms).mean()

    # Within-eye dispersion-match term (anti-shrinkage; targets slope/floor, the severe goal).
    if dispersion_weight > 0 and len(disp_terms) > 0:
        loss = loss + dispersion_weight * torch.stack(disp_terms).mean()

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


def _flat_severity_stats(pred, target, laterality):
    """Flat (masked) bias + deep-region MAEs for live training diagnostics.

    Returns {'bias','rmse','floor_mae','floor_n','deep_mae','deep_n'} where
    floor = GT 0-10 dB (the goal-1 metric) and deep = GT < 16 dB."""
    if isinstance(laterality, str):
        laterality = [laterality]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    P, T = [], []
    for i in range(min(len(laterality), pred.shape[0], target.shape[0])):
        lat = laterality[i] if i < len(laterality) else laterality[0]
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        t52 = target[min(i, target.shape[0]-1)][valid_idx]
        p52 = pred[min(i, pred.shape[0]-1)]
        m = t52 < MASKED_VALUE_THRESHOLD
        if m.sum() > 0:
            P.extend(p52[m].tolist())
            T.extend(t52[m].tolist())
    if len(P) == 0:
        return {}
    P = np.array(P); T = np.array(T); err = P - T; ae = np.abs(err)

    def band(lo, hi):
        m = (T >= lo) & (T < hi)
        return (float(ae[m].mean()) if m.any() else float('nan')), int(m.sum())

    floor_mae, floor_n = band(0, 10)
    deep_mae,  deep_n  = band(0, 16)
    return {'bias': float(err.mean()), 'rmse': float(np.sqrt((err ** 2).mean())),
            'floor_mae': floor_mae, 'floor_n': floor_n,
            'deep_mae': deep_mae,  'deep_n': deep_n}


def evaluate(model, loader, detailed=False):
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
        return (float('inf'), 0.0, 0.0, 0.0, 0.0, {}) if detailed \
            else (float('inf'), 0.0, 0.0, 0.0, 0.0)
    stacked_preds   = torch.cat(all_preds, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    corr = pearson_correlation(stacked_preds, stacked_targets, all_lats)
    r2, slope = compute_r2_slope(stacked_preds, stacked_targets, all_lats)
    per_eye_corr = compute_per_eye_metrics(stacked_preds, stacked_targets, all_lats)
    mae = total_mae / n_valid
    if not detailed:
        return mae, corr, r2, slope, per_eye_corr
    extra = _flat_severity_stats(stacked_preds, stacked_targets, all_lats)
    return mae, corr, r2, slope, per_eye_corr, extra


@torch.no_grad()
def precompute_features(model, loader, device, desc=""):
    """Cache frozen encoder outputs for every sample in a val-mode loader.

    Returns list of {'latent': cpu_tensor[N,197,1024], 'hvf': tensor, 'lat': str}.
    One-time cost at the start of training; each subsequent val check runs
    evaluate_from_cache() and skips the encoder entirely."""
    model.eval()
    cache = []
    for sample in tqdm(loader, desc=f"  Caching {desc}", leave=False):
        imgs, hvf, lat = sample
        imgs = imgs.to(device)
        latent = model._encode(imgs)   # unshuffled (see PerPointVFModel._encode)
        cache.append({'latent': latent.cpu(), 'hvf': hvf, 'lat': lat})
    return cache


def evaluate_from_cache(model, cache, device, detailed=False):
    """Fast evaluation using precomputed encoder latents.

    Identical accuracy to evaluate(); skips the frozen encoder forward pass so
    a full val check takes <1 s instead of ~30 s on MPS."""
    model.eval()
    total_mae = n_valid = 0
    all_preds, all_targets, all_lats = [], [], []
    with torch.no_grad():
        for item in cache:
            latent = item['latent'].to(device)
            hvf    = item['hvf']
            lat    = item['lat']
            pred   = model.decode_latent(latent, lat, average_multi=True)
            _, mae, nv = compute_loss(pred, hvf, lat, epoch=0)
            total_mae += mae * nv
            n_valid   += nv
            all_preds.append(pred.cpu())
            all_targets.append(hvf.unsqueeze(0) if hvf.dim() == 1 else hvf)
            all_lats.append(lat)
    if n_valid == 0:
        return (float('inf'), 0.0, 0.0, 0.0, 0.0, {}) if detailed \
            else (float('inf'), 0.0, 0.0, 0.0, 0.0)
    stacked_preds   = torch.cat(all_preds, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    corr         = pearson_correlation(stacked_preds, stacked_targets, all_lats)
    r2, slope    = compute_r2_slope(stacked_preds, stacked_targets, all_lats)
    per_eye_corr = compute_per_eye_metrics(stacked_preds, stacked_targets, all_lats)
    mae = total_mae / n_valid
    if not detailed:
        return mae, corr, r2, slope, per_eye_corr
    extra = _flat_severity_stats(stacked_preds, stacked_targets, all_lats)
    return mae, corr, r2, slope, per_eye_corr, extra


def diagnose_training(history, target_mae=SUBGOAL_TARGET_MAE):
    """Inspect the validation trajectory and decide whether to auto-stop with a
    plain-English reason. Pure function of the recorded history dict (lists keyed
    by 'val_mae','val_slope','val_bias','val_floor').

    Returns (should_stop: bool, reason: str|None, warnings: list[str]).
    Warnings are printed live every check; a reason means STOP now."""
    warns = []
    mae   = history['val_mae']
    slope = history['val_slope']
    bias  = history['val_bias']
    floor = history['val_floor']
    cur_mae, cur_slope, cur_bias = mae[-1], slope[-1], bias[-1]

    # Hard stop 1 — numerical blow-up.
    if not all(np.isfinite([cur_mae, cur_slope, cur_bias])):
        return True, ("NUMERICAL INSTABILITY — a validation metric went "
                      "non-finite (NaN/Inf). Usually exploding gradients or too "
                      "high an LR. Lower the LR or inspect the inputs."), warns

    # Hard stop 2 — sustained divergence (val MAE climbing well off its best).
    if len(mae) >= 4:
        best = min(mae[:-1])
        if mae[-1] > mae[-2] > mae[-3] and cur_mae > best + 1.0:
            return True, (f"VAL MAE DIVERGING — rose 3 checks straight to "
                          f"{cur_mae:.2f} dB, >1 dB above the best ({best:.2f}). "
                          f"The model is unlearning/overfitting; the best "
                          f"checkpoint is already saved."), warns

    # Hard stop 3 — slope collapse (predictions flattening to a constant mean).
    # Guard: require ≥8 val evaluations so initialization-phase low slope
    # (naturally near 0 in the first ~20 epochs) cannot trigger this.
    if len(slope) >= 8 and all(s < 0.08 for s in slope[-8:]):
        return True, (f"SLOPE COLLAPSE — val slope stuck <0.08 for 8 consecutive "
                      f"checks (now {cur_slope:.3f}); predictions are regressing "
                      f"to a constant. Ease regularization (dropout/weight-decay) "
                      f"or the variance penalty."), warns

    # Hard stop 4 — goal plateau (clearly won't reach the MAE target). Counts
    # only MEANINGFUL improvements (>GOAL_PLATEAU_MIN_DELTA) so a trickle of
    # 0.01 dB gains can't keep a doomed run alive (the loophole seen in run-3).
    finite_mae = [m for m in mae if np.isfinite(m)]
    if len(mae) >= GOAL_PLATEAU_CHECKS + 1 and finite_mae:
        running_best = float('inf')
        last_improve = 0
        for idx, m in enumerate(mae):
            if np.isfinite(m) and m < running_best - GOAL_PLATEAU_MIN_DELTA:
                running_best = m
                last_improve = idx
            elif np.isfinite(m) and m < running_best:
                running_best = m   # tiny gain updates best but not "last_improve"
        checks_since = (len(mae) - 1) - last_improve
        best_so_far  = min(finite_mae)
        if checks_since >= GOAL_PLATEAU_CHECKS and best_so_far > target_mae + 0.05:
            return True, (f"GOAL PLATEAU — no meaningful val-MAE gain "
                          f"(>{GOAL_PLATEAU_MIN_DELTA} dB) for {checks_since} checks; "
                          f"best {best_so_far:.2f} dB stays above the {target_mae:.1f} dB "
                          f"target. Further training won't reach it — switch methods "
                          f"(decoder/specs/severe_point_improvement_plan.md)."), warns

    # Soft warnings (printed live; do NOT stop).
    # Guard: require ≥5 checks so uninitialized-model noise doesn't fire these.
    if len(slope) >= 5 and cur_slope < 0.20:
        warns.append(f"slope low ({cur_slope:.3f}) — watch for regression-to-mean")
    if len(bias) >= 5 and np.isfinite(cur_bias) and abs(cur_bias) > 2.0:
        if cur_bias > 0:
            warns.append(f"large +bias ({cur_bias:+.2f} dB) — scotoma depth is "
                         f"being UNDER-estimated; raise --overpred-penalty")
        else:
            warns.append(f"large -bias ({cur_bias:+.2f} dB) — depth is being "
                         f"OVER-estimated; lower --overpred-penalty/--floor-boost")
    if len(floor) >= 5 and floor[-1] > floor[-2] > floor[-3]:
        warns.append(f"deep-floor (0-10 dB) MAE rising ({floor[-1]:.2f}) — the "
                     f"goal-1 metric is regressing; consider raising "
                     f"--floor-boost / --overpred-penalty")
    return False, None, warns


# ============== Weight EMA ==============
class WeightEMA:
    """Exponential moving average of the trainable decoder params. Evaluated and
    saved instead of the raw weights to cut val variance on the plateau."""
    def __init__(self, modules, decay=EMA_DECAY):
        self.decay = decay
        self.step = 0
        self.params = [p for m in modules for p in m.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self._backup = None

    @torch.no_grad()
    def update(self):
        # Warmup: low effective decay early so the average tracks the model from
        # the start instead of anchoring to the (near-constant) init weights —
        # the iter-6 slope-collapse bug on short runs.
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self):
        """Swap EMA weights into the live model (keeps a backup to restore)."""
        self._backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.copy_(s)

    @torch.no_grad()
    def restore(self):
        if self._backup is None:
            return
        for p, b in zip(self.params, self._backup):
            p.copy_(b)
        self._backup = None


# ============== Champion tracking (best-ever model + methods) ==============
CHAMPION_DIR = os.path.join(CURRENT_DIR, "results", "champion")

def _update_champion(best_mae, best_extra, config, model_src):
    """Persist the best-ever model + the exact methods that produced it.

    Keeps decoder/results/champion/{best_model.pth, champion.json}. Only
    overwrites when this run's best val MAE strictly beats the stored champion,
    so the all-time-best survives across autonomous iterations."""
    import shutil, datetime
    os.makedirs(CHAMPION_DIR, exist_ok=True)
    rec_path = os.path.join(CHAMPION_DIR, "champion.json")
    prev = None
    if os.path.exists(rec_path):
        try:
            with open(rec_path) as f:
                prev = json.load(f)
        except Exception:
            prev = None
    prev_mae = prev.get('val_mae', float('inf')) if prev else float('inf')

    if np.isfinite(best_mae) and best_mae < prev_mae:
        if model_src and os.path.exists(model_src):
            shutil.copy(model_src, os.path.join(CHAMPION_DIR, "best_model.pth"))
        rec = {'val_mae': float(best_mae),
               'severe': best_extra or {},
               'methods': config,
               'previous_best': (None if not np.isfinite(prev_mae) else float(prev_mae)),
               'timestamp': datetime.datetime.now().isoformat(timespec='seconds')}
        with open(rec_path, 'w') as f:
            json.dump(rec, f, indent=2, default=str)
        prev_str = f"{prev_mae:.3f}" if np.isfinite(prev_mae) else "—"
        print(f"\n🏆 NEW CHAMPION: val MAE {best_mae:.3f} dB (prev {prev_str}). "
              f"Model + methods → {CHAMPION_DIR}")
        return True
    prev_str = f"{prev_mae:.3f}" if np.isfinite(prev_mae) else "—"
    cur_str  = f"{best_mae:.3f}" if np.isfinite(best_mae) else "inf"
    print(f"\n(Champion unchanged: this run {cur_str} ≥ best-ever {prev_str} dB)")
    return False


# ============== Training ==============
def train(weighting='baseline', sector_combine='both', epochs=EPOCHS,
          out_best=BEST_SAVE, out_inference=INFERENCE_SAVE, deep_cfg=None,
          use_dist=False, dist_blend=DIST_BLEND, dist_loss_weight=DIST_LOSS_WEIGHT,
          reweight='value', lds_sigma=LDS_SIGMA_DB, target_mae=SUBGOAL_TARGET_MAE,
          bias_penalty=BIAS_PENALTY_WEIGHT,
          attn_dropout=None, head_dropout=None, weight_decay=None, lr=None,
          ema_decay=EMA_DECAY, disc_crop=False, unfreeze_blocks=0,
          update_champion=True, dispersion_weight=DISPERSION_WEIGHT,
          label_noise=None, entropy_weight=None, mean_residual=False, global_head=False,
          finetune_norm=False, enc_lr=None, heavy_aug=False, batch_size=None):
    # Regularization / LR overrides (for fast autonomous sweeps without edits).
    global ATTN_DROPOUT, HEAD_DROPOUT, ATTN_WD, ATTN_LR, LABEL_NOISE_STD, ATTN_ENTROPY_WEIGHT, ENC_LR
    if attn_dropout is not None: ATTN_DROPOUT = attn_dropout
    if head_dropout is not None: HEAD_DROPOUT = head_dropout
    if weight_decay is not None: ATTN_WD = weight_decay
    if lr is not None: ATTN_LR = lr
    if enc_lr is not None: ENC_LR = enc_lr
    # Session-3: knobs to UN-wash the within-eye spatial signal (MLP probe showed the
    # frozen features support eyeCorr ~0.51 but the champion only reaches 0.41 — the entropy
    # bonus + label noise over-smooth/shrink the per-point predictions).
    if label_noise is not None:    LABEL_NOISE_STD = label_noise
    if entropy_weight is not None: ATTN_ENTROPY_WEIGHT = entropy_weight

    print("=" * 60)
    print("Training v10.2 — Per-point attention + Garway–Heath")
    print("=" * 60)

    # ── Garway–Heath sector weighting (opt-in) ─────────────────
    # weighting='baseline'      → original behavior (value-based weight only).
    # weighting='garway_heath'  → add the spatial sector weight + deep-floor
    #                             shaping (deep_cfg) in compute_loss.
    sector_weights = None
    if weighting == 'garway_heath':
        from garway_heath_weighting import (sector_weight_tensors,
                                            save_resolved_config, RESULTS_DIR,
                                            deep_loss_config)
        sector_weights = sector_weight_tensors(device=DEVICE, normalize=True)
        if deep_cfg is None:
            deep_cfg = deep_loss_config()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_resolved_config(extra={'combine': sector_combine, 'epochs': epochs,
                                    'deep_loss': deep_cfg})
        print(f"✓ Garway–Heath weighting ON  (combine={sector_combine})")
        print(f"  Deep-floor rescue: floor_db={deep_cfg['floor_db']:.1f} dB | "
              f"floor_boost={deep_cfg['floor_boost']:.2f} | "
              f"overpred_penalty={deep_cfg['overpred_penalty']:.2f}")
        print(f"  Resolved config + checkpoints → {RESULTS_DIR}")
    else:
        deep_cfg = None
        print("✓ Weighting: baseline (value-based only)")

    # ── Tier-1 improvements (opt-in) ───────────────────────────
    # M2 — LDS reweighting replaces the value weight + floor_boost with a
    # principled, capped inverse-density weight from the train distribution.
    lds_weights = None
    if reweight == 'lds':
        vals = load_train_db_values(TRAIN_JSON)
        lds_weights = compute_lds_weights(vals, sigma_db=lds_sigma).to(DEVICE)
        deep_cfg = None   # LDS subsumes floor_boost; drop the asymmetric overpred
                          # penalty that drove the bias oscillation in run-2/3.
        print(f"✓ M2 LDS reweighting ON (σ={lds_sigma} dB, cap={LDS_MAX_WEIGHT}, "
              f"range=[{lds_weights.min():.2f},{lds_weights.max():.2f}]) "
              f"— floor_boost/overpred disabled")
    # M1 — distributional head + soft-CE term.
    dist_cfg = None
    if use_dist:
        dist_cfg = {'centers': dist_bin_centers(DEVICE),
                    'sigma': DIST_LABEL_SIGMA, 'weight': dist_loss_weight}
        print(f"✓ M1 distributional head ON (bins={len(DIST_BIN_CENTERS)}, "
              f"blend={dist_blend}, ce_weight={dist_loss_weight})")
    if bias_penalty > 0:
        print(f"✓ Bias-control penalty ON (weight={bias_penalty}) — pins per-eye "
              f"mean so deep emphasis can't shift the global level")
    if dispersion_weight > 0:
        print(f"✓ Dispersion-match ON (weight={dispersion_weight}, start ep {DISPERSION_START}) "
              f"— anti-shrinkage: σ_pred(field)→σ_true(field); targets slope/floor (severe goal)")
    print(f"✓ Goal self-stop: give up if no >{GOAL_PLATEAU_MIN_DELTA} dB val gain "
          f"for {GOAL_PLATEAU_CHECKS} checks while best > {target_mae} dB")

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
    if disc_crop:
        print("✓ Disc-crop view ON — each eye also gets a laterality-aware optic-disc "
              "zoom (extra view, prediction-averaged with the full image)")
    # Heavier augmentation for encoder fine-tuning (208 eyes overfit easily). Mild geometric
    # only (preserve the anatomical prior orientation) + light photometric; no aggressive crop.
    tr_tfm = train_transform
    if heavy_aug:
        tr_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("✓ Heavy augmentation ON (rot10 + mild RRC + ColorJitter) — for encoder fine-tune")
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, tr_tfm, mode='train',
                                      disc_crop=disc_crop)
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val',
                                      use_tta=USE_TTA, disc_crop=disc_crop)

    bs = batch_size or BATCH_SIZE
    if bs != BATCH_SIZE:
        print(f"  Batch size override: {bs} (default {BATCH_SIZE})")
    if reweight == 'lds':
        # LDS reweights at the LOSS level, so it REPLACES severity resampling
        # (Yang et al. 2021). Stacking both triple-counted the deep emphasis and
        # drove the run-4 bias runaway → use uniform sampling here.
        train_loader = DataLoader(train_dataset, bs, shuffle=True,
                                  num_workers=0, drop_last=True)
        print(f"  Sampler: UNIFORM (LDS handles imbalance at the loss level)")
    else:
        sample_weights = train_dataset.get_sample_severity()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        print(f"  Sampler weight range: [{min(sample_weights):.1f}, {max(sample_weights):.1f}]")
        train_loader = DataLoader(train_dataset, bs, sampler=sampler,
                                  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=val_collate_fn)

    # ── Model ──────────────────────────────────────────────────
    model = PerPointVFModel(base_model, use_dist=use_dist, dist_blend=dist_blend,
                            unfreeze_blocks=unfreeze_blocks, mean_residual=mean_residual,
                            global_head=global_head, finetune_norm=finetune_norm)
    model.to(DEVICE)
    enc_trainable = (unfreeze_blocks > 0) or finetune_norm

    # ── Pre-cache frozen encoder features (one-time cost) ──────
    # The RETFound encoder is frozen throughout training, so its outputs for
    # the deterministic val/train-eval sets never change. Cache them once now;
    # each subsequent val check runs only the small trainable decoder (~1 M
    # params) rather than the full 1.2 GB ViT-Large — ~50× faster per check.
    train_eval_ds  = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, val_transform,
                                       mode='val', use_tta=False, disc_crop=disc_crop)
    train_eval_pre = DataLoader(train_eval_ds, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=val_collate_fn)
    if not enc_trainable:
        print(f"\nPre-caching encoder features for val + train-eval sets …")
        val_cache        = precompute_features(model, val_loader,    DEVICE, "val")
        train_eval_cache = precompute_features(model, train_eval_pre, DEVICE, "train-eval")
        print(f"  Cached {len(val_cache)} val + {len(train_eval_cache)} train-eval samples.")
        print(f"  Val checks are now ~50× faster (decoder-only, encoder skipped).")
    else:
        val_cache = train_eval_cache = None
        mode = "BitFit norm+bias" if finetune_norm else f"last {unfreeze_blocks} block(s)"
        print(f"\nEncoder is trainable ({mode}) → feature caching DISABLED "
              f"(features change each step); val runs the full encoder live (slower but correct).")
    print(f"  Val gate: skip unless epoch train MAE < {TRAIN_MAE_GATE} dB "
          f"(forced every {FORCE_VAL_EVERY} epochs).\n")

    # ── Optimizer ──────────────────────────────────────────────
    attn_head_params = list(model.attention.parameters()) + \
                       list(model.point_head.parameters())
    if use_dist:
        attn_head_params += list(model.dist_head.parameters())
    if mean_residual:
        attn_head_params += list(model.mean_head.parameters())
    if global_head:
        attn_head_params += list(model.global_spatial.parameters())
    refine_params = list(model.refinement.parameters())

    param_groups = [
        {'params': attn_head_params, 'lr': ATTN_LR,   'weight_decay': ATTN_WD},
        {'params': refine_params,    'lr': REFINE_LR,  'weight_decay': REFINE_WD},
    ]
    if enc_trainable:
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        param_groups.append({'params': enc_params, 'lr': ENC_LR, 'weight_decay': ENC_WD})
        print(f"  Encoder fine-tune param group: {len(enc_params)} tensors @ "
              f"lr={ENC_LR}, wd={ENC_WD}")
    optimizer = optim.AdamW(param_groups)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Weight EMA over the trainable decoder modules (eval/save the EMA weights).
    ema = None
    if ema_decay and ema_decay > 0:
        ema_mods = [model.attention, model.point_head, model.refinement]
        if use_dist:
            ema_mods.append(model.dist_head)
        if mean_residual:
            ema_mods.append(model.mean_head)
        if global_head:
            ema_mods.append(model.global_spatial)
        ema = WeightEMA(ema_mods, decay=ema_decay)
        print(f"✓ Weight EMA ON (decay={ema_decay})")

    best_mae    = float('inf')
    best_corr   = 0.0
    best_slope  = 0.0
    best_score  = float('inf')
    patience    = 0

    # Live diagnostics / self-stopping state.
    history    = {'epoch': [], 'val_mae': [], 'val_slope': [],
                  'val_bias': [], 'val_floor': []}
    best_extra = None
    stop_diag  = None

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        epoch_mae_sum = 0.0
        epoch_mae_n   = 0

        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            pred = model(imgs, laterality=lat, average_multi=False)

            attn_w = model._last_attn_weights if hasattr(model, '_last_attn_weights') else None

            dist_logits = model._last_dist_logits if use_dist else None

            loss, mae, nv = compute_loss(pred, hvf, lat, epoch=epoch, attn_weights=attn_w,
                                         sector_weights=sector_weights,
                                         sector_combine=sector_combine,
                                         deep_cfg=deep_cfg,
                                         dist_logits=dist_logits, dist_cfg=dist_cfg,
                                         lds_weights=lds_weights, bias_penalty=bias_penalty,
                                         dispersion_weight=dispersion_weight)

            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update()
                epoch_mae_sum += mae * nv
                epoch_mae_n   += nv

            pbar.set_postfix({'MAE': f'{mae:.2f}'})

        epoch_train_mae = epoch_mae_sum / epoch_mae_n if epoch_mae_n > 0 else float('inf')
        scheduler.step()

        if epoch <= 5 or epoch % 10 == 0:
            ref_alpha = torch.sigmoid(model.refinement.alpha).item()
            temp = F.softplus(model.attention.temperature).item() + 0.5
            print(f"  Refinement alpha: {ref_alpha:.3f} | Attn temp: {temp:.3f}")

        # ── Validate ───────────────────────────────────────────
        is_scheduled = (epoch % VAL_EVERY == 0) or (epoch <= 3)
        # A "forced" check fires every FORCE_VAL_EVERY epochs regardless of the
        # gate so we still get periodic monitoring even if train hasn't converged.
        forced_check = (epoch % FORCE_VAL_EVERY == 0) and is_scheduled
        train_good   = epoch_train_mae < TRAIN_MAE_GATE
        run_val      = is_scheduled and (train_good or forced_check)

        if is_scheduled and not run_val:
            # Gate fired — skip the (expensive) val pass and say why.
            print(f"[Epoch {epoch}] train MAE={epoch_train_mae:.2f} dB "
                  f"(gate {TRAIN_MAE_GATE} not reached — skipping val)")

        elif run_val:
            # Evaluate (and save) the EMA weights, not the raw ones.
            if ema is not None:
                ema.apply_to()
            # ── Eval: fast cache path (frozen enc) or live path (fine-tune) ──
            if not enc_trainable:
                val_mae, val_corr, val_r2, val_slope, val_eye_corr, val_extra = \
                    evaluate_from_cache(model, val_cache, DEVICE, detailed=True)
                train_mae, train_corr, train_r2, train_slope, train_eye_corr = \
                    evaluate_from_cache(model, train_eval_cache, DEVICE)
            else:
                val_mae, val_corr, val_r2, val_slope, val_eye_corr, val_extra = \
                    evaluate(model, val_loader, detailed=True)
                train_mae, train_corr, train_r2, train_slope, train_eye_corr = \
                    evaluate(model, train_eval_pre)
            gap = train_mae - val_mae

            val_bias  = val_extra.get('bias', float('nan'))
            val_floor = val_extra.get('floor_mae', float('nan'))
            val_deep  = val_extra.get('deep_mae', float('nan'))
            val_fn    = val_extra.get('floor_n', 0)

            history['epoch'].append(epoch)
            history['val_mae'].append(val_mae)
            history['val_slope'].append(val_slope)
            history['val_bias'].append(val_bias)
            history['val_floor'].append(val_floor)

            tag = " [FORCED]" if forced_check and not train_good else ""
            print(f"\n[Epoch {epoch}{tag}]  batch train MAE={epoch_train_mae:.2f} dB")
            print(f"  Train(clean): MAE={train_mae:.2f} | Corr={train_corr:.3f} | Slope={train_slope:.3f} | EyeCorr={train_eye_corr:.3f}")
            print(f"  Val:          MAE={val_mae:.2f} | Corr={val_corr:.3f} | Slope={val_slope:.3f} | EyeCorr={val_eye_corr:.3f}")
            print(f"  Severe watch: bias={val_bias:+.2f} dB | floor(0-10)={val_floor:.2f} (n={val_fn}) | "
                  f"deep(<16)={val_deep:.2f}   [targets: bias→0, floor↓, deep↓]")
            print(f"  Gap: {gap:+.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if gap < -0.4:
                print(f"  ⚠️  Gap < -0.4 — possible overfitting")

            should_stop, stop_reason, warns = diagnose_training(history, target_mae=target_mae)
            for w in warns:
                print(f"  ⚠️  {w}")

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
                best_extra = {'epoch': epoch, 'mae': val_mae, 'slope': val_slope,
                              'bias': val_bias, 'floor_mae': val_floor,
                              'deep_mae': val_deep}
                torch.save({'model': model.state_dict(), 'mae': val_mae,
                            'corr': val_corr, 'r2': val_r2, 'slope': val_slope,
                            'eye_corr': val_eye_corr, 'epoch': epoch,
                            'use_dist': use_dist, 'dist_blend': dist_blend,
                            'mean_residual': mean_residual, 'global_head': global_head}, out_best)
                torch.save({'model_state_dict': model.state_dict(),
                            'encoder_checkpoint': CHECKPOINT_PATH,
                            'val_mae': val_mae, 'val_corr': val_corr,
                            'use_dist': use_dist, 'dist_blend': dist_blend,
                            'mean_residual': mean_residual, 'global_head': global_head}, out_inference)
                print(f"  ✓ Saved! ({', '.join(save_reason)})")
                print(f"    MAE={val_mae:.2f} | Slope={val_slope:.3f} | EyeCorr={val_eye_corr:.3f} | Score={composite:.2f}")
                patience = 0
            else:
                # Only count patience on real (gated) checks, not forced monitors.
                if not forced_check or train_good:
                    patience += VAL_EVERY

            # Restore raw weights so training continues from them.
            if ema is not None:
                ema.restore()

            # ── Self-stopping ───────────────────────────────────
            if should_stop:
                print(f"\n{'='*60}")
                print(f"⛔ AUTO-STOP at epoch {epoch}")
                print(f"   {stop_reason}")
                print(f"{'='*60}")
                stop_diag = stop_reason
                break
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs — plateau).")
                stop_diag = (f"Plateau — no val improvement for {patience} epochs.")
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
    print(f"  Model saved to: {out_inference}")
    print("=" * 60)

    # ── Diagnosis / goal check (explains how the run went) ─────
    print(f"\n{'='*60}")
    print("DIAGNOSIS — what happened this run")
    print(f"{'='*60}")
    if stop_diag:
        print(f"  Stop reason: {stop_diag}")
    else:
        print(f"  Stop reason: reached the epoch budget ({epochs}) without a "
              f"diagnostic trip.")
    if history['val_mae']:
        def _arrow(a, b, lower_better=True):
            if not (np.isfinite(a) and np.isfinite(b)):
                return "?"
            better = (b < a) if lower_better else (b > a)
            return "✓ improved" if better else "✗ worse/flat"
        print(f"  Val MAE:    {history['val_mae'][0]:.2f} → best {best_mae:.2f} dB  "
              f"({_arrow(history['val_mae'][0], best_mae)})")
        print(f"  Val slope:  {history['val_slope'][0]:.3f} → {history['val_slope'][-1]:.3f}  "
              f"(target ↑; {_arrow(history['val_slope'][0], history['val_slope'][-1], lower_better=False)})")
        print(f"  Val bias:   {history['val_bias'][0]:+.2f} → {history['val_bias'][-1]:+.2f} dB  "
              f"(target → 0)")
        print(f"  Floor 0-10: {history['val_floor'][0]:.2f} → {history['val_floor'][-1]:.2f} dB  "
              f"(GOAL-1 metric, target ↓; {_arrow(history['val_floor'][0], history['val_floor'][-1])})")
    if best_extra:
        print(f"\n  Best checkpoint (epoch {best_extra['epoch']}): "
              f"MAE={best_extra['mae']:.2f} | slope={best_extra['slope']:.3f} | "
              f"bias={best_extra['bias']:+.2f} | floor(0-10)={best_extra['floor_mae']:.2f} | "
              f"deep(<16)={best_extra['deep_mae']:.2f}")
        b = best_extra['bias']
        if np.isfinite(b) and b > 1.0:
            print("  ▸ Bias still positive → scotomata still under-deepened. "
                  "Re-run with a higher --overpred-penalty (e.g. 0.9) or --floor-boost (e.g. 3.0).")
        elif np.isfinite(b) and b < -1.0:
            print("  ▸ Bias negative → over-deepening. Lower --overpred-penalty / --floor-boost.")
        else:
            print("  ▸ Bias near 0 → scotoma-depth calibration looks healthy.")
    print(f"  Next: evaluate vs baseline with "
          f"`python decoder/garway_heath_weighting.py --evaluate`")
    print("=" * 60)

    # ── Champion: keep the best-ever model + methods across all runs ───
    run_config = {
        'weighting': weighting, 'sector_combine': sector_combine,
        'head': 'distributional' if use_dist else 'scalar',
        'dist_blend': (dist_blend if use_dist else None),
        'dist_loss_weight': (dist_loss_weight if use_dist else None),
        'reweight': reweight, 'lds_sigma': (lds_sigma if reweight == 'lds' else None),
        'bias_penalty': bias_penalty, 'deep_cfg': deep_cfg,
        'attn_dropout': ATTN_DROPOUT, 'head_dropout': HEAD_DROPOUT,
        'attn_weight_decay': ATTN_WD, 'attn_lr': ATTN_LR,
        'ema_decay': ema_decay, 'label_noise': LABEL_NOISE_STD,
        'variance_weight': VARIANCE_WEIGHT, 'epochs': epochs,
        'disc_crop': disc_crop, 'unfreeze_blocks': unfreeze_blocks,
        'dispersion_weight': dispersion_weight, 'mean_residual': mean_residual,
        'global_head': global_head, 'entropy_weight': ATTN_ENTROPY_WEIGHT,
        'finetune_norm': finetune_norm, 'enc_lr': (ENC_LR if (finetune_norm or unfreeze_blocks) else None),
        'heavy_aug': heavy_aug,
    }
    if update_champion:
        _update_champion(best_mae, best_extra, run_config, out_inference)
    return best_mae, best_extra


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v10.2 PerPointVFModel (GRAPE)")
    parser.add_argument('--weighting', choices=['baseline', 'garway_heath'],
                        default='baseline',
                        help="'baseline' = original value-based weight only (default); "
                             "'garway_heath' = add the opt-in spatial sector weight.")
    parser.add_argument('--sector-combine', choices=['both', 'sector_only', 'value_only'],
                        default='both',
                        help="How the sector weight combines with the existing "
                             "value-based weight (only used with --weighting garway_heath).")
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help="Override number of epochs (e.g. small value for a smoke test).")
    parser.add_argument('--floor-boost', type=float, default=None,
                        help="GH only: extra additive loss weight on the deepest GT "
                             "points (<floor-db). 0 disables. Default from garway_heath_weighting.")
    parser.add_argument('--overpred-penalty', type=float, default=None,
                        help="GH only: extra Huber multiplier when OVER-predicting a "
                             "deep point (counters +bias). 0 disables. Default from module.")
    parser.add_argument('--floor-db', type=float, default=None,
                        help="GH only: GT threshold (dB) defining the 'deep floor' for "
                             "the boost/penalty. Default from module.")
    # ── Tier-1 improvements (see decoder/specs/severe_point_improvement_plan.md) ──
    parser.add_argument('--head', choices=['scalar', 'distributional'], default='scalar',
                        help="'scalar' = original Huber regression head (default); "
                             "'distributional' = M1: add a per-point dB-bin head trained "
                             "with soft-CE (attacks deep-point mean-collapse).")
    parser.add_argument('--dist-blend', type=float, default=DIST_BLEND,
                        help="M1: weight on E[dist] in the blended prediction "
                             "(0=scalar only, 1=dist only). Default %(default)s.")
    parser.add_argument('--dist-loss-weight', type=float, default=DIST_LOSS_WEIGHT,
                        help="M1: λ on the soft-CE term. Default %(default)s.")
    parser.add_argument('--reweight', choices=['value', 'lds'], default='value',
                        help="'value' = original value/severity weight (default); "
                             "'lds' = M2: capped inverse-density (LDS) weight from the "
                             "train distribution (disables floor_boost/overpred).")
    parser.add_argument('--lds-sigma', type=float, default=LDS_SIGMA_DB,
                        help="M2: Gaussian kernel σ (dB) for LDS smoothing. Default %(default)s.")
    parser.add_argument('--target-mae', type=float, default=SUBGOAL_TARGET_MAE,
                        help="Goal MAE for the goal-plateau self-stop. Default %(default)s.")
    parser.add_argument('--bias-penalty', type=float, default=BIAS_PENALTY_WEIGHT,
                        help="Per-eye mean-error penalty weight; pins the global level so "
                             "deep emphasis can't shift it down. 0 disables. Default %(default)s.")
    parser.add_argument('--dropout', type=float, default=None,
                        help="Override attention & head dropout (regularization sweep). "
                             "Default keeps the module constants (0.4).")
    parser.add_argument('--weight-decay', type=float, default=None,
                        help="Override attention/head weight decay. Default keeps 0.015.")
    parser.add_argument('--lr', type=float, default=None,
                        help="Override attention/head LR. Default keeps 5e-4 (v10.1 used 8e-4).")
    parser.add_argument('--ema-decay', type=float, default=EMA_DECAY,
                        help="Weight-EMA decay (eval/save the EMA weights). 0 disables. "
                             "Default %(default)s.")
    parser.add_argument('--disc-crop', action='store_true',
                        help="Add a fixed laterality-aware optic-disc zoom as an extra "
                             "per-eye view (prediction-averaged). Signal probe for severe points.")
    parser.add_argument('--unfreeze-blocks', type=int, default=0,
                        help="Fine-tune the last N RETFound encoder blocks (+norm) at a tiny "
                             "LR to raise the corr ceiling. 0=fully frozen (default). Disables "
                             "feature caching (slower val).")
    # ── Session-3: honest per-patient evaluation + non-destructive run tagging ──
    parser.add_argument('--train-json', default=None,
                        help="Override TRAIN_JSON (e.g. a per-patient CV fold). Default: leaky split.")
    parser.add_argument('--val-json', default=None,
                        help="Override VAL_JSON (e.g. a per-patient CV fold). Default: leaky split.")
    parser.add_argument('--out-tag', default=None,
                        help="Save checkpoints to results/auto/<tag>_{best,inference}.pth instead "
                             "of the default paths (keeps dev runs from clobbering anything).")
    parser.add_argument('--no-champion', action='store_true',
                        help="Skip the auto champion update (dev runs on a different eval protocol "
                             "must not be compared against the historical leaky champion).")
    parser.add_argument('--dispersion-weight', type=float, default=DISPERSION_WEIGHT,
                        help="Session-3 anti-shrinkage: weight on the per-eye within-field "
                             "σ_pred≈σ_true match. 0=off (default). Targets slope/floor (severe).")
    parser.add_argument('--label-noise', type=float, default=None,
                        help="Override LABEL_NOISE_STD (dB). Lower/0 to un-wash spatial signal.")
    parser.add_argument('--entropy-weight', type=float, default=None,
                        help="Override ATTN_ENTROPY_WEIGHT. Lower/0 lets attention localize "
                             "(less diffuse) — recovers within-eye spatial signal.")
    parser.add_argument('--mean-residual', action='store_true',
                        help="Session-3: CLS→eye-mean + zero-mean spatial residual head. "
                             "Decouples the solved severity signal from the within-eye spatial "
                             "pattern so the spatial head isn't drowned (MLP probe → eyeCorr 0.51).")
    parser.add_argument('--global-head', action='store_true',
                        help="Session-3: JOINT global-spatial head (mean-pool patches‖CLS → 52 "
                             "pattern) + mean head (severity). Reaches the global-feature eyeCorr "
                             "(0.51) a shared per-point head can't. GH stays on via sector loss weight.")
    parser.add_argument('--finetune-norm', action='store_true',
                        help="Session-3 Lever-2: BitFit encoder fine-tune (norm+bias only, ~50k "
                             "params) to push past the frozen-feature eyeCorr≈0.51 cap. Disables "
                             "caching (slower). Use with --heavy-aug + low --enc-lr to avoid overfit.")
    parser.add_argument('--enc-lr', type=float, default=None,
                        help="Encoder fine-tune LR (default 1e-5). BitFit can take ~1e-4.")
    parser.add_argument('--heavy-aug', action='store_true',
                        help="Stronger train augmentation (rot10 + mild RRC + ColorJitter) — "
                             "regularizes encoder fine-tuning on the small (208-eye) train set.")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Override batch size (default 32). Use 16 for encoder fine-tune "
                             "to fit the full-ViT-L backward graph in MPS memory.")
    args = parser.parse_args()

    # Per-patient split / tag overrides (applied before train()).
    if args.train_json: TRAIN_JSON = os.path.abspath(args.train_json)
    if args.val_json:   VAL_JSON   = os.path.abspath(args.val_json)
    if args.train_json or args.val_json:
        print(f"✓ Data override: TRAIN={os.path.basename(TRAIN_JSON)} VAL={os.path.basename(VAL_JSON)}")

    deep_cfg = None
    if args.weighting == 'garway_heath':
        # Save GH checkpoints alongside results, leaving the baseline ones intact.
        from garway_heath_weighting import RESULTS_DIR, GH_MODEL, deep_loss_config
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_best      = os.path.join(RESULTS_DIR, "best_model_gh.pth")
        out_inference = GH_MODEL
        deep_cfg = deep_loss_config(floor_db=args.floor_db,
                                    floor_boost=args.floor_boost,
                                    overpred_penalty=args.overpred_penalty)
    else:
        out_best, out_inference = BEST_SAVE, INFERENCE_SAVE

    if args.out_tag:
        auto_dir = os.path.join(CURRENT_DIR, "results", "auto")
        os.makedirs(auto_dir, exist_ok=True)
        out_best      = os.path.join(auto_dir, f"{args.out_tag}_best.pth")
        out_inference = os.path.join(auto_dir, f"{args.out_tag}_inference.pth")
        print(f"✓ Output tag: checkpoints → {out_best}")

    train(weighting=args.weighting, sector_combine=args.sector_combine,
          epochs=args.epochs, out_best=out_best, out_inference=out_inference,
          deep_cfg=deep_cfg,
          use_dist=(args.head == 'distributional'), dist_blend=args.dist_blend,
          dist_loss_weight=args.dist_loss_weight,
          reweight=args.reweight, lds_sigma=args.lds_sigma, target_mae=args.target_mae,
          bias_penalty=args.bias_penalty,
          attn_dropout=args.dropout, head_dropout=args.dropout,
          weight_decay=args.weight_decay, lr=args.lr, ema_decay=args.ema_decay,
          disc_crop=args.disc_crop, unfreeze_blocks=args.unfreeze_blocks,
          update_champion=(not args.no_champion),
          dispersion_weight=args.dispersion_weight,
          label_noise=args.label_noise, entropy_weight=args.entropy_weight,
          mean_residual=args.mean_residual, global_head=args.global_head,
          finetune_norm=args.finetune_norm, enc_lr=args.enc_lr, heavy_aug=args.heavy_aug,
          batch_size=args.batch_size)