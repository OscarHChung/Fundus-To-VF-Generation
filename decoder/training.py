"""
Training v8 — Per-region VF prediction from spatial patch features

Key insight: All previous versions predicted 52 VF points from a single
global feature vector (CLS token or CLS+avg pool). This ignores the
spatial correspondence between retinal regions and VF sectors.

The VF has a known anatomical mapping to retinal regions:
  - Superior retina → inferior VF
  - Inferior retina → superior VF  
  - Nasal retina → temporal VF
  - Temporal retina → nasal VF

Instead of global→all_points, we:
  1. Keep the 14×14 = 196 spatial patch tokens from the encoder
  2. Pool patches into 4 quadrants matching VF anatomy
  3. Each quadrant head predicts its corresponding VF sector points
  4. Concatenate sector predictions → full 52-point VF
  5. Pass through frozen decoder for refinement

This means each VF region is predicted from the retinal patches that
actually correspond to it anatomically, rather than from a global summary.

Encoder: FULLY FROZEN (features are good enough — proven by v5-v7)
Decoder: FROZEN (prevents overfitting — proven by v5)
Only the 4 sector heads + a small fusion layer are trained.

Banned: LoRA, flips/affine, MixUp
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
BASE_LR      = 1e-3           # Higher LR — small trainable model
WEIGHT_DECAY = 5e-3
PATIENCE     = 30

MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.3

# Projection
PROJ_INIT_BIAS = 18.0

# Augmentation
ROTATION_DEG = 5

# TTA
USE_TTA       = True
TTA_ROTATIONS = [-5, 0, 5]

# Loss
LOW_DB_THRESHOLD = 10.0
LOW_VALUE_WEIGHT = 1.5
HUBER_DELTA      = 1.0

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
NUM_VALID_POINTS = len(valid_indices_od)  # 52

# ==============================================================
# VF sector definitions — which of the 52 valid points belong to
# each anatomical sector. The 8×9 grid rows 0-3 are superior VF
# (maps to inferior retina), rows 4-7 are inferior VF (maps to
# superior retina). Left/right split at column ~4.
#
# We define 4 sectors: superior-nasal, superior-temporal,
# inferior-nasal, inferior-temporal (from VF perspective).
# For OD: nasal = left cols (0-4), temporal = right cols (5-8)
# For OS: mirrored
# ==============================================================
def build_sector_indices():
    """Build sector-to-valid-index mapping for the 52-point VF."""
    grid = mask_OD.copy()  # 8×9
    rows, cols = grid.shape

    # Sector assignments for each grid position
    # Superior VF = rows 0-3, Inferior VF = rows 4-7
    # For OD: Nasal = cols 0-4, Temporal = cols 5-8
    sectors = {
        'sup_nasal': [],     # Superior VF, nasal side
        'sup_temporal': [],  # Superior VF, temporal side
        'inf_nasal': [],     # Inferior VF, nasal side
        'inf_temporal': [],  # Inferior VF, temporal side
    }

    valid_count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c]:
                is_superior = r < 4
                is_nasal = c <= 4  # For OD; OS will be handled at forward time

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

# Patch quadrant mapping for 14×14 grid
# The fundus image has INVERTED mapping to VF:
#   Superior retina (top patches) → Inferior VF
#   Inferior retina (bottom patches) → Superior VF
#   Nasal retina → Temporal VF (side depends on OD/OS)
#   Temporal retina → Nasal VF
#
# For OD fundus: nasal = right side of image, temporal = left
# So: left patches (temporal retina) → nasal VF
#     right patches (nasal retina) → temporal VF
#     top patches (superior retina) → inferior VF
#     bottom patches (inferior retina) → superior VF

def build_patch_quadrants():
    """Map 14×14 patch grid to 4 retinal quadrants."""
    quadrants = {
        # Retinal quadrant → maps to VF sector
        'sup_retina_nasal': [],     # → inf_temporal VF (OD)
        'sup_retina_temporal': [],  # → inf_nasal VF (OD)
        'inf_retina_nasal': [],     # → sup_temporal VF (OD)
        'inf_retina_temporal': [],  # → sup_nasal VF (OD)
    }

    for r in range(14):
        for c in range(14):
            patch_idx = r * 14 + c
            is_sup_retina = r < 7
            is_right = c >= 7  # Right side of image

            # For OD: right side of image = nasal retina
            if is_sup_retina and is_right:
                quadrants['sup_retina_nasal'].append(patch_idx)
            elif is_sup_retina and not is_right:
                quadrants['sup_retina_temporal'].append(patch_idx)
            elif not is_sup_retina and is_right:
                quadrants['inf_retina_nasal'].append(patch_idx)
            else:
                quadrants['inf_retina_temporal'].append(patch_idx)

    return quadrants

PATCH_QUADRANTS = build_patch_quadrants()

# For OD: retinal quadrant → VF sector mapping
# Superior retina nasal → Inferior VF temporal
# Superior retina temporal → Inferior VF nasal
# Inferior retina nasal → Superior VF temporal
# Inferior retina temporal → Superior VF nasal
RETINA_TO_VF_OD = {
    'sup_retina_nasal':    'inf_temporal',
    'sup_retina_temporal': 'inf_nasal',
    'inf_retina_nasal':    'sup_temporal',
    'inf_retina_temporal': 'sup_nasal',
}

# For OS: nasal/temporal are flipped
RETINA_TO_VF_OS = {
    'sup_retina_nasal':    'inf_nasal',      # "nasal" in image is actually temporal for OS
    'sup_retina_temporal': 'inf_temporal',
    'inf_retina_nasal':    'sup_nasal',
    'inf_retina_temporal': 'sup_temporal',
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
    """Predicts VF points for one sector from pooled patch features + CLS context."""
    def __init__(self, input_dim, num_points, dropout=0.3, bias_init=18.0):
        super().__init__()
        hidden = max(64, num_points * 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_points)
        )
        # Warm-start
        nn.init.constant_(self.net[-1].bias, bias_init)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.net(x)


# ============== Model ==============
class RegionalVFModel(nn.Module):
    def __init__(self, encoder, pretrained_decoder_state=None, dropout=0.3):
        super().__init__()
        self.encoder = encoder

        # Freeze entire encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("✓ Encoder: FULLY FROZEN")

        self.embed_dim = 1024  # RETFound ViT-Large

        # Register patch indices as buffers
        for quad_name, indices in PATCH_QUADRANTS.items():
            self.register_buffer(f'patch_idx_{quad_name}',
                                 torch.tensor(indices, dtype=torch.long))

        # Register VF sector indices as buffers
        for sec_name, indices in VF_SECTORS.items():
            self.register_buffer(f'vf_idx_{sec_name}',
                                 torch.tensor(indices, dtype=torch.long))

        # Each sector head gets: quadrant_pool(1024) + CLS(1024) = 2048
        sector_input_dim = self.embed_dim * 2
        self.sector_heads = nn.ModuleDict()
        for sec_name, sec_indices in VF_SECTORS.items():
            n_points = len(sec_indices)
            self.sector_heads[sec_name] = SectorHead(
                sector_input_dim, n_points, dropout, PROJ_INIT_BIAS
            )

        # Fusion: light refinement after concatenating sector outputs
        self.fusion = nn.Sequential(
            nn.Linear(NUM_VALID_POINTS, NUM_VALID_POINTS * 2),
            nn.LayerNorm(NUM_VALID_POINTS * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(NUM_VALID_POINTS * 2, NUM_VALID_POINTS),
        )
        # Residual weight for fusion
        self.fusion_alpha = nn.Parameter(torch.tensor(0.3))
        # Init fusion final to near-identity
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        # Decoder: frozen
        self.decoder = VFAutoDecoder(input_dim=NUM_VALID_POINTS)
        if pretrained_decoder_state is not None:
            try:
                self.decoder.load_state_dict(pretrained_decoder_state, strict=True)
                for p in self.decoder.parameters():
                    p.requires_grad = False
                print("✓ Decoder: FROZEN (pretrained)")
            except Exception as e:
                print(f"⚠️  Decoder load failed: {e}")

        # Count params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  Trainable: {trainable:,} | Frozen: {frozen:,}")

        # Print sector sizes
        for sec_name, sec_indices in VF_SECTORS.items():
            print(f"    {sec_name}: {len(sec_indices)} VF points")

    def extract_features(self, x):
        """Extract CLS token and patch tokens from frozen encoder."""
        with torch.no_grad():
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        # latent: (B, 197, 1024) = [CLS] + 196 patches
        cls_token = latent[:, 0, :]      # (B, 1024)
        patches = latent[:, 1:, :]       # (B, 196, 1024)
        return cls_token, patches

    def pool_quadrant(self, patches, quad_name):
        """Average pool patches belonging to a retinal quadrant."""
        idx = getattr(self, f'patch_idx_{quad_name}')  # (N_patches,)
        quad_patches = patches[:, idx, :]  # (B, N_patches, 1024)
        return quad_patches.mean(dim=1)    # (B, 1024)

    def forward(self, x, laterality='OD', average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")

        cls_token, patches = self.extract_features(x)  # (B, 1024), (B, 196, 1024)

        # Select mapping based on laterality
        if isinstance(laterality, str):
            mapping = RETINA_TO_VF_OD if laterality.startswith('OD') else RETINA_TO_VF_OS
        else:
            mapping = RETINA_TO_VF_OD if laterality[0].startswith('OD') else RETINA_TO_VF_OS

        # Predict each VF sector from its corresponding retinal quadrant
        sector_preds = {}
        for retina_quad, vf_sector in mapping.items():
            quad_feat = self.pool_quadrant(patches, retina_quad)  # (B, 1024)
            head_input = torch.cat([quad_feat, cls_token], dim=1)  # (B, 2048)
            sector_preds[vf_sector] = self.sector_heads[vf_sector](head_input)

        # Assemble full 52-point prediction
        pred = torch.zeros(x.shape[0], NUM_VALID_POINTS, device=x.device)
        for sec_name, sec_pred in sector_preds.items():
            idx = getattr(self, f'vf_idx_{sec_name}')
            pred[:, idx] = sec_pred

        # Fusion: light cross-sector refinement with residual
        fused = self.fusion(pred)
        pred = pred + self.fusion_alpha * fused

        # Frozen decoder refinement
        pred = self.decoder(pred)

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
            pred = model(imgs, laterality=lat, average_multi=True)
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


# ============== Training ==============
def train():
    print("=" * 60)
    print("Training v8 — Per-region VF prediction")
    print("=" * 60)

    print(f"\nArchitecture:")
    print(f"  Encoder: FULLY FROZEN → 196 spatial patches (14×14)")
    print(f"  4 retinal quadrants → pool → 4 sector heads → 52 VF points")
    print(f"  + Fusion layer for cross-sector refinement")
    print(f"  + Frozen decoder for VF prior")

    print(f"\nConfig:")
    print(f"  LR={BASE_LR:.1e} | WD={WEIGHT_DECAY:.1e} | dropout={DROPOUT_RATE}")
    print(f"  Loss: Huber + {LOW_VALUE_WEIGHT}x <{LOW_DB_THRESHOLD}dB")
    print(f"  Aug: ±{ROTATION_DEG}° | TTA: {TTA_ROTATIONS}")

    print(f"\nBanned: LoRA, flips/affine, MixUp")

    print(f"\nExpected:")
    print(f"  Ep 1-10:  MAE 5-6   (sector heads adapting)")
    print(f"  Ep 10-30: MAE 4-5   (spatial inductive bias should help)")
    print(f"  Ep 30-60: MAE <4.0  (target — spatial structure is the key)")
    print(f"  CANCEL if MAE >4.5 at ep 20 (spatial mapping wrong)")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val', use_tta=USE_TTA)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=val_collate_fn)

    # ── Model ──────────────────────────────────────────────────
    model = RegionalVFModel(base_model, pretrained_decoder_state, DROPOUT_RATE)
    model.to(DEVICE)

    # ── Optimizer: only sector heads + fusion ──────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-5
    )

    best_mae  = float('inf')
    best_corr = 0.0
    patience  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            pred = model(imgs, laterality=lat, average_multi=False)
            loss, mae, nv = compute_loss(pred, hvf, lat)

            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
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

            if val_mae < best_mae:
                best_mae  = val_mae
                best_corr = val_corr
                torch.save({'model': model.state_dict(), 'mae': val_mae,
                            'corr': val_corr, 'epoch': epoch}, BEST_SAVE)
                torch.save({'model_state_dict': model.state_dict(),
                            'encoder_checkpoint': CHECKPOINT_PATH,
                            'val_mae': val_mae, 'val_corr': val_corr}, INFERENCE_SAVE)
                print(f"  ✓ New Best! (MAE={val_mae:.2f}, Corr={val_corr:.3f})")
                patience = 0
            else:
                patience += VAL_EVERY
                if patience >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val MAE:  {best_mae:.2f} dB")
    print(f"  Best Val Corr: {best_corr:.3f}")
    gain = 3.74 - best_mae
    if best_mae < 3.0:
        print(f"  🎯 SUB-3 dB! Beat baseline by {gain:.2f} dB")
    elif best_mae < 3.74:
        print(f"  ✓ Beat baseline by {gain:.2f} dB")
    else:
        print(f"  Gap to baseline: {best_mae - 3.74:+.2f} dB")
    print(f"  Model saved to: {INFERENCE_SAVE}")
    print("=" * 60)


if __name__ == "__main__":
    train()