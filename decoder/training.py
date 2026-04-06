"""
Training v2 — Simplified, principled approach
Target: sub-3.74 dB val MAE

Changes from v1:
  - Removed label smoothing (classification technique, not suited for regression)
  - Removed low-dB upweighting (distorts loss landscape with small dataset)
  - Removed TTA (±5° too narrow to meaningfully help; re-add if needed)
  - Lowered decoder LR to 2e-4 (was 5e-4; pretrained decoder needs gentle tuning)
  - Lowered decoder WD to 1e-3 (was 5e-3; less fighting with lower LR)
  - Validate every 3 epochs instead of 5 (catch best checkpoint earlier)
  - Added linear warmup (5 epochs) to avoid early thrashing
  - Progressive unfreezing: start 3 blocks, unfreeze to 6 at epoch 20
  - Gradient accumulation (effective batch 64) to reduce per-epoch noise
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
BATCH_SIZE    = 32
ACCUM_STEPS   = 2           # Effective batch = 64, reduces noise
EPOCHS        = 120
BASE_LR       = 5e-4        # Restored from v1 — 2e-4 was too slow (corr stuck 0.21)
WEIGHT_DECAY  = 5e-4
PATIENCE      = 40

MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE  = 0.3

# ── Technique 1: Partial encoder fine-tuning ───────────────────
NUM_ENCODER_BLOCKS_INIT = 3   # Start with 3 blocks
NUM_ENCODER_BLOCKS_FULL = 6   # Unfreeze to 6 at UNFREEZE_EPOCH
UNFREEZE_EPOCH          = 20  # Progressive unfreezing milestone
ENCODER_LR_SCALE        = 0.05

# ── Technique 2: Projection warm-start ─────────────────────────
PROJ_INIT_BIAS = 18.0

# ── Technique 3: Rotation augmentation (laterality-safe) ───────
ROTATION_DEG = 5

# ── Technique 4: Mild low-dB upweighting ──────────────────────
# Restored from v1 at reduced strength (1.5x vs 2.5x).
# v2 without it: corr stuck at 0.21 — model predicts near-mean for
# everything. The upweighting forces learning the normal-vs-damaged
# distinction which drives correlation.
LOW_DB_THRESHOLD = 10.0
LOW_VALUE_WEIGHT = 1.5       # Reduced from 2.5x — less distortion, still helps

# ── Decoder config ─────────────────────────────────────────────
DECODER_LR_SCALE = 0.4       # 0.4 × 5e-4 = 2e-4 (gentle on pretrained weights)
DECODER_WD_SCALE = 2.0       # 2 × 5e-4 = 1e-3 (mild regularisation)

# ── LR schedule ────────────────────────────────────────────────
WARMUP_EPOCHS = 2             # Shortened — 5 epochs wasted too much time at near-zero LR
VAL_EVERY     = 3             # Validate more often to catch best checkpoint

# Outlier clipping
OUTLIER_CLIP_RANGE = (0, 35)

# ── Banned techniques (DO NOT USE) ─────────────────────────────
# - LoRA           (too slow on MPS — 30 min/epoch)
# - Flips/affine   (laterality-unsafe / not in methodology)
# - MixUp          (destroys spatial VF structure)

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
else:
    print("⚠️  No pretrained decoder found — training decoder from scratch")

# ============== Augmentation ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(ROTATION_DEG),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============== Dataset ==============
class MultiImageDataset(Dataset):
    def __init__(self, json_path, fundus_dir, transform, mode='train'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.fundus_dir = fundus_dir
        self.transform  = transform
        self.mode       = mode

        self.samples = []
        for item in self.data:
            images     = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            hvf        = item['hvf']
            laterality = item.get('Laterality', 'OD').strip().upper()
            patient_id = item.get('PatientID', 0)

            if self.mode == 'train':
                for img_path in images:
                    self.samples.append({
                        'image': img_path, 'hvf': hvf,
                        'laterality': laterality, 'patient_id': patient_id
                    })
            else:
                self.samples.append({
                    'images': images, 'hvf': hvf,
                    'laterality': laterality, 'patient_id': patient_id
                })

        if self.mode == 'train':
            print(f"  Train: {len(self.data)} eyes → {len(self.samples)} images")
        else:
            print(f"  Val: {len(self.data)} eyes with {sum(len(s['images']) for s in self.samples)} images")

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
    def __init__(self, encoder, pretrained_state=None, num_blocks=3, dropout=0.3):
        super().__init__()
        self.encoder = encoder

        # Freeze all encoder params initially
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze last N blocks
        self._unfreeze_blocks(num_blocks)

        # Projection: 1024 → 512 → 256 → 52
        self.projection = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(256, 52)
        )
        # Kaiming init for hidden layers
        for m in list(self.projection.modules())[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Warm-start final layer
        final = self.projection[-1]
        nn.init.constant_(final.bias, PROJ_INIT_BIAS)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        print(f"✓ Projection warm-start: final bias = {PROJ_INIT_BIAS} dB")

        # Refinement decoder (pretrained)
        self.decoder = VFAutoDecoder(input_dim=52)
        if pretrained_state is not None:
            try:
                self.decoder.load_state_dict(pretrained_state, strict=True)
                print(f"✓ Pre-trained decoder loaded (trainable)")
            except Exception as e:
                print(f"⚠️  Decoder load failed: {e}")
        else:
            print("✓ Decoder training from scratch")

    def _unfreeze_blocks(self, num_blocks):
        """Unfreeze the last `num_blocks` encoder blocks."""
        if hasattr(self.encoder, 'blocks'):
            total = len(self.encoder.blocks)
            # First re-freeze all blocks
            for block in self.encoder.blocks:
                for p in block.parameters():
                    p.requires_grad = False
            # Then unfreeze last N
            for i in range(max(0, total - num_blocks), total):
                for p in self.encoder.blocks[i].parameters():
                    p.requires_grad = True
            print(f"✓ Unfrozen {num_blocks}/{total} encoder blocks")

    def forward(self, x, average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]   # CLS token
        pred = self.decoder(self.projection(latent))
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred


# ============== Loss ==============
def compute_loss(pred, target, laterality):
    """Huber loss with mild low-dB upweighting."""
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
        # target is 72-dim (full 8x9 grid) → select valid points to get 52
        target_valid = target[i][valid_idx]
        # pred is already 52-dim (model outputs only valid points)
        pred_valid   = pred[i]
        mask         = target_valid < MASKED_VALUE_THRESHOLD
        if mask.sum() == 0:
            continue

        pred_clean   = pred_valid[mask]
        target_clean = target_valid[mask]

        # Mild upweighting for low-dB points (scotoma preservation)
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
        # target is 72-dim → select valid to get 52; pred is already 52-dim
        target_i  = target[min(i, target.shape[0]-1)][valid_idx]
        pred_i    = pred[min(i, pred.shape[0]-1)]
        mask      = target_i < MASKED_VALUE_THRESHOLD
        if mask.sum() > 0:
            all_pred.extend(pred_i[mask].tolist())
            all_target.extend(target_i[mask].tolist())
    if len(all_pred) < 2:
        return 0.0
    return float(np.corrcoef(np.array(all_pred), np.array(all_target))[0, 1])


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
    corr = pearson_correlation(
        torch.tensor(np.stack(all_preds)),
        torch.tensor(np.stack(all_targets)),
        all_lats
    )
    return total_mae / n_valid, corr


# ============== LR helpers ==============
def get_lr_scale(epoch, warmup_epochs):
    """Linear warmup for `warmup_epochs`, then 1.0."""
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def set_lr(optimizer, base_lrs, scale):
    """Apply a multiplicative scale to each param group's base LR."""
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        pg['lr'] = blr * scale


# ============== Training ==============
def train():
    print("=" * 60)
    print("Training v2 — Simplified Recipe")
    print("=" * 60)
    print(f"\nActive techniques (5):")
    print(f"  1. Partial encoder fine-tune  ({NUM_ENCODER_BLOCKS_INIT}→{NUM_ENCODER_BLOCKS_FULL} blocks, progressive)")
    print(f"  2. Projection warm-start      (bias={PROJ_INIT_BIAS} dB)")
    print(f"  3. Rotation + color jitter     (±{ROTATION_DEG}°, mild color aug)")
    print(f"  4. Gradient accumulation       (effective batch={BATCH_SIZE * ACCUM_STEPS})")
    print(f"  5. Low-dB upweighting          ({LOW_VALUE_WEIGHT}x below {LOW_DB_THRESHOLD} dB)")
    print(f"\nLR: {BASE_LR:.1e} (projection) | {BASE_LR * ENCODER_LR_SCALE:.1e} (encoder) | {BASE_LR * DECODER_LR_SCALE:.1e} (decoder)")
    print(f"Warmup: {WARMUP_EPOCHS} epochs linear | Schedule: CosineAnnealingWarmRestarts")
    print(f"Validate every {VAL_EVERY} epochs | Patience: {PATIENCE}")
    print(f"\nBanned techniques:")
    print(f"  - LoRA           (too slow on MPS)")
    print(f"  - Flips/affine   (laterality-unsafe)")
    print(f"  - MixUp          (destroys spatial VF structure)")
    print(f"\nExpected behavior (cancel if off-track):")
    print(f"  Ep 1-5:   MAE 5-7, Corr ~0.2      (warmup phase, should steadily drop)")
    print(f"  Ep 6-15:  MAE 4-5, Corr 0.3-0.5   (main learning; CANCEL if Corr<0.3 at ep15)")
    print(f"  Ep 15-30: MAE 3.5-4.2, Corr 0.5+  (refinement; unfreeze at ep20)")
    print(f"  Ep 30-60: MAE <3.74, Corr 0.6+     (target zone; CANCEL if MAE stuck >4.0)")
    print(f"  Ep 60+:   plateau → early stop")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val')

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=val_collate_fn)

    # ── Model ──────────────────────────────────────────────────
    model = MultiImageModel(base_model, pretrained_decoder_state,
                            NUM_ENCODER_BLOCKS_INIT, DROPOUT_RATE)
    model.to(DEVICE)

    # ── Optimizer ──────────────────────────────────────────────
    optimizer = optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad],
         'lr': BASE_LR * ENCODER_LR_SCALE, 'weight_decay': WEIGHT_DECAY * 2},
        {'params': model.projection.parameters(),
         'lr': BASE_LR, 'weight_decay': WEIGHT_DECAY},
        {'params': model.decoder.parameters(),
         'lr': BASE_LR * DECODER_LR_SCALE, 'weight_decay': WEIGHT_DECAY * DECODER_WD_SCALE},
    ])

    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_mae  = float('inf')
    best_corr = 0.0
    patience  = 0
    unfrozen_more = False

    for epoch in range(1, EPOCHS + 1):

        # ── Warmup ─────────────────────────────────────────────
        if epoch <= WARMUP_EPOCHS:
            scale = get_lr_scale(epoch, WARMUP_EPOCHS)
            set_lr(optimizer, base_lrs, scale)

        # ── Progressive unfreezing ─────────────────────────────
        if epoch == UNFREEZE_EPOCH and not unfrozen_more:
            model._unfreeze_blocks(NUM_ENCODER_BLOCKS_FULL)
            # Add newly unfrozen params to optimizer
            unfrozen_params = []
            total = len(model.encoder.blocks)
            for i in range(total - NUM_ENCODER_BLOCKS_FULL, total - NUM_ENCODER_BLOCKS_INIT):
                unfrozen_params.extend(model.encoder.blocks[i].parameters())
            if unfrozen_params:
                optimizer.add_param_group({
                    'params': unfrozen_params,
                    'lr': BASE_LR * ENCODER_LR_SCALE * 0.5,  # Even more conservative for newly unfrozen
                    'weight_decay': WEIGHT_DECAY * 2,
                })
                base_lrs.append(BASE_LR * ENCODER_LR_SCALE * 0.5)
            unfrozen_more = True
            print(f"\n  → Progressive unfreeze: {NUM_ENCODER_BLOCKS_INIT} → {NUM_ENCODER_BLOCKS_FULL} blocks at epoch {epoch}")

        # ── Train ──────────────────────────────────────────────
        model.train()
        epoch_mae = 0
        epoch_n   = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for step, (imgs, hvf, lat) in enumerate(pbar, 1):
            imgs = imgs.to(DEVICE)
            pred = model(imgs, average_multi=False)
            loss, mae, nv = compute_loss(pred, hvf, lat)

            if nv > 0:
                # Scale loss for gradient accumulation
                (loss / ACCUM_STEPS).backward()
                epoch_mae += mae * nv
                epoch_n   += nv

            if step % ACCUM_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix({'MAE': f'{mae:.2f}'})

        # Step scheduler (only after warmup)
        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        # ── Validate ───────────────────────────────────────────
        if epoch % VAL_EVERY == 0 or epoch <= 5:
            val_mae, val_corr = evaluate(model, val_loader)

            # Compute train metrics for diagnostics
            train_eval_ds = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, val_transform, mode='val')
            train_eval_loader = DataLoader(train_eval_ds, batch_size=1, shuffle=False,
                                           num_workers=0, collate_fn=val_collate_fn)
            train_mae, train_corr = evaluate(model, train_eval_loader)
            gap = train_mae - val_mae

            print(f"\n[Epoch {epoch}]")
            print(f"  Train: {train_mae:.2f} dB | Corr: {train_corr:.3f}")
            print(f"  Val:   {val_mae:.2f} dB | Corr: {val_corr:.3f} | Gap: {gap:+.2f}")

            current_lrs = [f"{pg['lr']:.1e}" for pg in optimizer.param_groups]
            print(f"  LRs: {current_lrs}")

            # ── Early warning checks ──────────────────────────
            if epoch == 15 and val_corr < 0.35:
                print("  ⚠️  WARNING: Corr < 0.35 at epoch 15 — check warm-start / LR")
            if gap < -0.5:
                print("  ⚠️  WARNING: Val much worse than train — overfitting risk")

            # ── Checkpoint ─────────────────────────────────────
            if val_mae < best_mae:
                best_mae  = val_mae
                best_corr = val_corr
                torch.save({
                    'model': model.state_dict(),
                    'mae': val_mae, 'corr': val_corr, 'epoch': epoch
                }, BEST_SAVE)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'encoder_checkpoint': CHECKPOINT_PATH,
                    'val_mae': val_mae, 'val_corr': val_corr
                }, INFERENCE_SAVE)
                print(f"  ✓ New Best! (MAE={val_mae:.2f}, Corr={val_corr:.3f})")
                patience = 0
            else:
                patience += VAL_EVERY  # Increment by gap between validations
                if patience >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
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