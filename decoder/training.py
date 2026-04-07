"""
Training v5.1 — Two-phase training with frozen decoder

Phase 1 (ep 1-P): 3 encoder blocks + projection, frozen decoder
  → Establishes encoder→VF mapping (expect MAE ~4.2-4.3)
  → Stops when val MAE plateaus (patience_phase1 = 15 epochs)

Phase 2 (ep P-end): Unfreeze to 6 blocks, fresh OneCycleLR, lower max LR
  → More encoder capacity to learn fundus→VF features
  → Frozen decoder still prevents overfitting
  → Should push below 4.0 dB

Frozen decoder rationale (proven in v5):
  - Train-val gap stays tight (±0.3 vs -0.57 with trainable decoder)
  - Val correlation stable at 0.53 (vs diverging in v4)
  - Prevents the decoder from overfitting its VF priors away

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
EPOCHS       = 120
PATIENCE     = 30

MASKED_VALUE_THRESHOLD = 99.0
DROPOUT_RATE = 0.4

# Phase 1: 3 blocks, higher LR
PHASE1_ENCODER_BLOCKS = 3
PHASE1_BASE_LR        = 5e-4
PHASE1_ENCODER_LR     = 2.5e-5    # 0.05x
PHASE1_WEIGHT_DECAY   = 1e-3
PHASE1_PATIENCE       = 15        # Switch to phase 2 after 15 epochs no improvement

# Phase 2: 6 blocks, lower LR to not destroy what phase 1 learned
PHASE2_ENCODER_BLOCKS = 6
PHASE2_BASE_LR        = 2e-4      # Lower — fine-tuning, not retraining
PHASE2_ENCODER_LR     = 1e-5      # Very conservative for encoder
PHASE2_WEIGHT_DECAY   = 1e-3

# Projection warm-start
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

# Decoder: FROZEN
DECODER_FROZEN = True

# Validation
VAL_EVERY = 3

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
                 freeze_decoder=True):
        super().__init__()
        self.encoder = encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.unfreeze_blocks(num_blocks)

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(256, NUM_VALID_POINTS)
        )
        for m in list(self.projection.modules())[:-1]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final = self.projection[-1]
        nn.init.constant_(final.bias, PROJ_INIT_BIAS)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        print(f"✓ Projection warm-start: final bias = {PROJ_INIT_BIAS} dB")

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
                    print(f"✓ Pre-trained decoder loaded (FROZEN)")
                else:
                    print(f"✓ Pre-trained decoder loaded (trainable)")
            except Exception as e:
                print(f"⚠️  Decoder load failed: {e}")
        else:
            print("✓ Decoder from scratch (trainable)")

    def unfreeze_blocks(self, num_blocks):
        """Unfreeze last N encoder blocks."""
        if hasattr(self.encoder, 'blocks'):
            total = len(self.encoder.blocks)
            for block in self.encoder.blocks:
                for p in block.parameters():
                    p.requires_grad = False
            for i in range(max(0, total - num_blocks), total):
                for p in self.encoder.blocks[i].parameters():
                    p.requires_grad = True
            print(f"✓ Unfrozen {num_blocks}/{total} encoder blocks")

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

    def print_trainable(self):
        enc  = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        proj = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
        dec  = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f"  Trainable: encoder={enc:,} + projection={proj:,} + decoder={dec:,} = {enc+proj+dec:,}")


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


# ============== Training ==============
def make_optimizer_and_scheduler(model, base_lr, encoder_lr, weight_decay,
                                  epochs_remaining, steps_per_epoch):
    """Create fresh optimizer + OneCycleLR for a training phase."""
    param_groups = [
        {'params': [p for p in model.encoder.parameters() if p.requires_grad],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': model.projection.parameters(),
         'lr': base_lr, 'weight_decay': weight_decay},
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[encoder_lr, base_lr],
        epochs=epochs_remaining,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100,
    )
    return optimizer, scheduler


def train():
    print("=" * 60)
    print("Training v5.1 — Two-phase with frozen decoder")
    print("=" * 60)

    print(f"\nPhase 1: {PHASE1_ENCODER_BLOCKS} encoder blocks")
    print(f"  LR: proj={PHASE1_BASE_LR:.1e} enc={PHASE1_ENCODER_LR:.1e}")
    print(f"  Plateau patience: {PHASE1_PATIENCE} epochs → switch to Phase 2")

    print(f"\nPhase 2: {PHASE2_ENCODER_BLOCKS} encoder blocks (progressive unfreeze)")
    print(f"  LR: proj={PHASE2_BASE_LR:.1e} enc={PHASE2_ENCODER_LR:.1e}")
    print(f"  Fresh optimizer — gives new blocks room to learn")

    print(f"\nShared: decoder=FROZEN | dropout={DROPOUT_RATE} | WD={PHASE1_WEIGHT_DECAY:.1e}")
    print(f"  Loss: Huber(δ={HUBER_DELTA}) + {LOW_VALUE_WEIGHT}x <{LOW_DB_THRESHOLD}dB")
    print(f"  Aug: ±{ROTATION_DEG}° | TTA: {TTA_ROTATIONS}")
    print(f"\nBanned: LoRA, flips/affine, MixUp")

    print(f"\nExpected:")
    print(f"  Phase 1 ep 1-15:  MAE 5→4.3  (same as v5)")
    print(f"  Phase 1 ep 15-30: MAE ~4.2   (plateau → triggers Phase 2)")
    print(f"  Phase 2 ep 1-20:  MAE 4.2→3.8 (new encoder capacity)")
    print(f"  Phase 2 ep 20+:   MAE <3.74   (target; CANCEL if stuck >4.0)")

    # ── Data ───────────────────────────────────────────────────
    train_dataset = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, train_transform, mode='train')
    val_dataset   = MultiImageDataset(VAL_JSON,   FUNDUS_DIR, val_transform,   mode='val', use_tta=USE_TTA)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=val_collate_fn)

    steps_per_epoch = len(train_loader)

    # ── Model ──────────────────────────────────────────────────
    model = MultiImageModel(base_model, pretrained_decoder_state,
                            PHASE1_ENCODER_BLOCKS, DROPOUT_RATE,
                            freeze_decoder=DECODER_FROZEN)
    model.to(DEVICE)
    model.print_trainable()

    # ── Phase 1 setup ──────────────────────────────────────────
    phase = 1
    phase1_epochs = 60  # Max epochs for phase 1
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, PHASE1_BASE_LR, PHASE1_ENCODER_LR, PHASE1_WEIGHT_DECAY,
        phase1_epochs, steps_per_epoch
    )

    best_mae  = float('inf')
    best_corr = 0.0
    phase_patience = 0
    global_patience = 0
    epoch = 0

    print(f"\n{'─'*40}")
    print(f"Phase 1: {PHASE1_ENCODER_BLOCKS} blocks, LR={PHASE1_BASE_LR:.1e}")
    print(f"{'─'*40}")

    for ep in range(1, EPOCHS + 1):
        epoch = ep
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS} [P{phase}]")

        for imgs, hvf, lat in pbar:
            imgs = imgs.to(DEVICE)
            pred = model(imgs, average_multi=False)
            loss, mae, nv = compute_loss(pred, hvf, lat)

            if nv > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                try:
                    scheduler.step()
                except Exception:
                    pass  # OneCycleLR may raise if past total_steps

            pbar.set_postfix({'MAE': f'{mae:.2f}', 'P': phase})

        # ── Validate ───────────────────────────────────────────
        if ep % VAL_EVERY == 0 or ep <= 3:
            val_mae, val_corr = evaluate(model, val_loader)

            train_eval = MultiImageDataset(TRAIN_JSON, FUNDUS_DIR, val_transform,
                                           mode='val', use_tta=False)
            train_eval_loader = DataLoader(train_eval, batch_size=1, shuffle=False,
                                           num_workers=0, collate_fn=val_collate_fn)
            train_mae, train_corr = evaluate(model, train_eval_loader)
            gap = train_mae - val_mae

            print(f"\n[Epoch {ep} | Phase {phase}]")
            print(f"  Train: {train_mae:.2f} dB | Corr: {train_corr:.3f}")
            print(f"  Val:   {val_mae:.2f} dB | Corr: {val_corr:.3f} | Gap: {gap:+.2f}")

            if val_mae < best_mae:
                best_mae  = val_mae
                best_corr = val_corr
                torch.save({'model': model.state_dict(), 'mae': val_mae,
                            'corr': val_corr, 'epoch': ep, 'phase': phase}, BEST_SAVE)
                torch.save({'model_state_dict': model.state_dict(),
                            'encoder_checkpoint': CHECKPOINT_PATH,
                            'val_mae': val_mae, 'val_corr': val_corr}, INFERENCE_SAVE)
                print(f"  ✓ New Best! (MAE={val_mae:.2f}, Corr={val_corr:.3f})")
                phase_patience = 0
                global_patience = 0
            else:
                phase_patience += VAL_EVERY
                global_patience += VAL_EVERY

                # Phase 1 → Phase 2 transition
                if phase == 1 and phase_patience >= PHASE1_PATIENCE:
                    print(f"\n{'─'*40}")
                    print(f"Phase 1 plateaued at MAE={best_mae:.2f}")
                    print(f"Switching to Phase 2: {PHASE2_ENCODER_BLOCKS} blocks")
                    print(f"{'─'*40}")

                    phase = 2
                    phase_patience = 0
                    model.unfreeze_blocks(PHASE2_ENCODER_BLOCKS)
                    model.print_trainable()

                    remaining = EPOCHS - ep
                    optimizer, scheduler = make_optimizer_and_scheduler(
                        model, PHASE2_BASE_LR, PHASE2_ENCODER_LR, PHASE2_WEIGHT_DECAY,
                        remaining, steps_per_epoch
                    )
                    continue

                # Global early stopping
                if global_patience >= PATIENCE:
                    print(f"\nEarly stopping at epoch {ep}")
                    break

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val MAE:  {best_mae:.2f} dB")
    print(f"  Best Val Corr: {best_corr:.3f}")
    print(f"  Reached in phase {phase}")
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