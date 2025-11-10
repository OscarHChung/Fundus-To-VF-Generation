"""
Advanced Fundus -> Visual Field (VF) training script
- Input: 1556 x 1556 fundus images
- Output: 54 VF points
- PyTorch + timm backbone + UNet-style decoder with cross-attention
- Loss: weighted MAE (L1) primary, SmoothL1 auxiliary
- Augmentations: strong (Flip, RandCrop/Resize, ColorJitter, Blur, Noise)
- Regularization: MixUp + CutMix, k-fold, ensemble averaging
- Mixed precision (AMP) + gradient accumulation for large imgs
- Save per-image predictions + MAE into JSON
"""

import os
import json
import math
import random
import argparse
from functools import partial
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

# requires: timm, torchvision
import timm
from torchvision import transforms

# -----------------------------
# Utility helpers
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# -----------------------------
# Dataset classes
# -----------------------------
class PairedVFDFDataset(Dataset):
    """GRAPE paired dataset: expects JSON list with fields:
       - FundusImage: relative path to image file
       - hvf: list of 54 floats (use 100.0 as masked value)
       - Laterality: 'OD' or 'OS'
       - id: optional
    """
    def __init__(self, json_path, fundus_root, transform=None):
        self.fundus_root = Path(fundus_root)
        with open(json_path, 'r') as f:
            self.entries = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        path = self.fundus_root / e['FundusImage']
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        vf = torch.tensor(np.array(e['hvf'], dtype=np.float32).flatten())
        laterality = e.get('Laterality', 'OD')
        img_id = e.get('id', os.path.basename(str(path)))
        return img, vf, laterality, img_id

# -----------------------------
# Masks / utility for VF shape
# -----------------------------
VF_POINTS = 54
MASK_VALUE = 100.0

# Example mask construction: here we'll treat all points as potentially valid;
# but the script supports a per-point boolean mask provided in the JSON if needed.
# If you have an OD/OS mask mapping (like earlier), plug it in here.
def make_default_mask():
    # default all True (valid), but still allow masked points (value==MASK_VALUE)
    return torch.ones(VF_POINTS, dtype=torch.bool)

DEFAULT_VF_MASK = make_default_mask()

def apply_laterality_mask(preds: torch.Tensor, laterality_list):
    # if you have a specific mask mapping for OD/OS you can implement here.
    # For now, we just return preds unchanged (no forced 100s).
    return preds

# -----------------------------
# Augmentations
# -----------------------------
def get_transforms(img_size=1556, train=True):
    if train:
        t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # small rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    return t

# -----------------------------
# MixUp and CutMix utilities
# -----------------------------
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, 1.0, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bw = int(W * math.sqrt(1 - lam))
    bh = int(H * math.sqrt(1 - lam))
    x1 = np.clip(cx - bw // 2, 0, W)
    x2 = np.clip(cx + bw // 2, 0, W)
    y1 = np.clip(cy - bh // 2, 0, H)
    y2 = np.clip(cy + bh // 2, 0, H)
    if x2 <= x1 or y2 <= y1:
        return x, y, None
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_eff = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, (index, lam_eff)

# -----------------------------
# Model: backbone + UNet-ish decoder with VF queries
# -----------------------------
class ImageEncoderMultiScale(nn.Module):
    """
    Uses timm backbone and returns multi-scale feature maps for decoder skip connections.
    We'll use a generic timm backbone that supports features_out or create hooks.
    """
    def __init__(self, backbone_name='resnetv2_50', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(1,2,3,4))
        # features_only returns a list of feature maps from earlier to later layers

    def forward(self, x):
        feats = self.backbone(x)
        # feats is list of tensors: [C1 H1 W1, C2 H2 W2, C3 H3 W3, C4 H4 W4]
        return feats

class VFDecoderUNet(nn.Module):
    """
    UNet-style decoder that takes multi-scale features and produces VF_POINTS output.
    Additionally we create a set of learnable VF query embeddings (one per point)
    and use cross-attention to map spatial features -> point predictions.
    """

    def __init__(self, feat_channels=[64, 128, 320, 512], vf_points=54, hidden_dim=256):
        super().__init__()
        self.vf_points = vf_points
        # simple conv up blocks
        self.up_convs = nn.ModuleList()
        in_ch = feat_channels[-1]
        for ch in reversed(feat_channels[:-1]):
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ))
            in_ch = ch
        # after ups, produce a spatial map
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1,1))  # global pool -> per-image vector
        )
        # VF queries: each VF point has an embedding that will query the image features
        self.vf_queries = nn.Parameter(torch.randn(vf_points, hidden_dim))
        # map queries to final scalar predictions
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)  # scalar dB per point
        )

        # small cross-attention: queries attend to flattened spatial features
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1)

    def forward(self, feats):
        # feats: list low->high [C1,H1,W1,...,C4,H4,W4]
        x = feats[-1]
        for i, up in enumerate(self.up_convs):
            x = up(x)
            # skip add
            skip = feats[-2 - i]
            # if shape mismatch, interpolate skip
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip  # simple residual fusion

        img_vec = self.final_conv(x).flatten(1)  # (B, hidden_dim)
        # expand and prepare for attention: create spatial tokens by repeating img_vec (simple)
        # to let queries attend to image representation we expand img_vec to tokens
        # Alternatively could flatten spatial map tokens for richer attention (left simple for speed)
        # Build key/value as img_vec repeated
        B = img_vec.size(0)
        hidden_dim = img_vec.size(1)
        # prepare queries: (T, B, E) for multihead attention
        queries = self.vf_queries.unsqueeze(1).repeat(1, B, 1)  # (T, B, E)
        keys = img_vec.unsqueeze(0).repeat(self.vf_points, 1, 1)  # (T, B, E)
        vals = keys
        attn_out, _ = self.attn(queries, keys, vals)  # (T, B, E)
        attn_out = attn_out.transpose(0,1)  # (B, T, E)
        preds = self.query_mlp(attn_out).squeeze(-1)  # (B, T)
        return preds

# -----------------------------
# Training and eval functions
# -----------------------------
def weighted_mae(preds, targets, mask_values=MASK_VALUE, weights=None):
    """preds, targets: (B, VF_POINTS)"""
    # mask invalid points where target == mask_values
    valid = (targets != mask_values)
    diff = torch.abs(preds - targets)
    if weights is None:
        weights = torch.ones_like(diff)
    weighted = diff * weights
    # only average over valid points
    weighted_valid = weighted * valid.float()
    denom = valid.float().sum(dim=1).clamp_min(1.0)  # avoid divide by zero
    per_sample = weighted_valid.sum(dim=1) / denom
    return per_sample.mean()

def compute_per_sample_mae_np(pred_np, label_np, mask_val=MASK_VALUE):
    valid = label_np != mask_val
    if valid.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(pred_np[valid] - label_np[valid])))

def train_one_epoch(model_enc, model_dec, loader, optim_dec, device, scaler, epoch,
                    mixup_prob=0.5, cutmix_prob=0.0,
                    weights_tensor=None, grad_accum_steps=1):
    model_enc.eval()  # backbone frozen mostly
    model_dec.train()
    running_loss = 0.0
    n = 0
    criterion_l1 = nn.L1Loss(reduction='none')  # to compute per-point for weighting
    criterion_smooth = nn.SmoothL1Loss()
    for step, (imgs, vfs, laterality, ids) in enumerate(tqdm(loader, desc=f"Train E{epoch}")):
        imgs = imgs.to(device)
        vfs = vfs.to(device)
        B = imgs.size(0)

        # Apply mixup / cutmix occasionally
        use_mixup = (random.random() < mixup_prob)
        use_cutmix = (random.random() < cutmix_prob)
        if use_cutmix:
            imgs, vfs, cutmix_meta = cutmix_data(imgs, vfs)
        elif use_mixup:
            imgs, y_a, y_b, lam, idx = mixup_data(imgs, vfs, alpha=0.3)
            y_a = y_a.to(device); y_b = y_b.to(device)

        with autocast():
            feats = model_enc(imgs)  # list of feature maps
            preds = model_dec(feats)  # (B, VF_POINTS)

            # If mixup: interpolate losses
            if use_mixup:
                loss_a = criterion_l1(preds, y_a).mean(dim=1)
                loss_b = criterion_l1(preds, y_b).mean(dim=1)
                loss_points = lam * loss_a + (1-lam) * loss_b
                loss = loss_points.mean()
            else:
                # weighted MAE across points
                if weights_tensor is not None:
                    # weights_tensor shape (VF_POINTS,)
                    per_point_l1 = criterion_l1(preds, vfs)  # (B, VF_POINTS)
                    weighted = per_point_l1 * weights_tensor.unsqueeze(0).to(device)
                    # mask invalid targets
                    mask = (vfs != MASK_VALUE).float()
                    loss = (weighted * mask).sum() / mask.sum().clamp_min(1.0)
                else:
                    loss = weighted_mae(preds, vfs, mask_values=MASK_VALUE)

            # small auxiliary SmoothL1 to stabilize
            aux = criterion_smooth(preds, torch.clamp(vfs, min=-10.0, max=40.0))
            loss = 0.9 * loss + 0.1 * aux.mean()

        scaler.scale(loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optim_dec)
            scaler.update()
            optim_dec.zero_grad()

        running_loss += float(loss.item()) * B
        n += B

    return running_loss / max(1, n)

@torch.no_grad()
def validate(model_enc, model_dec, loader, device, weights_tensor=None):
    model_enc.eval()
    model_dec.eval()
    total_loss = 0.0
    n = 0
    per_image_maes = []
    for imgs, vfs, laterality, ids in tqdm(loader, desc="Val"):
        imgs = imgs.to(device)
        vfs = vfs.to(device)
        feats = model_enc(imgs)
        preds = model_dec(feats)
        # apply laterality mask if available (not forcing 100)
        preds = apply_laterality_mask(preds, laterality)

        # compute per-sample MAE (weighted)
        if weights_tensor is not None:
            per_point_abs = torch.abs(preds - vfs) * weights_tensor.unsqueeze(0).to(device)
            mask = (vfs != MASK_VALUE).float()
            per_sample = (per_point_abs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            batch_loss = per_sample.mean()
        else:
            batch_loss = weighted_mae(preds, vfs, mask_values=MASK_VALUE)

        total_loss += float(batch_loss.item()) * imgs.size(0)
        n += imgs.size(0)

        # collect per-image MAE for JSON
        preds_np = preds.cpu().numpy()
        vfs_np = vfs.cpu().numpy()
        for i in range(preds_np.shape[0]):
            per_image_maes.append((ids[i], compute_per_sample_mae_np(preds_np[i], vfs_np[i])))

    avg_loss = total_loss / max(1, n)
    return avg_loss, per_image_maes

# -----------------------------
# Training orchestrator (k-fold)
# -----------------------------
def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load dataset
    train_transform = get_transforms(img_size=args.img_size, train=True)
    val_transform = get_transforms(img_size=args.img_size, train=False)
    dataset = PairedVFDFDataset(args.grape_json, args.fundus_dir, transform=train_transform)

    # optionally create a per-point weighting array emphasizing central / clinically important points:
    # Default: stronger weight in middle indices; tune as per your domain knowledge.
    weights = torch.ones(VF_POINTS, dtype=torch.float32)
    # Example: triangular weights peaking in middle
    mid = VF_POINTS // 2
    for i in range(VF_POINTS):
        weights[i] = 1.0 + 1.0 * (1 - abs(i - mid) / (VF_POINTS/2))

    # K-fold indices
    N = len(dataset)
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    folds = np.array_split(idxs, args.k_folds)

    fold_maes = []

    # Backbone encoder - pretrained strong model from timm
    model_enc = ImageEncoderMultiScale(backbone_name=args.backbone, pretrained=True).to(device)

    # Freeze encoder partially (fine tune later optionally)
    for param in model_enc.parameters():
        param.requires_grad = False
    # optionally unfreeze last stage
    # for p in list(model_enc.backbone.parameters())[-80:]:
    #     p.requires_grad = True

    for fold_idx in range(args.k_folds):
        print(f"\n--- Fold {fold_idx+1}/{args.k_folds} ---")
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([f for i,f in enumerate(folds) if i != fold_idx], axis=0)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset_raw = Subset(dataset, val_idx.tolist())
        # ensure val uses eval transforms
        val_subset = []
        for i in val_idx.tolist():
            # create a lightweight wrapper to use val_transform
            img, vf, laterality, img_id = dataset[i]
            # re-open original with val transform
            entry = dataset.entries[i]
            path = Path(args.fundus_dir) / entry['FundusImage']
            pil = Image.open(path).convert('RGB')
            img_t = val_transform(pil)
            val_subset.append((img_t, vf, laterality, img_id))
        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        # custom val loader wrapper
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Decoder for fold
        feat_channels = model_enc.backbone.feature_info.channels()  # try to get channels from timm features
        # fallback if not available
        try:
            feat_chs = [int(c) for c in feat_channels]
        except Exception:
            feat_chs = [64, 128, 320, 512]

        model_dec = VFDecoderUNet(feat_channels=feat_chs, vf_points=VF_POINTS, hidden_dim=256).to(device)

        # Optionally initialize decoder from previous pretrain checkpoint if you have one
        if args.pretrained_decoder and os.path.exists(args.pretrained_decoder):
            try:
                model_dec.load_state_dict(torch.load(args.pretrained_decoder, map_location=device))
                print("Loaded pretrained decoder:", args.pretrained_decoder)
            except Exception as e:
                print("Could not load pretrained decoder:", e)

        # Optimizer - only decoder parameters (encoder frozen)
        optimizer_dec = torch.optim.AdamW(model_dec.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dec, T_max=args.epochs)
        # Also monitor val -> reduce on plateau
        reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dec, mode='min', patience=3, factor=0.5, verbose=True)

        scaler = GradScaler()
        best_val = 1e9
        patience = args.patience
        patience_count = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model_enc, model_dec, train_loader, optimizer_dec, device, scaler, epoch,
                                         mixup_prob=args.mixup_prob, cutmix_prob=args.cutmix_prob,
                                         weights_tensor=weights if args.use_weights else None,
                                         grad_accum_steps=args.grad_accum)
            scheduler.step()
            val_loss, per_image_maes = validate(model_enc, model_dec, val_loader, device, weights_tensor=weights if args.use_weights else None)
            reduce_on_plateau.step(val_loss)
            print(f"Fold {fold_idx+1} Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            # save best
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model_dec.state_dict(), f"best_decoder_fold{fold_idx+1}.pth")
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print("Early stopping triggered.")
                    break

        # after fold training -> load best and evaluate on val set thoroughly
        model_dec.load_state_dict(torch.load(f"best_decoder_fold{fold_idx+1}.pth"))
        _, per_image_maes_val = validate(model_enc, model_dec, val_loader, device, weights_tensor=weights if args.use_weights else None)
        fold_mae = np.nanmean([m for (_,m) in per_image_maes_val if not math.isnan(m)])
        print(f"Fold {fold_idx+1} MAE: {fold_mae:.4f} dB")
        fold_maes.append(fold_mae)

    avg_mae = float(np.mean(fold_maes))
    print(f"\n=== Cross-Validation Summary ===")
    for i,m in enumerate(fold_maes):
        print(f" Fold {i+1}: MAE = {m:.4f} dB")
    print("Average MAE across folds:", avg_mae)
    return avg_mae

# -----------------------------
# CLI and main
# -----------------------------
if __name__ == "__main__":
    import torch

    # ==== Configuration ====
    config = {
        "grape_json": "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data/vf_tests/grape_new_vf_tests.json",
        "fundus_dir": "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data/fundus/grape_fundus_images/",
        "img_size": 1556,
        "batch_size": 2,
        "epochs": 30,
        "lr": 3e-4,
        "k_folds": 5,
        "backbone": "efficientnet_b3",
        "use_weights": True,
        "mixup_prob": 0.1,
        "cutmix_prob": 0.1,
        "patience": 7,
        "grad_accum": 2,
    }

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔥 Using device: {device}")

    print("\n🚀 Starting Fundus-to-VF training pipeline...\n")

    try:
        args = argparse.Namespace(**config)
        run_training(args)
        print("\n✅ Training completed successfully.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}\n")
