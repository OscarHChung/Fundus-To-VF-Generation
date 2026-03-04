"""
Visualize average per-location MAE heatmap for OD eyes.
Uses saved inference model - no retraining needed.

Usage:
    python visualize_mae_heatmap.py
    python visualize_mae_heatmap.py --split train   # use train set instead
    python visualize_mae_heatmap.py --no-tta        # disable TTA
"""

import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============== Paths (mirror training.py) ==============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
INFERENCE_SAVE   = os.path.join(CURRENT_DIR, "best_multi_image_model.pth")
BASE_DIR         = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR       = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON       = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON         = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
OUTPUT_DIR       = os.path.join(CURRENT_DIR, "mae_heatmaps")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== Config (mirror training.py) ==============
MASKED_VALUE_THRESHOLD = 99.0
OUTLIER_CLIP_RANGE     = (0, 35)
NUM_TTA_AUGS           = 4

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("⚠  Using CPU")

# ============== Mask (mirror training.py) ==============
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

# ============== Transforms (mirror training.py) ==============
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_tta_transforms():
    return [
        val_transform,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: transforms.functional.hflip(img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: transforms.functional.rotate(img, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: transforms.functional.rotate(img, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

# ============== Dataset (OD only) ==============
class ODValDataset(Dataset):
    def __init__(self, json_path, fundus_dir, use_tta=True):
        with open(json_path, 'r') as f:
            raw = json.load(f)

        self.fundus_dir = fundus_dir
        self.use_tta    = use_tta
        self.samples    = []

        for item in raw:
            lat = item.get('Laterality', 'OD').strip().upper()
            if not lat.startswith('OD'):
                continue
            images = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            self.samples.append({
                'images':    images,
                'hvf':       item['hvf'],
                'patient_id': item.get('PatientID', 0),
            })

        print(f"  OD samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        aug_images = []
        tta_list   = get_tta_transforms() if self.use_tta else [val_transform]

        for img_path in s['images']:
            img = Image.open(os.path.join(self.fundus_dir, img_path)).convert('RGB')
            for t in tta_list:
                aug_images.append(t(img))

        hvf    = np.array(s['hvf'], dtype=np.float32).flatten()
        images = torch.stack(aug_images)
        return images, torch.tensor(hvf)

# ============== Model (mirror training.py exactly) ==============
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

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

class MultiImageModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.21),
            nn.Linear(256, 52)
        )
        self.decoder = VFAutoDecoder(input_dim=52)

    def forward(self, x):
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        features = self.projection(latent)
        pred     = self.decoder(features)
        pred     = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred     = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        return pred.mean(dim=0, keepdim=True)   # average over TTA/multi-image

# ============== Load model ==============
def load_model():
    import argparse
    print("Loading RETFound encoder …")
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    encoder = mae_vit_large_patch16_dec512d8b()
    encoder.load_state_dict(ckpt['model'], strict=False)

    model = MultiImageModel(encoder)

    print(f"Loading inference checkpoint: {INFERENCE_SAVE}")
    inf_ckpt = torch.load(INFERENCE_SAVE, map_location='cpu', weights_only=False)
    state = inf_ckpt.get('model_state_dict', inf_ckpt.get('model'))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  ℹ  Ignored keys not in this model: {unexpected}")
    if missing:
        print(f"  ⚠  Missing keys (not loaded): {missing}")

    val_mae  = inf_ckpt.get('val_mae',  '?')
    val_corr = inf_ckpt.get('val_corr', '?')
    print(f"✓ Model loaded  |  Saved val MAE: {val_mae}  |  Corr: {val_corr}")

    model.to(DEVICE)
    model.eval()
    return model

# ============== Run inference & collect per-location errors ==============
def collect_errors(model, json_path, use_tta):
    dataset = ODValDataset(json_path, FUNDUS_DIR, use_tta=use_tta)

    # per-location accumulators over the 52 valid OD positions
    sum_ae  = np.zeros(52, dtype=np.float64)
    count   = np.zeros(52, dtype=np.int64)
    sum_gt  = np.zeros(52, dtype=np.float64)

    with torch.no_grad():
        for imgs, hvf in tqdm(dataset, desc="Inference"):
            imgs = imgs.to(DEVICE)
            pred = model(imgs)                         # (1, 52) — already averaged

            # Extract valid OD positions from ground truth
            hvf_flat   = hvf.numpy().flatten()         # length 72
            gt_valid   = hvf_flat[valid_indices_od]    # length 52
            pred_valid = pred.cpu().numpy().flatten()  # length 52

            # Only accumulate non-masked locations
            for loc in range(52):
                if gt_valid[loc] < MASKED_VALUE_THRESHOLD:
                    ae = abs(pred_valid[loc] - gt_valid[loc])
                    sum_ae[loc]  += ae
                    sum_gt[loc]  += gt_valid[loc]
                    count[loc]   += 1

    # Avoid div-by-zero
    with np.errstate(invalid='ignore'):
        mean_ae = np.where(count > 0, sum_ae / count, np.nan)
        mean_gt = np.where(count > 0, sum_gt / count, np.nan)

    overall_mae = np.nanmean(mean_ae)
    print(f"\nOverall OD MAE (per-location mean): {overall_mae:.3f} dB")
    return mean_ae, mean_gt, count

# ============== Map 52-vector back to 8×9 grid ==============
def vector_to_grid(vec_52):
    """Place 52 valid OD values back into an 8×9 grid; NaN elsewhere."""
    grid = np.full(72, np.nan)
    for k, flat_idx in enumerate(valid_indices_od):
        grid[flat_idx] = vec_52[k]
    return grid.reshape(8, 9)

# ============== Plot heatmap (matches heatmap visualizer style) ==============
def plot_heatmap(grid, title, filename, cmap='inferno', vmin=None, vmax=None,
                 cbar_label="MAE (dB)"):
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {filename}")

# ============== Side-by-side comparison ==============
def save_side_by_side(path1, path2, out_path, label1, label2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    h    = min(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * h / img1.height), h))
    img2 = img2.resize((int(img2.width * h / img2.height), h))
    combined = Image.new("RGBA", (img1.width + img2.width, h), (255,255,255,255))
    combined.paste(img1.convert("RGBA"), (0, 0))
    combined.paste(img2.convert("RGBA"), (img1.width, 0))
    combined.save(out_path)
    print(f"  Side-by-side saved → {out_path}")

# ============== Main ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'train'], default='val',
                        help='Which split to evaluate (default: val)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    args = parser.parse_args()

    json_path = VAL_JSON if args.split == 'val' else TRAIN_JSON
    use_tta   = not args.no_tta

    print(f"\n{'='*60}")
    print(f"OD MAE Heatmap  |  split={args.split}  |  TTA={'on' if use_tta else 'off'}")
    print(f"{'='*60}")

    model = load_model()

    mean_ae, mean_gt, count = collect_errors(model, json_path, use_tta)

    # Grids
    mae_grid = vector_to_grid(mean_ae)
    gt_grid  = vector_to_grid(mean_gt)

    tag = f"{args.split}_tta{NUM_TTA_AUGS if use_tta else 0}"

    # ── 1. MAE heatmap ──────────────────────────────────────────────
    mae_path = os.path.join(OUTPUT_DIR, f"OD_MAE_heatmap_{tag}.png")
    overall  = np.nanmean(mean_ae)
    plot_heatmap(
        mae_grid,
        title=f"OD | Average MAE per location | {args.split} set\n(mean MAE = {overall:.2f} dB)",
        filename=mae_path,
        cmap='inferno',
        vmin=0, vmax=10,
        cbar_label="MAE (dB)"
    )

    # ── 2. Ground-truth sensitivity heatmap ─────────────────────────
    gt_path = os.path.join(OUTPUT_DIR, f"OD_GT_sensitivity_{tag}.png")
    plot_heatmap(
        gt_grid,
        title=f"OD | Average GT sensitivity | {args.split} set",
        filename=gt_path,
        cmap='inferno',
        vmin=0, vmax=30,
        cbar_label="Sensitivity (dB)"
    )

    # ── 3. Side-by-side ─────────────────────────────────────────────
    combined_path = os.path.join(OUTPUT_DIR, f"OD_MAE_vs_GT_{tag}.png")
    save_side_by_side(mae_path, gt_path, combined_path,
                      label1="MAE", label2="GT sensitivity")

    # ── 4. Print per-location table ──────────────────────────────────
    print(f"\nPer-location summary (52 valid OD positions):")
    print(f"  {'Loc':>4}  {'MAE':>6}  {'GT':>6}  {'N':>5}")
    print(f"  {'-'*28}")
    for i, (ae, gt, n) in enumerate(zip(mean_ae, mean_gt, count)):
        print(f"  {i:4d}  {ae:6.2f}  {gt:6.2f}  {n:5d}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
