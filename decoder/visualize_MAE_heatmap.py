"""
Visualize average per-location MAE heatmap for OD and OS eyes side-by-side.
Uses saved inference model — no retraining needed.

Usage:
    python visualize_mae_heatmap.py                 # val set + TTA
    python visualize_mae_heatmap.py --split train
    python visualize_mae_heatmap.py --no-tta
"""

import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ============== Paths ==============
CURRENT_DIR     = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR    = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
INFERENCE_SAVE  = os.path.join(CURRENT_DIR, "inference_model.pth")
BASE_DIR        = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR      = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON      = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")
VAL_JSON        = os.path.join(BASE_DIR, "data", "vf_tests", "grape_test.json")
OUTPUT_DIR      = os.path.join(CURRENT_DIR, "mae_heatmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== Config ==============
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

# ============== Valid index masks (mirror training.py) ==============
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
valid_indices_os = list(reversed(valid_indices_od))

# ============== Transforms ==============
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

# ============== Dataset (both eyes) ==============
class BilateralValDataset:
    def __init__(self, json_path, fundus_dir, use_tta=True):
        with open(json_path, 'r') as f:
            raw = json.load(f)
        self.fundus_dir = fundus_dir
        self.use_tta    = use_tta
        self.samples    = []
        for item in raw:
            lat    = item.get('Laterality', 'OD').strip().upper()
            images = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            self.samples.append({
                'images':     images,
                'hvf':        item['hvf'],
                'laterality': lat,
                'patient_id': item.get('PatientID', 0),
            })
        od  = sum(1 for s in self.samples if s['laterality'].startswith('OD'))
        os_ = sum(1 for s in self.samples if s['laterality'].startswith('OS'))
        print(f"  Total samples: {len(self.samples)}  (OD: {od}, OS: {os_})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s        = self.samples[idx]
        tta_list = get_tta_transforms() if self.use_tta else [val_transform]
        aug_imgs = []
        for img_path in s['images']:
            img = Image.open(os.path.join(self.fundus_dir, img_path)).convert('RGB')
            for t in tta_list:
                aug_imgs.append(t(img))
        imgs = torch.stack(aug_imgs)
        hvf  = np.array(s['hvf'], dtype=np.float32).flatten()
        return imgs, torch.tensor(hvf), s['laterality']

# ============== Model (mirror training.py) ==============
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
        self.encoder    = encoder
        self.projection = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.21),
            nn.Linear(256, 52)
        )
        self.decoder = VFAutoDecoder(input_dim=52)

    def forward(self, x):
        latent  = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        features = self.projection(latent)
        pred     = self.decoder(features)
        pred     = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred     = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        return pred.mean(dim=0, keepdim=True)  # average TTA/multi-image

# ============== Load model ==============
def load_model():
    print("Loading RETFound encoder …")
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    encoder = mae_vit_large_patch16_dec512d8b()
    encoder.load_state_dict(ckpt['model'], strict=False)

    model    = MultiImageModel(encoder)
    inf_ckpt = torch.load(INFERENCE_SAVE, map_location='cpu', weights_only=False)
    state    = inf_ckpt.get('model_state_dict', inf_ckpt.get('model'))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  ℹ  Ignored keys: {unexpected}")
    if missing:
        print(f"  ⚠  Missing keys: {missing}")

    val_mae  = inf_ckpt.get('val_mae',  '?')
    val_corr = inf_ckpt.get('val_corr', '?')
    print(f"✓ Model loaded  |  Saved val MAE: {val_mae}  |  Corr: {val_corr}")
    model.to(DEVICE)
    model.eval()
    return model

# ============== Inference — collect per-location errors for both eyes ==============
def collect_errors_bilateral(model, json_path, use_tta):
    dataset = BilateralValDataset(json_path, FUNDUS_DIR, use_tta=use_tta)

    acc = {
        'OD': {'sum_ae': np.zeros(52), 'sum_gt': np.zeros(52), 'count': np.zeros(52, dtype=int)},
        'OS': {'sum_ae': np.zeros(52), 'sum_gt': np.zeros(52), 'count': np.zeros(52, dtype=int)},
    }

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            imgs, hvf, lat = dataset[idx]
            imgs = imgs.to(DEVICE)
            pred = model(imgs)  # (1, 52)

            hvf_flat  = hvf.numpy().flatten()         # length 72
            pred_flat = pred.cpu().numpy().flatten()  # length 52

            eye       = 'OD' if lat.startswith('OD') else 'OS'
            valid_idx = valid_indices_od if eye == 'OD' else valid_indices_os
            gt_valid  = hvf_flat[valid_idx]  # 52 values in eye-specific order

            for loc in range(52):
                if gt_valid[loc] < MASKED_VALUE_THRESHOLD:
                    acc[eye]['sum_ae'][loc] += abs(pred_flat[loc] - gt_valid[loc])
                    acc[eye]['sum_gt'][loc] += gt_valid[loc]
                    acc[eye]['count'][loc]  += 1

    results = {}
    for eye, a in acc.items():
        with np.errstate(invalid='ignore'):
            mean_ae = np.where(a['count'] > 0, a['sum_ae'] / a['count'], np.nan)
            mean_gt = np.where(a['count'] > 0, a['sum_gt'] / a['count'], np.nan)
        n       = int(a['count'].max()) if a['count'].max() > 0 else 0
        overall = np.nanmean(mean_ae)
        print(f"  {eye}: {n} samples  |  Overall MAE: {overall:.3f} dB")
        results[eye] = {'mean_ae': mean_ae, 'mean_gt': mean_gt, 'count': a['count']}

    return results

# ============== Map 52-vector → 8×9 display grid ==============
def vector_to_grid(vec_52, eye='OD'):
    """
    Place 52 valid values into an 8×9 grid.
    OS values are stored in reversed OD index order (valid_indices_os).
    We flip OS horizontally so nasal/temporal sides display correctly.
    """
    grid      = np.full(72, np.nan)
    valid_idx = valid_indices_od if eye == 'OD' else valid_indices_os
    for k, flat_idx in enumerate(valid_idx):
        grid[flat_idx] = vec_52[k]
    grid = grid.reshape(8, 9)
    if eye == 'OS':
        grid = np.fliplr(grid)
    return grid

# ============== Plot single panel ==============
def plot_single(ax, grid, title, cmap_name, vmin, vmax, cbar_label):
    cmap_obj = matplotlib.colormaps[cmap_name].copy()
    cmap_obj.set_bad(color='white')
    im = ax.imshow(grid, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

# ============== Build 4-panel figure ==============
def save_four_panel(results, tag, split):
    od_mae_grid = vector_to_grid(results['OD']['mean_ae'], 'OD')
    os_mae_grid = vector_to_grid(results['OS']['mean_ae'], 'OS')
    od_gt_grid  = vector_to_grid(results['OD']['mean_gt'], 'OD')
    os_gt_grid  = vector_to_grid(results['OS']['mean_gt'], 'OS')

    od_mae_val = np.nanmean(results['OD']['mean_ae'])
    os_mae_val = np.nanmean(results['OS']['mean_ae'])
    combined   = np.nanmean(np.concatenate([
        results['OD']['mean_ae'][~np.isnan(results['OD']['mean_ae'])],
        results['OS']['mean_ae'][~np.isnan(results['OS']['mean_ae'])]
    ]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Visual Field Prediction — MAE & GT Sensitivity  |  {split} set\n"
        f"OD MAE: {od_mae_val:.2f} dB   |   OS MAE: {os_mae_val:.2f} dB   |   Combined: {combined:.2f} dB",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Row 0: MAE heatmaps
    plot_single(axes[0, 0], od_mae_grid,
                "OD — Avg MAE per Location  (Right Eye)",
                'inferno', 0, 10, "MAE (dB)")
    plot_single(axes[0, 1], os_mae_grid,
                "OS — Avg MAE per Location  (Left Eye)",
                'inferno', 0, 10, "MAE (dB)")

    # Row 1: GT sensitivity heatmaps
    plot_single(axes[1, 0], od_gt_grid,
                "OD — Avg GT Sensitivity  (Right Eye)",
                'inferno', 0, 30, "Sensitivity (dB)")
    plot_single(axes[1, 1], os_gt_grid,
                "OS — Avg GT Sensitivity  (Left Eye)",
                'inferno', 0, 30, "Sensitivity (dB)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUTPUT_DIR, f"bilateral_MAE_{tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ 4-panel figure saved → {out_path}")
    return out_path

# ============== Print per-location table ==============
def print_table(results):
    print(f"\n{'='*60}")
    print("Per-location summary")
    print(f"{'='*60}")
    for eye in ['OD', 'OS']:
        r = results[eye]
        print(f"\n{eye}:")
        print(f"  {'Loc':>4}  {'MAE':>6}  {'GT':>6}  {'N':>5}")
        print(f"  {'-'*28}")
        for i in range(52):
            print(f"  {i:4d}  {r['mean_ae'][i]:6.2f}  {r['mean_gt'][i]:6.2f}  {r['count'][i]:5d}")

# ============== Main ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'train'], default='val')
    parser.add_argument('--no-tta', action='store_true')
    args = parser.parse_args()

    json_path = VAL_JSON if args.split == 'val' else TRAIN_JSON
    use_tta   = not args.no_tta
    tag       = f"{args.split}_tta{NUM_TTA_AUGS if use_tta else 0}"

    print(f"\n{'='*60}")
    print(f"Bilateral MAE Heatmap  |  split={args.split}  |  TTA={'on' if use_tta else 'off'}")
    print(f"{'='*60}")

    model   = load_model()
    results = collect_errors_bilateral(model, json_path, use_tta)

    save_four_panel(results, tag, args.split)
    print_table(results)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
