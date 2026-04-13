"""
VF Scatterplot — Per-point predictions vs ground truth
Loads inference_model.pth and plots all VF sensitivity points
(one dot per VF location per patient, not averaged).
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# ============================================================
# PATHS — adjust if your layout differs
# ============================================================
CURRENT_DIR        = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR       = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH    = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_cfp_weights.pth')
INFERENCE_SAVE     = os.path.join(CURRENT_DIR, 'inference_model.pth')
BASE_DIR           = os.path.join(CURRENT_DIR, '..')
FUNDUS_DIR         = os.path.join(BASE_DIR, 'data', 'fundus', 'grape_fundus_images')
VAL_JSON           = os.path.join(BASE_DIR, 'data', 'vf_tests', 'grape_test.json')
OUTPUT_PLOT        = os.path.join(CURRENT_DIR, 'vf_scatterplot_per_point.png')

# ============================================================
# CONSTANTS (must match training.py)
# ============================================================
MASKED_VALUE_THRESHOLD = 99.0
OUTLIER_CLIP_RANGE     = (0, 35)
DROPOUT_RATE           = 0.3
PROJ_INIT_BIAS         = 18.0
TTA_ROTATIONS          = [-5, 0, 5]
USE_TTA                = True

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

# ============================================================
# SECTOR / QUADRANT HELPERS
# ============================================================
def build_sector_indices():
    rows, cols = mask_OD.shape
    sectors = {'sup_nasal': [], 'sup_temporal': [], 'inf_nasal': [], 'inf_temporal': []}
    valid_count = 0
    for r in range(rows):
        for c in range(cols):
            if mask_OD[r, c]:
                is_superior = r < 4
                is_nasal    = c <= 4
                if   is_superior and is_nasal:      sectors['sup_nasal'].append(valid_count)
                elif is_superior and not is_nasal:  sectors['sup_temporal'].append(valid_count)
                elif not is_superior and is_nasal:  sectors['inf_nasal'].append(valid_count)
                else:                               sectors['inf_temporal'].append(valid_count)
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
            idx = r * 14 + c
            if   r < 7 and c >= 7: quadrants['sup_retina_nasal'].append(idx)
            elif r < 7 and c < 7:  quadrants['sup_retina_temporal'].append(idx)
            elif r >= 7 and c >= 7: quadrants['inf_retina_nasal'].append(idx)
            else:                   quadrants['inf_retina_temporal'].append(idx)
    return quadrants

PATCH_QUADRANTS = build_patch_quadrants()

RETINA_TO_VF_OD = {
    'sup_retina_nasal': 'inf_temporal', 'sup_retina_temporal': 'inf_nasal',
    'inf_retina_nasal': 'sup_temporal', 'inf_retina_temporal': 'sup_nasal',
}
RETINA_TO_VF_OS = {
    'sup_retina_nasal': 'inf_nasal',    'sup_retina_temporal': 'inf_temporal',
    'inf_retina_nasal': 'sup_nasal',    'inf_retina_temporal': 'sup_temporal',
}

# ============================================================
# MODEL DEFINITION (identical to training.py)
# ============================================================
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


class SectorHead(nn.Module):
    def __init__(self, input_dim, num_points, dropout=0.3, bias_init=18.0):
        super().__init__()
        hidden = max(64, num_points * 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_points)
        )
        nn.init.constant_(self.net[-1].bias, bias_init)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.net(x)


class RegionalVFModel(nn.Module):
    def __init__(self, encoder, dropout=0.3):
        super().__init__()
        self.encoder   = encoder
        self.embed_dim = 1024

        for p in self.encoder.parameters():
            p.requires_grad = False

        for quad_name, indices in PATCH_QUADRANTS.items():
            self.register_buffer(f'patch_idx_{quad_name}',
                                 torch.tensor(indices, dtype=torch.long))
        for sec_name, indices in VF_SECTORS.items():
            self.register_buffer(f'vf_idx_{sec_name}',
                                 torch.tensor(indices, dtype=torch.long))

        sector_input_dim = self.embed_dim * 2
        self.sector_heads = nn.ModuleDict({
            sec_name: SectorHead(sector_input_dim, len(sec_indices), dropout, PROJ_INIT_BIAS)
            for sec_name, sec_indices in VF_SECTORS.items()
        })

        self.fusion = nn.Sequential(
            nn.Linear(NUM_VALID_POINTS, NUM_VALID_POINTS * 2),
            nn.LayerNorm(NUM_VALID_POINTS * 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(NUM_VALID_POINTS * 2, NUM_VALID_POINTS),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(0.3))

        self.decoder = VFAutoDecoder(input_dim=NUM_VALID_POINTS)

    def pool_quadrant(self, patches, quad_name):
        idx = getattr(self, f'patch_idx_{quad_name}')
        return patches[:, idx, :].mean(dim=1)

    def forward(self, x, laterality='OD', average_multi=True):
        with torch.no_grad():
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        cls_token = latent[:, 0, :]
        patches   = latent[:, 1:, :]

        mapping = RETINA_TO_VF_OD if (
            laterality if isinstance(laterality, str) else laterality[0]
        ).startswith('OD') else RETINA_TO_VF_OS

        sector_preds = {}
        for retina_quad, vf_sector in mapping.items():
            quad_feat = self.pool_quadrant(patches, retina_quad)
            head_input = torch.cat([quad_feat, cls_token], dim=1)
            sector_preds[vf_sector] = self.sector_heads[vf_sector](head_input)

        pred = torch.zeros(x.shape[0], NUM_VALID_POINTS, device=x.device)
        for sec_name, sec_pred in sector_preds.items():
            idx = getattr(self, f'vf_idx_{sec_name}')
            pred[:, idx] = sec_pred

        fused = self.fusion(pred)
        pred  = pred + self.fusion_alpha * fused
        pred  = self.decoder(pred)
        pred  = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred  = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])

        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred

# ============================================================
# DATASET (val-mode, with TTA)
# ============================================================
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


class ValDataset(Dataset):
    def __init__(self, json_path, fundus_dir, use_tta=True):
        with open(json_path) as f:
            data = json.load(f)
        self.fundus_dir = fundus_dir
        self.use_tta    = use_tta
        self.samples    = []
        for item in data:
            images     = item['FundusImage'] if isinstance(item['FundusImage'], list) else [item['FundusImage']]
            hvf        = item['hvf']
            laterality = item.get('Laterality', 'OD').strip().upper()
            self.samples.append({'images': images, 'hvf': hvf, 'laterality': laterality})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        imgs = []
        for p in s['images']:
            img = Image.open(os.path.join(self.fundus_dir, p)).convert('RGB')
            if self.use_tta:
                for t in get_tta_transforms():
                    imgs.append(t(img))
            else:
                imgs.append(val_transform(img))
        hvf = np.array(s['hvf'], dtype=np.float32).flatten()
        return torch.stack(imgs), torch.tensor(hvf), s['laterality']

def collate_fn(batch):
    return batch[0]

# ============================================================
# MAIN
# ============================================================
def main():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('✓ Using MPS')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('✓ Using CUDA')
    else:
        device = torch.device('cpu')
        print('⚠  Using CPU')

    # ── Load encoder ──────────────────────────────────────────
    sys.path.insert(0, RETFOUND_DIR)
    from models_mae import mae_vit_large_patch16_dec512d8b

    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    encoder = mae_vit_large_patch16_dec512d8b()
    encoder.load_state_dict(ckpt['model'], strict=False)
    print('✓ Loaded RETFound encoder')

    # ── Load model ────────────────────────────────────────────
    model = RegionalVFModel(encoder, DROPOUT_RATE)
    saved = torch.load(INFERENCE_SAVE, map_location='cpu', weights_only=False)
    state = saved.get('model_state_dict', saved.get('model', saved))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    print(f'✓ Loaded inference model  (val_mae={saved.get("val_mae", "?"):.2f} dB)')

    # ── Run inference ─────────────────────────────────────────
    dataset = ValDataset(VAL_JSON, FUNDUS_DIR, use_tta=USE_TTA)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    all_pred, all_true = [], []
    n_samples = 0

    with torch.no_grad():
        for imgs, hvf, lat in loader:
            imgs = imgs.to(device)
            pred = model(imgs, laterality=lat, average_multi=True)  # (1, 52)
            pred_np = pred.squeeze(0).cpu().numpy()

            hvf_np = hvf.numpy().flatten()
            valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
            true_52   = hvf_np[valid_idx]

            # Keep only unmasked points
            mask = true_52 < MASKED_VALUE_THRESHOLD
            all_pred.extend(pred_np[mask].tolist())
            all_true.extend(true_52[mask].tolist())
            n_samples += 1

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    n_points = len(all_pred)
    print(f'✓ Collected {n_points:,} VF points from {n_samples} samples')

    # ── Metrics ───────────────────────────────────────────────
    abs_err = np.abs(all_pred - all_true)
    mae     = abs_err.mean()
    rmse    = np.sqrt(((all_pred - all_true) ** 2).mean())
    bias    = (all_pred - all_true).mean()
    slope, intercept, r_value, _, _ = stats.linregress(all_true, all_pred)
    r2      = r_value ** 2

    print(f'\nMetrics:')
    print(f'  MAE:  {mae:.2f} dB')
    print(f'  RMSE: {rmse:.2f} dB')
    print(f'  Bias: {bias:+.2f} dB')
    print(f'  R²:   {r2:.3f}')

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by absolute error (0–10 dB range like the reference)
    cmap   = plt.cm.RdYlGn_r
    norm   = mcolors.Normalize(vmin=0, vmax=10)
    colors = cmap(norm(abs_err))

    sc = ax.scatter(
        all_true, all_pred,
        c=abs_err, cmap=cmap, norm=norm,
        s=18, alpha=0.65, linewidths=0,
        zorder=2
    )

    # Perfect prediction line
    lim = (min(all_true.min(), all_pred.min()) - 1,
           max(all_true.max(), all_pred.max()) + 1)
    lim = (max(-1, lim[0]), min(36, lim[1]))
    ax.plot(lim, lim, 'k--', linewidth=1.5, label='Perfect prediction', zorder=3)

    # Regression line
    x_line = np.linspace(lim[0], lim[1], 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'Fit: y={slope:.2f}x+{intercept:.2f} (R²={r2:.3f})', zorder=4)

    # Colorbar
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('Absolute Error (dB)', fontsize=11)
    cb.set_ticks(range(0, 11, 2))

    # Text box (top-left, matching reference style)
    info = (
        f'Fit: y={slope:.2f}x+{intercept:.2f} (R²={r2:.3f})\n'
        f'Bias: {bias:+.2f} dB'
    )
    ax.text(0.03, 0.97, info, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan',
                      edgecolor='steelblue', alpha=0.85))

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('True VF Sensitivity (dB)', fontsize=12)
    ax.set_ylabel('Predicted VF Sensitivity (dB)', fontsize=12)
    ax.set_title(
        f'All Predictions vs Ground Truth\n'
        f'N = {n_points:,} points from {n_samples} samples',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f'\n✓ Saved → {OUTPUT_PLOT}')
    plt.show()


if __name__ == '__main__':
    main()