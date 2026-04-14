"""
VF Scatterplot — Per-point predictions vs ground truth
Loads inference_model.pth (v10.2 PerPointVFModel) and plots all VF
sensitivity points (one dot per VF location per patient).
"""

import os, sys, json, argparse, math
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
# PATHS
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
# CONSTANTS
# ============================================================
MASKED_VALUE_THRESHOLD = 99.0
OUTLIER_CLIP_RANGE     = (0, 35)
PROJ_INIT_BIAS         = 18.0
TTA_ROTATIONS          = [-5, 0, 5]
USE_TTA                = True
ATTN_DROPOUT           = 0.4
HEAD_DROPOUT           = 0.4
REFINE_DROPOUT         = 0.15
ATTN_TEMP_INIT         = 2.0

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
# ANATOMICAL PRIOR
# ============================================================
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

# ============================================================
# MODEL DEFINITION (v10.2 — PerPointVFModel)
# ============================================================
class PerPointAttention(nn.Module):
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

        temp = F.softplus(self.temperature) + 0.5
        logits = logits / temp
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attended = torch.bmm(attn_weights, vals)
        out = self.out_proj(attended)
        return out, attn_weights


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


class PerPointVFModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = 1024

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.attention = PerPointAttention(
            embed_dim=self.embed_dim,
            num_points=NUM_VALID_POINTS,
            attn_dim=128,
            dropout=ATTN_DROPOUT
        )
        self.point_head = PointHead(
            input_dim=self.embed_dim * 2,
            hidden=256,
            dropout=HEAD_DROPOUT,
            bias_init=PROJ_INIT_BIAS
        )
        self.refinement = CrossPointRefinement(NUM_VALID_POINTS, hidden=104, dropout=REFINE_DROPOUT)

    def forward(self, x, laterality='OD', average_multi=True):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.shape}")

        with torch.no_grad():
            latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        cls_token = latent[:, 0, :]
        patches = latent[:, 1:, :]

        point_feats, _ = self.attention(patches, laterality)

        B = x.shape[0]
        cls_expanded = cls_token.unsqueeze(1).expand(B, NUM_VALID_POINTS, self.embed_dim)
        combined = torch.cat([point_feats, cls_expanded], dim=2)

        pred = self.point_head(combined)
        pred = self.refinement(pred)

        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])

        if average_multi and pred.shape[0] > 1:
            pred = pred.mean(dim=0, keepdim=True)
        return pred

# ============================================================
# DATASET
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

    # ── Load model (v10.2 PerPointVFModel) ────────────────────
    model = PerPointVFModel(encoder)
    saved = torch.load(INFERENCE_SAVE, map_location='cpu', weights_only=False)
    state = saved.get('model_state_dict', saved.get('model', saved))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    val_mae = saved.get("val_mae", None)
    mae_str = f'{val_mae:.2f} dB' if val_mae is not None else '?'
    print(f'✓ Loaded inference model  (val_mae={mae_str})')

    # ── Run inference ─────────────────────────────────────────
    dataset = ValDataset(VAL_JSON, FUNDUS_DIR, use_tta=USE_TTA)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    all_pred, all_true = [], []
    n_samples = 0

    with torch.no_grad():
        for imgs, hvf, lat in loader:
            imgs = imgs.to(device)
            pred = model(imgs, laterality=lat, average_multi=True)
            pred_np = pred.squeeze(0).cpu().numpy()

            hvf_np = hvf.numpy().flatten()
            valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
            true_52   = hvf_np[valid_idx]

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
    print(f'  MAE:   {mae:.2f} dB')
    print(f'  RMSE:  {rmse:.2f} dB')
    print(f'  Bias:  {bias:+.2f} dB')
    print(f'  Slope: {slope:.3f}')
    print(f'  R²:    {r2:.3f}')

    # ── Plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap   = plt.cm.RdYlGn_r
    norm   = mcolors.Normalize(vmin=0, vmax=10)

    sc = ax.scatter(
        all_true, all_pred,
        c=abs_err, cmap=cmap, norm=norm,
        s=18, alpha=0.65, linewidths=0,
        zorder=2
    )

    lim = (min(all_true.min(), all_pred.min()) - 1,
           max(all_true.max(), all_pred.max()) + 1)
    lim = (max(-1, lim[0]), min(36, lim[1]))
    ax.plot(lim, lim, 'k--', linewidth=1.5, label='Perfect prediction', zorder=3)

    x_line = np.linspace(lim[0], lim[1], 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'Fit: y={slope:.2f}x+{intercept:.2f} (R²={r2:.3f})', zorder=4)

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('Absolute Error (dB)', fontsize=11)
    cb.set_ticks(range(0, 11, 2))

    info = (
        f'Fit: y={slope:.2f}x+{intercept:.2f} (R²={r2:.3f})\n'
        f'Bias: {bias:+.2f} dB\n'
        f'Slope: {slope:.3f}'
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
        f'All Predictions vs Ground Truth (v10.2)\n'
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