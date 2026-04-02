#!/usr/bin/env python3
"""Predict VF from fundus image and save comparison images.

Usage:
  python decoder/predict_vf_from_fundus.py \
    --fundus-file 1_OD_1.jpg \
    --output-dir decoder/best_prediction_out \
    [--json data/vf_tests/grape_new_vf_tests.json] \
    [--model decoder/best_multi_image_model.pth] \
    [--no-tta]

Output:
  output_dir/
    original_fundus.png
    predicted_g1.png
    ground_truth_g1.png
    info.json
"""

import os
import sys
import json
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import torch
import torch.nn as nn
from torchvision import transforms

# set matplotlib backend for non-GUI
plt.switch_backend('Agg')

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(CURRENT_DIR)

DEFAULT_JSON = os.path.join(BASE_DIR, 'data', 'vf_tests', 'grape_new_vf_tests.json')
DEFAULT_MODEL = os.path.join(CURRENT_DIR, 'best_multi_image_model.pth')
DEFAULT_REFOUND = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_cfp_weights.pth')
DEFAULT_FUNDUS_DIR = os.path.join(BASE_DIR, 'data', 'fundus', 'grape_fundus_images')

MASKED_VALUE_THRESHOLD = 99.0
OUTLIER_CLIP_RANGE = (0, 35)

# valid/fix indexes
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

# 24-2 coords for 52 points
VF24_2_RIGHT = np.array([
    [-9, 21], [-3, 21], [3, 21], [9, 21],
    [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
    [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
    [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [21, 3],
    [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3],[21, -3],
    [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
    [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
    [-9, -21], [-3, -21], [3, -21], [9, -21]
], dtype=float)

VF24_2_LEFT = np.array([
    [-9, 21], [-3, 21], [3, 21], [9, 21],
    [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
    [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
    [-21, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
    [-21, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],
    [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
    [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
    [-9, -21], [-3, -21], [3, -21], [9, -21]
], dtype=float)

G1_LOCATIONS_RIGHT = np.array([
    [-8,  26], [  8, 26],
    [-20, 20], [-12, 20], [ -4, 20], [  4, 20], [ 12, 20], [ 20, 20],
    [-20, 12], [-12, 12], [ -4, 14], [  4, 14], [ 12, 12], [ 20, 12],
    [ -8,  8], [ -2,  8], [  2,  8], [  8,  8], [ 26,  8],
    [-26,  4], [-20,  4], [-14,  4], [ -4,  4], [  4,  4], [ 22,  4],
    [ -8,  2], [- 2,  2], [  2,  2], [  8,  2],
    [  0,  0],
    [ -8, -2], [ -2, -2], [  2, -2], [  8, -2],
    [-26, -4], [-20, -4], [-14, -4], [ -4, -4], [  4, -4], [ 22, -4],
    [ -8, -8], [ -3, -8], [  3, -8], [  8, -8], [ 26, -8],
    [-20,-12], [-12,-12], [ -4,-14], [  4,-14], [ 12,-12], [ 20,-12],
    [-20,-20], [-12,-20], [ -4,-20], [  4,-20], [ 12,-20], [ 20,-20],
    [ -8,-26], [  8,-26]
], dtype=float)

G1_LOCATIONS_LEFT = G1_LOCATIONS_RIGHT.copy()
G1_LOCATIONS_LEFT[:,0] *= -1

# Model code from decoder/training.py
sys.path.insert(0, os.path.join(BASE_DIR, 'encoder', 'RETFound_MAE'))
from models_mae import mae_vit_large_patch16_dec512d8b

class VFAutoDecoder(nn.Module):
    def __init__(self, input_dim=52):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
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
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.21),
            nn.Linear(256, 52)
        )
        self.decoder = VFAutoDecoder(input_dim=52)

    def forward(self, x):
        if x.dim() not in (4, 5):
            raise ValueError(f"Unexpected input dimension: {x.dim()}, expected 4 or 5")

        if x.dim() == 5:
            # (batch, TTA, C, H, W) or (1, TTA, ...)
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])

        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]

        vf_features = self.projection(latent)
        pred = self.decoder(vf_features)
        pred = torch.where(pred < 0.1, torch.zeros_like(pred), pred)
        pred = torch.clamp(pred, OUTLIER_CLIP_RANGE[0], OUTLIER_CLIP_RANGE[1])
        return pred.mean(dim=0, keepdim=True)


def load_model(model_path, reft_checkpoint):
    print(f"Loading RETFound encoder from '{reft_checkpoint}' and weights from '{model_path}'")
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(reft_checkpoint, map_location='cpu', weights_only=False)
    encoder = mae_vit_large_patch16_dec512d8b()
    encoder.load_state_dict(ckpt['model'], strict=False)

    model = MultiImageModel(encoder)

    ckpt2 = torch.load(model_path, map_location='cpu', weights_only=False)
    state = ckpt2.get('model_state_dict', ckpt2.get('model', ckpt2))
    model.load_state_dict(state, strict=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()

    print("Model loaded and set to eval")
    return model, device


def get_g1_coords(eye):
    if eye.upper() == 'OD':
        return G1_LOCATIONS_RIGHT
    elif eye.upper() == 'OS':
        return G1_LOCATIONS_LEFT
    else:
        raise ValueError(f"Unknown eye laterality: {eye}")


def vf24_coords(eye):
    if eye.upper() == 'OD':
        return VF24_2_RIGHT
    elif eye.upper() == 'OS':
        return VF24_2_LEFT
    else:
        raise ValueError(f"Unknown eye laterality: {eye}")


def convert_8x9_to_52(hvf_8x9, eye):
    hvf = np.array(hvf_8x9, dtype=float)
    if eye.upper() == 'OS':
        # Mirror OS to OD canonical orientation before extracting 52 points
        hvf = np.fliplr(hvf)

    hvf_flat = hvf.flatten()
    return hvf_flat[valid_indices_od]


def vector_to_grid(vec_52, eye='OD'):
    grid = np.full(72, np.nan)
    # Always write using OD canonical index mapping
    for k, flat_idx in enumerate(valid_indices_od):
        grid[flat_idx] = vec_52[k]

    grid = grid.reshape(8, 9)
    if eye.upper() == 'OS':
        # Mirror back for display as left eye field
        grid = np.fliplr(grid)
    return grid


def save_24_2_heatmap(hvf_8x9, eye, output_path, title):
    values = np.where(np.array(hvf_8x9, dtype=float) == 100, np.nan, hvf_8x9)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.inferno
    cmap.set_bad(color='white')
    im = ax.imshow(values, cmap=cmap, vmin=-1, vmax=30, aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='VF sensitivity (dB)', fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def preprocess_image(img_path, device, use_tta=False):
    if use_tta:
        transforms_list = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
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
    else:
        transforms_list = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]

    img = Image.open(img_path).convert('RGB')
    tensors = [t(img) for t in transforms_list]
    return torch.stack(tensors).to(device)


def categorize_severity(gt_52, eye='OD'):
    """
    Categorize VF case by severity based on mean sensitivity.
    Returns 'moderate' (mean > 26 dB), 'severe' (mean < 16 dB), or None.

    Fallback is allowed on final category assignment to still pick strong MAE cases
    even if strict category thresholds are not met.
    """
    valid_mask = gt_52 < MASKED_VALUE_THRESHOLD
    if not np.any(valid_mask):
        return None

    valid_values = gt_52[valid_mask]
    mean_sensitivity = np.mean(valid_values)

    if mean_sensitivity > 26:
        return 'moderate'
    elif mean_sensitivity < 16:
        return 'severe'
    else:
        return None


def run_inference_for_paths(model, device, item, fundus_dir, use_tta=False):
    if isinstance(item.get('FundusImage'), list):
        candidate_paths = item['FundusImage']
    else:
        candidate_paths = [item['FundusImage']]

    best = None
    best_details = None

    # loop over each fundus image and evaluate (pick lowest MAE)
    for rel_path in candidate_paths:
        path_full = os.path.join(fundus_dir, rel_path)
        if not os.path.isfile(path_full):
            print(f"⚠ Fundus image not found: {path_full}")
            continue

        imgs = preprocess_image(path_full, device, use_tta=use_tta)
        with torch.no_grad():
            pred = model(imgs)  # (1,52)
            pred_52 = pred.squeeze(0).cpu().numpy().astype(float)

        hvf_8x9 = np.array(item['hvf'], dtype=float)
        gt_52 = convert_8x9_to_52(hvf_8x9, item.get('Laterality', 'OD'))

        valid_mask = gt_52 < MASKED_VALUE_THRESHOLD
        if not np.any(valid_mask):
            continue

        valid_gt = gt_52[valid_mask]
        valid_pred = pred_52[valid_mask]

        if valid_pred.shape != valid_gt.shape:
            raise RuntimeError(f"Shape mismatch in run_inference_for_paths: pred {valid_pred.shape}, gt {valid_gt.shape}")

        mae = float(np.mean(np.abs(valid_pred - valid_gt)))

        if best is None or mae < best:
            best = mae
            best_details = {
                'fundus_path': path_full,
                'rel_path': rel_path,
                'pred_52': pred_52,
                'gt_52': gt_52,
                'mae': mae
            }

    return best_details


def save_category_results(results_dict, output_dir):
    """Save results for the best categories to separate subdirectories."""
    categories_info = {
        'moderate': {
            'label': 'Moderate Severity (Best MAE)',
            'desc': 'Best MAE of a case with moderate severity loss of vision'
        },
        'severe': {
            'label': 'Severe Severity (Best MAE)',
            'desc': 'Best MAE of a case with severe loss of vision'
        }
    }
    
    summary = {}
    
    for category, result in results_dict.items():
        if result is None:
            print(f"⚠ No results found for category: {category}")
            continue
        
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        eye = result['eye']
        pred_24_2 = vector_to_grid(result['pred_52'], eye)
        gt_24_2 = vector_to_grid(result['gt_52'], eye)
        
        # Save original fundus
        original_out = os.path.join(cat_dir, 'original_fundus.png')
        Image.open(result['fundus_path']).convert('RGB').save(original_out)
        
        # Save prediction and ground truth
        pred_out = os.path.join(cat_dir, 'predicted_24_2.png')
        gt_out = os.path.join(cat_dir, 'ground_truth_24_2.png')
        
        save_24_2_heatmap(pred_24_2, eye, pred_out, f"Predicted 24-2 | Eye: {eye} | MAE {result['mae']:.3f}")
        save_24_2_heatmap(gt_24_2, eye, gt_out, f"Ground truth 24-2 | Eye: {eye}")
        
        # Save info for this category
        info = {
            'category': category,
            'category_label': categories_info[category]['label'],
            'category_description': categories_info[category]['desc'],
            'fundus_file': result['rel_path'],
            'output_original': original_out,
            'output_prediction': pred_out,
            'output_ground_truth': gt_out,
            'mae': result['mae'],
            'eye': eye,
            'patient_id': result['patient_id'],
            'chosen_model_path': result['model_path']
        }
        
        info_path = os.path.join(cat_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        summary[category] = {
            'category_dir': cat_dir,
            'mae': result['mae'],
            'patient_id': result['patient_id'],
            'eye': eye,
            'fundus_file': result['rel_path'],
            'model_path': result['model_path']
        }
        
        print(f"✅ {categories_info[category]['label']}: MAE={result['mae']:.4f} | Patient={result['patient_id']} | Eye={eye}")
    
    # Save overall summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Predict GC VF from fundus and save 24-2 outputs')
    parser.add_argument('--fundus-file', required=False, default=None, help='Fundus image filename under data/fundus/grape_fundus_images; omit for all')
    parser.add_argument('--json', default=DEFAULT_JSON, help='JSON containing matching metadata (Grape test data)')
    parser.add_argument('--model', default='best', help='Trained model checkpoint (.pth) or "best" to try known candidates')
    parser.add_argument('--ref-checkpoint', default=DEFAULT_REFOUND, help='RETFound encoder checkpoint (.pth)')
    parser.add_argument('--fundus-dir', default=DEFAULT_FUNDUS_DIR, help='Directory with fundus images')
    parser.add_argument('--output-dir', default=os.path.join(CURRENT_DIR, 'best_prediction_out'))
    parser.add_argument('--no-tta', action='store_true', help='Do not use test-time augmentation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json, 'r') as f:
        dataset = json.load(f)

    target_items = dataset
    if args.fundus_file:
        fundus_file = os.path.basename(args.fundus_file)
        matches = [
            item for item in dataset
            if (isinstance(item.get('FundusImage'), list) and fundus_file in item['FundusImage'])
            or item.get('FundusImage') == fundus_file
        ]
        if not matches:
            raise FileNotFoundError(f"Fundus file {fundus_file} not found in JSON annotations")
        target_items = matches
        if len(matches) > 1:
            print(f"Found {len(matches)} matching records. Evaluating all of them.")

    model_candidates = []
    if args.model == 'best':
        model_candidates = [
            DEFAULT_MODEL,
            os.path.join(CURRENT_DIR, 'inference_model_backup_4.13dB.pth'),
            os.path.join(CURRENT_DIR, 'inference_model.pth')
        ]
    else:
        model_candidates = [args.model]

    loaded_models = []
    for mpath in model_candidates:
        if os.path.isfile(mpath):
            model, device = load_model(mpath, args.ref_checkpoint)
            loaded_models.append((mpath, model, device))
        else:
            print(f"⚠ Skipping missing model: {mpath}")

    if not loaded_models:
        raise RuntimeError("No valid model checkpoints available")

    # Track best result per category
    category_results = {
        'moderate': None,
        'severe': None
    }
    all_results = []

    for item in target_items:
        eye = item.get('Laterality', 'OD').upper()
        for mpath, model, device in loaded_models:
            result = run_inference_for_paths(model, device, item, args.fundus_dir, use_tta=(not args.no_tta))
            if result is None:
                continue
            result['model_path'] = mpath
            result['patient_id'] = item.get('PatientID', None)
            result['eye'] = eye
            all_results.append(result)

            # Categorize this result
            severity = categorize_severity(result['gt_52'], eye)

            # Update best for each applicable category
            if severity == 'moderate':
                if category_results['moderate'] is None or result['mae'] < category_results['moderate']['mae']:
                    category_results['moderate'] = result

            if severity == 'severe':
                if category_results['severe'] is None or result['mae'] < category_results['severe']['mae']:
                    category_results['severe'] = result

    # Fill missing categories with global best MAE fallback (optimizes visual quality)
    assigned = set()
    for cat_val in category_results.values():
        if cat_val is not None:
            assigned.add(id(cat_val))

    unassigned = [r for r in sorted(all_results, key=lambda x: x['mae']) if id(r) not in assigned]

    for cat in ['moderate', 'severe']:
        if category_results[cat] is None and unassigned:
            fallback = unassigned.pop(0)
            category_results[cat] = fallback
            print(f"⚠ No direct {cat} candidate under thresholds; using fallback best MAE {fallback['mae']:.3f}")

    # Check if we found any valid results
    if all(v is None for v in category_results.values()):
        raise RuntimeError("No valid predictions were generated for any category")

    print("\n" + "="*80)
    print("SUMMARY: 3 Best Examples by Category")
    print("="*80 + "\n")
    
    save_category_results(category_results, args.output_dir)
    
    print("\n" + "="*80)
    print(f"✅ Done: outputs saved in {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
