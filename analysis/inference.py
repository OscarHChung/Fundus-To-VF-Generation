#!/usr/bin/env python3
"""
Inference Script - Predict VF from Fundus Images

Usage:
    # Single image prediction
    python inference.py --image path/to/fundus.jpg --laterality OD
    
    # Batch prediction on test set
    python inference.py --test_json path/to/test.json --output results.json
    
    # Generate MAE heatmap
    python inference.py --test_json path/to/test.json --heatmap heatmap.png
"""

import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ============== Config ==============
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
INFERENCE_MODEL = os.path.join(CURRENT_DIR, "inference_model.pth")

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

valid_indices_od: List[int] = [i for i, v in enumerate(mask_OD.flatten()) if v]
valid_indices_os: List[int] = list(reversed(valid_indices_od))

# ============== Model Architecture ==============
sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

class VFAutoDecoder(nn.Module):
    def __init__(self, input_dim: int = 52, hidden_dims: List[int] = [256, 512, 512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:len(hidden_dims)//2]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        for hidden_dim in hidden_dims[len(hidden_dims)//2:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ImprovedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.residual = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        self.to_vf = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 52)
        )
    
    def forward(self, x):
        x = self.proj1(x)
        x = x + self.residual(x)
        x = self.to_vf(x)
        return x

class FundusToVFModel(nn.Module):
    def __init__(self, encoder, use_pretrained=True):
        super().__init__()
        self.encoder = encoder
        self.projection = ImprovedProjection()
        
        if use_pretrained:
            self.decoder = VFAutoDecoder(input_dim=52, hidden_dims=[256, 512, 512, 256])
        else:
            self.decoder = nn.Sequential(
                nn.Linear(52, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 52)
            )
    
    def forward(self, x):
        latent = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        vf_features = self.projection(latent)
        pred = self.decoder(vf_features)
        return pred

# ============== Load Model ==============
def load_model():
    """Load trained model for inference"""
    if not os.path.exists(INFERENCE_MODEL):
        raise FileNotFoundError(f"Model not found at {INFERENCE_MODEL}. Train model first!")
    
    # Load checkpoint
    checkpoint = torch.load(INFERENCE_MODEL, map_location=DEVICE, weights_only=False)
    
    # Load encoder
    encoder_checkpoint = torch.load(checkpoint['encoder_checkpoint'], map_location='cpu', weights_only=False)
    encoder = mae_vit_large_patch16_dec512d8b()
    encoder.load_state_dict(encoder_checkpoint['model'], strict=False)
    
    # Create model
    model = FundusToVFModel(encoder, use_pretrained=checkpoint.get('use_pretrained', True))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded (Val MAE: {checkpoint['val_mae']:.2f} dB)")
    return model

# ============== Transform ==============
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============== Inference Functions ==============
def predict_single_image(model, image_path: str, laterality: str = 'OD'):
    """
    Predict VF from single fundus image
    
    Returns:
        8x9 grid with predictions at valid locations, NaN elsewhere
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(img_tensor).cpu().numpy()[0]
    
    # Create 8x9 grid
    grid = np.full((8, 9), np.nan)
    valid_idx = valid_indices_od if laterality.upper().startswith('OD') else valid_indices_os
    
    for i, idx in enumerate(valid_idx):
        if i < len(pred):
            row, col = divmod(idx, 9)
            grid[row, col] = pred[i]
    
    return grid, pred

def predict_batch(model, test_json: str, fundus_dir: str, output_json: str = None):
    """
    Predict VF for all images in test set
    
    Returns:
        Dictionary with predictions and errors
    """
    with open(test_json, 'r') as f:
        data = json.load(f)
    
    results = []
    total_mae = 0.0
    n_valid = 0
    
    for item in data:
        img_path = os.path.join(fundus_dir, item['FundusImage'])
        laterality = item.get('Laterality', 'OD').strip().upper()
        
        # Predict
        grid, pred = predict_single_image(model, img_path, laterality)
        
        # Compute error if ground truth available
        if 'hvf' in item:
            hvf = np.array(item['hvf']).flatten()
            valid_idx = valid_indices_od if laterality.startswith('OD') else valid_indices_os
            target = hvf[valid_idx]
            
            # Filter masked values
            mask = target < 99.0
            if mask.sum() > 0:
                mae = np.abs(pred[mask] - target[mask]).mean()
                total_mae += mae * mask.sum()
                n_valid += mask.sum()
            else:
                mae = None
        else:
            mae = None
        
        results.append({
            'image': item['FundusImage'],
            'laterality': laterality,
            'prediction': pred.tolist(),
            'mae': float(mae) if mae is not None else None
        })
    
    avg_mae = total_mae / n_valid if n_valid > 0 else None
    
    output = {
        'results': results,
        'average_mae': avg_mae,
        'n_samples': len(results)
    }
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Results saved to {output_json}")
    
    if avg_mae:
        print(f"Average MAE: {avg_mae:.2f} dB")
    
    return output

def generate_heatmap(model, test_json: str, fundus_dir: str, output_path: str = 'heatmap.png'):
    """Generate MAE heatmap"""
    with open(test_json, 'r') as f:
        data = json.load(f)
    
    errors_od = [[] for _ in range(52)]
    errors_os = [[] for _ in range(52)]
    
    for item in data:
        if 'hvf' not in item:
            continue
        
        img_path = os.path.join(fundus_dir, item['FundusImage'])
        laterality = item.get('Laterality', 'OD').strip().upper()
        
        _, pred = predict_single_image(model, img_path, laterality)
        
        hvf = np.array(item['hvf']).flatten()
        valid_idx = valid_indices_od if laterality.startswith('OD') else valid_indices_os
        target = hvf[valid_idx]
        
        for i, (p, t) in enumerate(zip(pred, target)):
            if t < 99.0:
                error = abs(p - t)
                if laterality.startswith('OD'):
                    errors_od[i].append(error)
                else:
                    errors_os[i].append(error)
    
    # Create grids
    grid_od = np.full((8, 9), np.nan)
    grid_os = np.full((8, 9), np.nan)
    
    valid_idx_od = valid_indices_od
    valid_idx_os = valid_indices_os
    
    for i, idx in enumerate(valid_idx_od):
        if len(errors_od[i]) > 0:
            row, col = divmod(idx, 9)
            grid_od[row, col] = np.mean(errors_od[i])
    
    for i, idx in enumerate(valid_idx_os):
        if len(errors_os[i]) > 0:
            row, col = divmod(idx, 9)
            grid_os[row, col] = np.mean(errors_os[i])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(grid_od, cmap='hot', vmin=0, vmax=10)
    ax1.set_title('OD Eye - MAE Heatmap (dB)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(grid_os, cmap='hot', vmin=0, vmax=10)
    ax2.set_title('OS Eye - MAE Heatmap (dB)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Heatmap saved to {output_path}")

# ============== CLI ==============
def main():
    parser = argparse.ArgumentParser(description='Fundus to VF Inference')
    parser.add_argument('--image', type=str, help='Path to single fundus image')
    parser.add_argument('--laterality', type=str, default='OD', choices=['OD', 'OS'], 
                       help='Eye laterality')
    parser.add_argument('--test_json', type=str, help='Path to test JSON file')
    parser.add_argument('--fundus_dir', type=str, help='Directory containing fundus images')
    parser.add_argument('--output', type=str, help='Output JSON file for batch predictions')
    parser.add_argument('--heatmap', type=str, help='Generate MAE heatmap and save to path')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model()
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting VF for {args.image}...")
        grid, pred = predict_single_image(model, args.image, args.laterality)
        print(f"\nPredicted VF values (52 points):")
        print(pred)
        print(f"\n8x9 Grid (NaN = invalid location):")
        print(grid)
    
    # Batch prediction
    elif args.test_json:
        print(f"\nRunning batch prediction...")
        fundus_dir = args.fundus_dir or os.path.dirname(args.test_json)
        predict_batch(model, args.test_json, fundus_dir, args.output)
        
        # Generate heatmap if requested
        if args.heatmap:
            print(f"\nGenerating heatmap...")
            generate_heatmap(model, args.test_json, fundus_dir, args.heatmap)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()