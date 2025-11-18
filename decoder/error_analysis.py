#!/usr/bin/env python3
"""
Diagnostic script to understand why the model isn't learning
FIXED: Handle masked values (100) properly
"""
import os, sys, json, numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETFOUND_DIR = os.path.join(CURRENT_DIR, '..', 'encoder', 'RETFound_MAE')
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "..", "encoder", "RETFound_cfp_weights.pth")
BASE_DIR = os.path.join(CURRENT_DIR, "..")
FUNDUS_DIR = os.path.join(BASE_DIR, "data", "fundus", "grape_fundus_images")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "vf_tests", "grape_train.json")

sys.path.insert(0, RETFOUND_DIR)
from models_mae import mae_vit_large_patch16_dec512d8b

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

retfound_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GRAPEDataset(Dataset):
    def __init__(self, json_path: str, fundus_dir: str, transform):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.fundus_dir = fundus_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.fundus_dir, item['FundusImage'])
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        hvf = np.array(item['hvf'], dtype=np.float32).flatten()
        laterality = item.get('Laterality', 'OD').strip().upper()
        return img_tensor, torch.tensor(hvf), laterality

print("="*60)
print("DIAGNOSTIC: Checking Data and Model Behavior")
print("="*60)

# Load dataset
dataset = GRAPEDataset(TRAIN_JSON, FUNDUS_DIR, retfound_transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get a batch
imgs, hvf, laterality = next(iter(loader))

print("\n1. DATA STATISTICS:")
print(f"   Batch size: {len(imgs)}")
print(f"   Image shape: {imgs.shape}")
print(f"   HVF shape: {hvf.shape}")
print(f"   Laterality: {laterality}")

# Check HVF statistics with MASKING
print("\n   CRITICAL: Checking for masked values (100)...")
for i, lat in enumerate(laterality):
    valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
    valid_hvf = hvf[i, valid_idx].numpy()
    
    # Check for masked values
    mask_100 = valid_hvf >= 99  # Values of 100 are masked
    n_masked = mask_100.sum()
    
    # Get only VALID (non-masked) values
    valid_hvf_clean = valid_hvf[~mask_100]
    
    print(f"\n   Sample {i} ({lat}):")
    print(f"     Total valid indices: {len(valid_hvf)}")
    print(f"     MASKED values (100): {n_masked}")
    print(f"     USABLE values: {len(valid_hvf_clean)}")
    if len(valid_hvf_clean) > 0:
        print(f"     Clean HVF range: [{valid_hvf_clean.min():.2f}, {valid_hvf_clean.max():.2f}]")
        print(f"     Clean HVF mean: {valid_hvf_clean.mean():.2f}, std: {valid_hvf_clean.std():.2f}")
        print(f"     Sample values: {valid_hvf[:5]}")
    else:
        print(f"     ERROR: No usable values after masking!")

# Load encoder
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
encoder = mae_vit_large_patch16_dec512d8b()
encoder.load_state_dict(checkpoint['model'], strict=False)
encoder.to(DEVICE)
encoder.eval()

print("\n2. ENCODER OUTPUT:")
with torch.no_grad():
    imgs_device = imgs.to(DEVICE)
    latent = encoder.forward_encoder(imgs_device, mask_ratio=0.0)[0]
    if latent.dim() == 3:
        latent = latent[:, 0, :]
    print(f"   Latent shape: {latent.shape}")
    print(f"   Latent range: [{latent.min().item():.4f}, {latent.max().item():.4f}]")
    print(f"   Latent mean: {latent.mean().item():.4f}, std: {latent.std().item():.4f}")

print("\n3. SIMPLE BASELINE TEST (WITH PROPER MASKING):")
print("   Testing if a simple linear layer can map encoder→VF...")

simple_model = nn.Linear(1024, 52).to(DEVICE)
optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

# Train for 100 steps on this single batch
losses = []
for step in range(100):
    optimizer.zero_grad()
    pred = simple_model(latent)
    
    # Compute loss ONLY on non-masked points
    total_loss = 0.0
    total_points = 0
    for i, lat in enumerate(laterality):
        valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
        
        # Get predictions and targets
        pred_i = pred[i]
        target_i = hvf[i].to(DEVICE)
        
        # Create mask for valid indices AND non-100 values
        target_valid = target_i[valid_idx]
        pred_valid = pred_i
        
        # Filter out masked values (100)
        non_masked = target_valid < 99
        if non_masked.sum() > 0:
            pred_clean = pred_valid[non_masked]
            target_clean = target_valid[non_masked]
            
            loss = torch.abs(pred_clean - target_clean).mean()
            total_loss += loss
            total_points += 1
    
    if total_points > 0:
        total_loss = total_loss / total_points
        total_loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            losses.append(total_loss.item())
            print(f"   Step {step}: Loss = {total_loss.item():.3f} dB")

if len(losses) > 1:
    print(f"\n   Loss decreased from {losses[0]:.3f} to {losses[-1]:.3f} dB")
    if losses[-1] < losses[0] * 0.5:
        print("   ✓ Simple model CAN learn on this batch (after proper masking)!")
    else:
        print("   ✗ WARNING: Even simple model struggles after masking")

print("\n4. DATASET-WIDE MASKING STATISTICS:")
all_hvf_clean = []
all_masked_counts = []

for i in range(len(dataset)):
    _, hvf_sample, lat = dataset[i]
    valid_idx = valid_indices_od if lat.startswith('OD') else valid_indices_os
    valid_hvf = hvf_sample[valid_idx].numpy()
    
    # Count masked values
    n_masked = (valid_hvf >= 99).sum()
    all_masked_counts.append(n_masked)
    
    # Get clean values
    clean_hvf = valid_hvf[valid_hvf < 99]
    all_hvf_clean.extend(clean_hvf.tolist())

all_hvf_clean = np.array(all_hvf_clean)
all_masked_counts = np.array(all_masked_counts)

print(f"   Total samples: {len(dataset)}")
print(f"   Average masked points per sample: {all_masked_counts.mean():.1f} / 52")
print(f"   Samples with >20 masked points: {(all_masked_counts > 20).sum()}")
print(f"\n   Clean HVF statistics (excluding 100s):")
print(f"     Total clean values: {len(all_hvf_clean):,}")
print(f"     Range: [{all_hvf_clean.min():.2f}, {all_hvf_clean.max():.2f}]")
print(f"     Mean: {all_hvf_clean.mean():.2f}, Std: {all_hvf_clean.std():.2f}")
print(f"     Median: {np.median(all_hvf_clean):.2f}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("\n⚠️  CRITICAL FINDING:")
print("   Values of 100 are MASKED and must be excluded from loss!")
print("="*60)
