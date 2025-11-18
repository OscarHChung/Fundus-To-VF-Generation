import json
import os
from typing import List
import numpy as np

# Mask definition
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

valid_indices_od = [i for i,v in enumerate(mask_OD.flatten()) if v]
valid_indices_os = list(reversed(valid_indices_od))

print("=== Data Diagnostics ===\n")

# Load data
with open('data/vf_tests/grape_train.json') as f:
    grape_train = json.load(f)
with open('data/vf_tests/grape_test.json') as f:
    grape_test = json.load(f)

print(f"GRAPE Train: {len(grape_train)} samples")
print(f"GRAPE Test: {len(grape_test)} samples")
print(f"Valid VF points: {len(valid_indices_od)} (should be 52)\n")

# Check if images exist
missing_count = 0
for sample in grape_train[:5]:  # Check first 5
    img_path = f"data/fundus/grape_fundus_images/{sample['FundusImage']}"
    exists = os.path.exists(img_path)
    print(f"{sample['FundusImage']}: {'✓' if exists else '✗ MISSING'}")
    if not exists:
        missing_count += 1

# Check VF data structure
sample = grape_train[0]
hvf_flat = np.array(sample['hvf']).flatten()
laterality = sample['Laterality']
valid_indices = valid_indices_od if laterality == 'OD' else valid_indices_os

print(f"\nSample HVF shape: {np.array(sample['hvf']).shape}")
print(f"Laterality: {laterality}")
print(f"Non-masked values (should be 52): {np.sum(hvf_flat != 100)}")

# Get valid values
valid_values = [hvf_flat[i] for i in valid_indices]
print(f"Valid value range: {min(valid_values):.1f} to {max(valid_values):.1f}")
print(f"Valid values mean: {np.mean(valid_values):.1f}")