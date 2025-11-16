import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoder.retfound_encoder import encoder

# ===========================
# Datasets
# ===========================
class VFOnlyDataset(Dataset):
    """Dataset for JSON containing VF test values used in decoder pretraining."""


class PairedDataset(Dataset):
    """Dataset for paired fundus image + VF data."""


def separate_datasets(VF_only_dataset, paired_dataset):
    """Splits and returns training and testing datasets"""

# ===========================
# Masking
# ===========================
mask_OD = np.array([
    [False, False, False,  True,  True,  True,  True, False, False],
    [False, False,  True,  True,  True,  True,  True,  True, False],
    [False,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  False,  True],
    [True,  True,  True,  True,  True,  True,  True,  False,  True],
    [False, True,  True,  True,  True,  True,  True,  True,  True],
    [False, False,  True,  True,  True,  True,  True,  True,  False],
    [False, False, False,  True,  True,  True,  True, False, False]
])
mask_OD_flat = torch.tensor(mask_OD.flatten(), dtype=torch.bool)
mask_OS_flat = torch.tensor(mask_OD.flatten()[::-1].copy(), dtype=torch.bool)

# TO DO FOR TRAINING DECODER

# ===========================
# Run Pipeline
# ===========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 1024

    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"

    # Stage 1: Gather data and separate data into training and testing
    uwhvf_json = os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json")
    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    # Stage 2: Train the decoder

    print(f"\nFinal average MAE after cross-validation: {avg_mae:.4f} dB")
