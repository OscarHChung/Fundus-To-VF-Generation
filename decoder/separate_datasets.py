# SEPARATE ALL DATA INTO TRAINING AND TESTING
# GRAPE + UWHVF => Training + Testing
import os
import json
import random
from torch.utils.data import Dataset


# ===========================
# Dataset Classes
# ===========================
class VFOnlyDataset(Dataset):
    """UWHVF Dataset containing VF tests-only."""

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Store as list or dict
        self.data = data if isinstance(data, list) else list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PairedDataset(Dataset):
    """GRAPE Dataset for paired fundus image + VF tests."""

    def __init__(self, json_path, fundus_dir):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Store as list or dict
        self.data = data if isinstance(data, list) else list(data.values())
        self.fundus_dir = fundus_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ===========================
# Splitting + Saving
# ===========================
def split_indices(n, test_ratio=0.2, seed=42):
    """Returns train indices, test indices."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    
    test_size = int(n * test_ratio)
    return idx[test_size:], idx[:test_size]   # train, test


def save_split(dataset, train_idx, test_idx, save_dir, prefix):
    """Writes JSON files for training and testing splits."""
    train_list = [dataset[i] for i in train_idx]
    test_list = [dataset[i] for i in test_idx]

    train_path = os.path.join(save_dir, f"{prefix}_train.json")
    test_path = os.path.join(save_dir, f"{prefix}_test.json")

    with open(train_path, "w") as f:
        json.dump(train_list, f, indent=2)

    with open(test_path, "w") as f:
        json.dump(test_list, f, indent=2)

    print(f"Saved: {train_path} ({len(train_list)} samples)")
    print(f"Saved: {test_path} ({len(test_list)} samples)")


def separate_datasets(uwhvf_path, grape_path, grape_fundus_dir, save_base):
    """Loads, splits, saves, and returns dataset objects."""

    # --- Load ---
    uwhvf_full = VFOnlyDataset(uwhvf_path)
    grape_full = PairedDataset(grape_path, grape_fundus_dir)

    # --- Split ---
    uwhvf_train_idx, uwhvf_test_idx = split_indices(len(uwhvf_full))
    grape_train_idx, grape_test_idx = split_indices(len(grape_full))

    # --- Save inside data/vf_tests/ ---
    save_dir = os.path.join(save_base, "vf_tests")
    save_split(uwhvf_full, uwhvf_train_idx, uwhvf_test_idx, save_dir, "uwhvf")
    save_split(grape_full, grape_train_idx, grape_test_idx, save_dir, "grape")

    return uwhvf_full, grape_full


# ===========================
# Run Pipeline
# ===========================
if __name__ == "__main__":
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"

    uwhvf_json = os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json")
    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    separate_datasets(
        uwhvf_path=uwhvf_json,
        grape_path=grape_json,
        grape_fundus_dir=grape_fundus_dir,
        save_base=base_dir
    )
