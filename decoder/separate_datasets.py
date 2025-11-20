import os
import json
import random
from torch.utils.data import Dataset


# ===========================
# GRAPE Dataset Class
# ===========================
class GrapeDataset(Dataset):
    """
    GRAPE Dataset after expansion.
    Each entry now has:
        "FundusImage": [list of image filenames]
    """

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
# Splitting helpers
# ===========================
def split_indices(n, test_ratio=0.2, seed=42):
    """Returns (train_idx, test_idx)."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    test_size = int(n * test_ratio)
    return idx[test_size:], idx[:test_size]


def save_split(dataset, train_idx, test_idx, save_dir, prefix):
    """Save train/test JSON files."""
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


# ===========================
# Main Pipeline
# ===========================
def split_grape_dataset(grape_json, grape_fundus_dir, save_base):
    """
    Loads updated GRAPE dataset (with list-of-images),
    splits train/test, saves results.
    """
    grape_dataset = GrapeDataset(grape_json, grape_fundus_dir)

    train_idx, test_idx = split_indices(len(grape_dataset))

    save_dir = os.path.join(save_base, "vf_tests")
    os.makedirs(save_dir, exist_ok=True)

    save_split(grape_dataset, train_idx, test_idx, save_dir, "grape")

    return grape_dataset


# ===========================
# Run
# ===========================
if __name__ == "__main__":
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"

    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    split_grape_dataset(
        grape_json=grape_json,
        grape_fundus_dir=grape_fundus_dir,
        save_base=base_dir
    )
