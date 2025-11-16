import os
from torch.utils.data import Dataset
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
# Run Pipeline
# ===========================
if __name__ == "__main__":
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"

    # Separate data into training and testing
    uwhvf_json = os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json")
    grape_json = os.path.join(base_dir, "vf_tests", "grape_new_vf_tests.json")
    grape_fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

