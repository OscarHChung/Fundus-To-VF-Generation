import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ===========================
# Device
# ===========================
device = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Using device:", device)

# ===========================
# Masking
# ===========================
mask_OD = np.array([
    [False, False, False, True, True, True, True, False, False],
    [False, False, True, True, True, True, True, True, False],
    [False, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, False, True],
    [True, True, True, True, True, True, True, False, True],
    [False, True, True, True, True, True, True, True, True],
    [False, False, True, True, True, True, True, True, False],
    [False, False, False, True, True, True, True, False, False]
])
mask_OD_flat = torch.tensor(mask_OD.flatten(), dtype=torch.bool)
mask_OS_flat = torch.tensor(mask_OD.flatten()[::-1].copy(), dtype=torch.bool)

# ===========================
# Dataset
# ===========================
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, img_size=(224,224)):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["FundusImage"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        vf_values = torch.tensor([v for row in item["hvf"] for v in row], dtype=torch.float32)
        laterality = item.get("Laterality", "OD")
        return img, vf_values, laterality

# ===========================
# Decoder model
# ===========================
class SimpleDecoder(torch.nn.Module):
    def __init__(self, latent_dim=1024, out_dim=72, use_laterality=True):
        super().__init__()
        self.use_laterality = use_laterality
        input_dim = latent_dim + (1 if use_laterality else 0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_dim)
        )

    def forward(self, latent, laterality=None):
        if self.use_laterality:
            laterality = laterality.float().unsqueeze(1)
            latent = torch.cat([latent, laterality], dim=1)
        return self.net(latent)

# ===========================
# Load encoder & decoder
# ===========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoder.retfound_encoder import encoder as RetEncoder

enc_model = RetEncoder.to(device).eval()
decoder = SimpleDecoder().to(device)
decoder.load_state_dict(torch.load("/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data/decoder.pt"))
decoder.eval()

# ===========================
# Load Grape test set
# ===========================
base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")
grape_test = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_test.json"), fundus_dir)
loader = DataLoader(grape_test, batch_size=16)

# ===========================
# Error analysis
# ===========================
all_true = []
all_pred = []

with torch.no_grad():
    for img, vf_values, laterality in loader:
        img, vf_values = img.to(device), vf_values.to(device)
        laterality_tensor = torch.tensor([1 if l=="OD" else 0 for l in laterality], dtype=torch.float32).to(device)
        latent = enc_model(img)
        pred = decoder(latent, laterality=laterality_tensor)

        for i in range(pred.shape[0]):
            mask = mask_OD_flat if laterality[i] == "OD" else mask_OS_flat
            mask = mask.to(device)
            all_true.append(vf_values[i, ~mask].cpu().numpy())
            all_pred.append(pred[i, ~mask].cpu().numpy())

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

# ===========================
# Plot predicted vs true
# ===========================
plt.figure(figsize=(8,8))
plt.scatter(all_true, all_pred, alpha=0.3)
plt.plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'r--')
plt.xlabel("True VF Values")
plt.ylabel("Predicted VF Values")
plt.title("Predicted vs True VF (Masked points ignored)")
plt.grid(True)
plt.show()

# ===========================
# Error histogram
# ===========================
errors = all_pred - all_true
plt.figure(figsize=(8,5))
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel("Prediction Error (Pred - True)")
plt.ylabel("Count")
plt.title("Histogram of VF Prediction Errors")
plt.grid(True)
plt.show()

# ===========================
# Print statistics
# ===========================
print("Prediction Error Mean:", errors.mean())
print("Prediction Error Std:", errors.std())
print("Prediction Error Min/Max:", errors.min(), errors.max())
