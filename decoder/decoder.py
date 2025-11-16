import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# =============================
# Device setup
# =============================
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS backend.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = get_device()

# =============================
# Masking for OD and OS
# =============================
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

def get_mask_by_laterality(laterality):
    if laterality == "OD":
        return mask_OD_flat.to(device)
    else:
        return mask_OS_flat.to(device)

# =============================
# Datasets
# =============================
class VFOnlyDataset(Dataset):
    """VF-only dataset (UWHVF)"""
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vf_values = torch.tensor([v for row in item["hvf"] for v in row], dtype=torch.float32)
        laterality = item.get("Laterality", "OD")
        return vf_values, laterality

class PairedDataset(Dataset):
    """GRAPE fundus + VF paired dataset"""
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

# =============================
# Decoder model
# =============================
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_dim=72):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# =============================
# Training function
# =============================
def train_decoder(encoder_model, grape_train_dataset, uwhvf_dataset,
                  save_path="decoder.pt", latent_dim=1024, epochs=5, batch_size=16, lr=1e-4):
    
    # Determine decoder output dim from a sample VF
    _, sample_vf, _ = grape_train_dataset[0]
    decoder_out_dim = sample_vf.shape[0]

    decoder = SimpleDecoder(latent_dim=latent_dim, out_dim=decoder_out_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    grape_loader = DataLoader(grape_train_dataset, batch_size=batch_size, shuffle=True)
    uwhvf_loader = DataLoader(uwhvf_dataset, batch_size=batch_size, shuffle=True)

    encoder_model.eval()
    decoder.train()

    for epoch in range(1, epochs+1):
        epoch_loss = 0
        total_batches = len(grape_loader) + len(uwhvf_loader)

        # --- GRAPE paired dataset ---
        for img, vf_values, laterality in tqdm(grape_loader, desc=f"Epoch {epoch} (GRAPE)"):
            img, vf_values = img.to(device), vf_values.to(device)
            mask = torch.stack([get_mask_by_laterality(lat) for lat in laterality])
            with torch.no_grad():
                latent = encoder_model(img)
            pred = decoder(latent)
            loss = loss_fn(pred[~mask], vf_values[~mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # --- UWHVF VF-only dataset ---
        for vf_values, laterality in tqdm(uwhvf_loader, desc=f"Epoch {epoch} (UWHVF)"):
            vf_values = vf_values.to(device)
            mask = torch.stack([get_mask_by_laterality(lat) for lat in laterality])
            latent = torch.randn(vf_values.shape[0], latent_dim).to(device)
            pred = decoder(latent)
            loss = loss_fn(pred[~mask], vf_values[~mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch} MAE = {epoch_loss / total_batches:.4f}")

        # Debug prints
        with torch.no_grad():
            print("Latent min/max/mean:", latent.min().item(), latent.max().item(), latent.mean().item())
            print("Latent variance:", latent.var().item())
            print("Prediction min/max/mean:", pred.min().item(), pred.max().item(), pred.mean().item())
            print("Prediction variance:", pred.var().item())

    torch.save(decoder.state_dict(), save_path)
    print(f"Decoder saved to {save_path}")
    return decoder

# =============================
# Evaluation
# =============================
def evaluate_decoder(encoder_model, decoder_model, test_dataset):
    decoder_model.eval()
    encoder_model.eval()
    loss_fn = nn.L1Loss()
    loader = DataLoader(test_dataset, batch_size=16)
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            img, vf_values, laterality = batch
            img, vf_values = img.to(device), vf_values.to(device)
            mask = torch.stack([get_mask_by_laterality(lat) for lat in laterality])
            latent = encoder_model(img)
            pred = decoder_model(latent)
            total_loss += loss_fn(pred[~mask], vf_values[~mask]).item()

    mae = total_loss / len(loader)
    print(f"Test MAE: {mae:.4f}")
    return mae

# =============================
# Run pipeline
# =============================
if __name__ == "__main__":
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    # --- Load datasets ---
    grape_train = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_train.json"), fundus_dir)
    grape_test  = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_test.json"), fundus_dir)
    uwhvf_dataset = VFOnlyDataset(os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json"))

    # --- Import RETFound encoder ---
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from encoder.retfound_encoder import encoder as RetEncoder

    enc_model = RetEncoder.to(device)

    # --- Train decoder ---
    dec_model = train_decoder(
        encoder_model=enc_model,
        grape_train_dataset=grape_train,
        uwhvf_dataset=uwhvf_dataset,
        save_path=os.path.join(base_dir, "decoder.pt"),
        epochs=5
    )

    # --- Evaluate ---
    evaluate_decoder(enc_model, dec_model, grape_test)
