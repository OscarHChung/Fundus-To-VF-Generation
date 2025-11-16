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

# ===========================
# Device setup
# ===========================
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

# ===========================
# Masking (True = valid, False = invalid)
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
], dtype=bool)

mask_OD_flat = torch.tensor(mask_OD.flatten(), dtype=torch.bool)
mask_OS_flat = torch.tensor(mask_OD.flatten()[::-1].copy(), dtype=torch.bool)

# ===========================
# Datasets
# ===========================
class VFOnlyDataset(Dataset):
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
# Decoder Model
# ===========================
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_dim=72, use_laterality=True):
        super().__init__()
        self.use_laterality = use_laterality
        input_dim = latent_dim + (1 if use_laterality else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, latent, laterality=None):
        if self.use_laterality:
            if laterality is None:
                raise ValueError("Laterality must be provided when use_laterality=True")
            laterality = laterality.float().unsqueeze(1)
            latent = torch.cat([latent, laterality], dim=1)
        return self.net(latent)

# ===========================
# Training
# ===========================
def train_decoder(encoder_model, grape_dataset, uwhvf_dataset,
                  save_path="decoder.pt", latent_dim=1024,
                  epochs=5, batch_size=16, lr=1e-3):

    # Determine output dimension
    sample_item = grape_dataset[0] if isinstance(grape_dataset, PairedDataset) else uwhvf_dataset[0]
    sample_vf = sample_item[1] if isinstance(sample_item, tuple) else sample_item[1]
    decoder_out_dim = sample_vf.shape[0]

    decoder = SimpleDecoder(latent_dim=latent_dim, out_dim=decoder_out_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    grape_loader = DataLoader(grape_dataset, batch_size=batch_size, shuffle=True)
    uwhvf_loader = DataLoader(uwhvf_dataset, batch_size=batch_size, shuffle=True)

    encoder_model.eval()
    decoder.train()

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        grape_loss_total, uwhvf_loss_total = 0.0, 0.0
        grape_count, uwhvf_count = 0, 0

        # --- GRAPE training ---
        for img, vf_values, laterality in grape_loader:
            img, vf_values = img.to(device), vf_values.to(device)
            laterality_tensor = torch.tensor([1 if l == "OD" else 0 for l in laterality],
                                             dtype=torch.float32).to(device)

            with torch.no_grad():
                latent = encoder_model(img)

            pred = decoder(latent, laterality=laterality_tensor)

            # Per-sample masking
            masks = [mask_OD_flat if l=="OD" else mask_OS_flat for l in laterality]
            masks = torch.stack(masks).to(device)
            masked_pred = torch.stack([p[m] for p, m in zip(pred, masks)])
            masked_vf = torch.stack([v[m] for v, m in zip(vf_values, masks)])

            loss = loss_fn(masked_pred, masked_vf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            grape_loss_total += loss.item()
            grape_count += 1

        # --- UWHVF training ---
        for vf_values, laterality in uwhvf_loader:
            vf_values = vf_values.to(device)
            laterality_tensor = torch.tensor([1 if l == "OD" else 0 for l in laterality],
                                             dtype=torch.float32).to(device)

            # Random latent vectors normalized
            latent = torch.randn(vf_values.shape[0], latent_dim).to(device) * 0.1

            pred = decoder(latent, laterality=laterality_tensor)

            masks = [mask_OD_flat if l=="OD" else mask_OS_flat for l in laterality]
            masks = torch.stack(masks).to(device)
            masked_pred = torch.stack([p[m] for p, m in zip(pred, masks)])
            masked_vf = torch.stack([v[m] for v, m in zip(vf_values, masks)])

            loss = loss_fn(masked_pred, masked_vf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            uwhvf_loss_total += loss.item()
            uwhvf_count += 1

        total_batches = grape_count + uwhvf_count
        avg_grape_loss = grape_loss_total / max(grape_count, 1)
        avg_uwhvf_loss = uwhvf_loss_total / max(uwhvf_count, 1)

        # Debug: sample valid VF mean
        with torch.no_grad():
            sample_vf, laterality_sample = grape_dataset[0][1], grape_dataset[0][2]
            mask_sample = mask_OD_flat if laterality_sample=="OD" else mask_OS_flat
            mean_vf_valid = sample_vf[mask_sample].mean().item()

        print(f"[Epoch {epoch}] MAE (dB) = {epoch_loss / total_batches:.4f} | "
              f"Avg GRAPE Loss = {avg_grape_loss:.4f}, Avg UWHVF Loss = {avg_uwhvf_loss:.4f} | "
              f"Sample VF mean (valid) = {mean_vf_valid:.2f}, Valid points = {mask_sample.sum().item()}")

    torch.save(decoder.state_dict(), save_path)
    print(f"Decoder saved to: {save_path}")
    return decoder

# ===========================
# Evaluation
# ===========================
def evaluate_decoder(encoder_model, decoder_model, dataset, batch_size=16):
    decoder_model.eval()
    encoder_model.eval()
    loss_fn = nn.L1Loss()
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(dataset, PairedDataset):
                img, vf_values, laterality = batch
                img, vf_values = img.to(device), vf_values.to(device)
                laterality_tensor = torch.tensor([1 if l=="OD" else 0 for l in laterality], dtype=torch.float32).to(device)
                latent = encoder_model(img)
                pred = decoder_model(latent, laterality=laterality_tensor)
            else:
                vf_values, laterality = batch
                vf_values = vf_values.to(device)
                laterality_tensor = torch.tensor([1 if l=="OD" else 0 for l in laterality], dtype=torch.float32).to(device)
                latent = torch.randn(vf_values.shape[0], 1024).to(device) * 0.1
                pred = decoder_model(latent, laterality=laterality_tensor)

            masks = [mask_OD_flat if l=="OD" else mask_OS_flat for l in laterality]
            masks = torch.stack(masks).to(device)
            masked_pred = torch.stack([p[m] for p, m in zip(pred, masks)])
            masked_vf = torch.stack([v[m] for v, m in zip(vf_values, masks)])

            total_loss += loss_fn(masked_pred, masked_vf).item()
            all_preds.append(masked_pred.cpu())
            all_true.append(masked_vf.cpu())

    mae = total_loss / len(loader)
    print(f"Test MAE (dB) = {mae:.4f}")
    return mae, torch.cat(all_preds), torch.cat(all_true)

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    grape_train = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_train.json"), fundus_dir)
    grape_test = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_test.json"), fundus_dir)
    uwhvf_train = VFOnlyDataset(os.path.join(base_dir, "vf_tests", "uwhvf_vf_tests_standardized.json"))

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from encoder.retfound_encoder import encoder as RetEncoder
    enc_model = RetEncoder.to(device)

    dec_model = train_decoder(
        encoder_model=enc_model,
        grape_dataset=grape_train,
        uwhvf_dataset=uwhvf_train,
        save_path=os.path.join(base_dir, "decoder.pt"),
        epochs=5
    )

    evaluate_decoder(enc_model, dec_model, grape_test)
