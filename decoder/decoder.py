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

# =====================================================
# Device setup
# =====================================================
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

# =====================================================
# Masking
# =====================================================
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
mask_OS_flat = torch.tensor(mask_OD.flatten()[::-1].copy(), dtype=torch.bool)  # flipped for OS

# =====================================================
# Datasets
# =====================================================
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

# =====================================================
# Decoder
# =====================================================
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_dim=72, laterality_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + laterality_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, latent, laterality_onehot):
        x = torch.cat([latent, laterality_onehot], dim=1)
        return self.net(x)

# =====================================================
# Training
# =====================================================
def train_decoder(encoder_model, grape_train_dataset, uwhvf_train_dataset,
                  save_path="decoder.pt", latent_dim=1024, epochs=5, batch_size=16, lr=1e-4):

    # Infer output dimension from first sample
    sample_vf, laterality = grape_train_dataset[0][1], grape_train_dataset[0][2]
    decoder_out_dim = sample_vf.shape[0]

    decoder = SimpleDecoder(latent_dim=latent_dim, out_dim=decoder_out_dim).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    grape_loader = DataLoader(grape_train_dataset, batch_size=batch_size, shuffle=True)
    uwhvf_loader = DataLoader(uwhvf_train_dataset, batch_size=batch_size, shuffle=True)

    encoder_model.eval()
    decoder.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        total_batches = len(grape_loader) + len(uwhvf_loader)

        # --- GRAPE paired images ---
        for img, vf_values, laterality in tqdm(grape_loader, desc=f"Epoch {epoch} (GRAPE)"):
            img, vf_values = img.to(device), vf_values.to(device)
            mask = mask_OD_flat if laterality[0] == "OD" else mask_OS_flat
            mask = mask.to(device)

            with torch.no_grad():
                latent = encoder_model(img)

            # Laterality one-hot
            laterality_tensor = torch.tensor([[1,0]] if laterality[0]=="OD" else [[0,1]], dtype=torch.float32).to(device)
            laterality_tensor = laterality_tensor.repeat(latent.size(0), 1)

            pred = decoder(latent, laterality_tensor)

            # Apply mask: only non-100 values
            loss = loss_fn(pred[:, ~mask], vf_values[:, ~mask])

            # Debug prints
            if torch.rand(1) < 0.05:  # print ~5% of batches
                print(f"Batch laterality: {laterality[0]}")
                print(f"Masked VF values (min/max/mean): {vf_values[:, ~mask].min().item()}/{vf_values[:, ~mask].max().item()}/{vf_values[:, ~mask].mean().item()}")
                print(f"Pred values (min/max/mean): {pred[:, ~mask].min().item()}/{pred[:, ~mask].max().item()}/{pred[:, ~mask].mean().item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # --- UWHVF VF-only ---
        for vf_values, laterality in tqdm(uwhvf_loader, desc=f"Epoch {epoch} (UWHVF)"):
            vf_values = vf_values.to(device)
            mask = mask_OD_flat if laterality[0]=="OD" else mask_OS_flat
            mask = mask.to(device)

            latent = torch.randn(vf_values.shape[0], latent_dim).to(device)
            laterality_tensor = torch.tensor([[1,0]] if laterality[0]=="OD" else [[0,1]], dtype=torch.float32).to(device)
            laterality_tensor = laterality_tensor.repeat(latent.size(0),1)

            pred = decoder(latent, laterality_tensor)
            loss = loss_fn(pred[:, ~mask], vf_values[:, ~mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch} MAE = {epoch_loss / total_batches:.4f}")

    torch.save(decoder.state_dict(), save_path)
    print(f"Decoder saved to: {save_path}")
    return decoder

# =====================================================
# Evaluation
# =====================================================
def evaluate_decoder(encoder_model, decoder_model, test_dataset):
    decoder_model.eval()
    encoder_model.eval()
    loss_fn = nn.L1Loss()
    loader = DataLoader(test_dataset, batch_size=16)
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(test_dataset, PairedDataset):
                img, vf_values, laterality = batch
                img, vf_values = img.to(device), vf_values.to(device)
                mask = mask_OD_flat if laterality[0]=="OD" else mask_OS_flat
                mask = mask.to(device)

                latent = encoder_model(img)
                laterality_tensor = torch.tensor([[1,0]] if laterality[0]=="OD" else [[0,1]], dtype=torch.float32).to(device)
                laterality_tensor = laterality_tensor.repeat(latent.size(0),1)
            else:
                vf_values, laterality = batch
                vf_values = vf_values.to(device)
                mask = mask_OD_flat if laterality[0]=="OD" else mask_OS_flat
                mask = mask.to(device)

                latent = torch.randn(vf_values.shape[0], 1024).to(device)
                laterality_tensor = torch.tensor([[1,0]] if laterality[0]=="OD" else [[0,1]], dtype=torch.float32).to(device)
                laterality_tensor = laterality_tensor.repeat(latent.size(0),1)

            pred = decoder_model(latent, laterality_tensor)
            total_loss += loss_fn(pred[:, ~mask], vf_values[:, ~mask]).item()

    mae = total_loss / len(loader)
    print(f"Test MAE: {mae:.4f}")
    return mae

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    base_dir = "/Users/oscarchung/Documents/Python Projects/Fundus-To-VF-Generation/data"
    fundus_dir = os.path.join(base_dir, "fundus", "grape_fundus_images")

    grape_train = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_train.json"), fundus_dir)
    uwhvf_train = VFOnlyDataset(os.path.join(base_dir, "vf_tests", "uwhvf_train.json"))

    grape_test = PairedDataset(os.path.join(base_dir, "vf_tests", "grape_test.json"), fundus_dir)
    uwhvf_test = VFOnlyDataset(os.path.join(base_dir, "vf_tests", "uwhvf_test.json"))

    # Load safe encoder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from encoder.retfound_encoder import encoder as RetEncoder
    enc_model = RetEncoder.to(device)

    # Train decoder
    dec_model = train_decoder(
        encoder_model=enc_model,
        grape_train_dataset=grape_train,
        uwhvf_train_dataset=uwhvf_train,
        save_path=os.path.join(base_dir, "decoder.pt"),
        epochs=5
    )

    # Evaluate
    evaluate_decoder(enc_model, dec_model, grape_test)
    evaluate_decoder(enc_model, dec_model, uwhvf_test)
