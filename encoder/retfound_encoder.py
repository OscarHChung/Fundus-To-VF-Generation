import sys
import os
import torch
import argparse
from PIL import Image
from torchvision import transforms

# Add RETFound_MAE folder to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
retfound_dir = os.path.join(current_dir, 'RETFound_MAE')
print("Appending path:", retfound_dir)
sys.path.insert(0, retfound_dir)

from models_mae import mae_vit_large_patch16_dec512d8b

# Path to checkpoint relative to this script
checkpoint_path = os.path.join(current_dir, "RETFound_cfp_weights.pth")
print(f"Loading checkpoint from: {checkpoint_path}")

# Load checkpoint safely allowing argparse.Namespace
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Initialize model and load weights
model = mae_vit_large_patch16_dec512d8b()
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load image (adjust the relative path if necessary)
img_path = os.path.join(current_dir, "../data/fundus/fundus_example.png")
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0)

# Extract latent representation
with torch.no_grad():
    latent = model.forward_encoder(x, mask_ratio=0.75)[0]  # CLS token (global latent)
    print("Latent vector shape:", latent.shape)
