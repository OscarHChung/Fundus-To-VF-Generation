import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

# =====================================================
# 1. Load RETFound model
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
retfound_dir = os.path.join(current_dir, 'RETFound_MAE')
sys.path.insert(0, retfound_dir)

from models_mae import mae_vit_large_patch16_dec512d8b

checkpoint_path = os.path.join(current_dir, "RETFound_cfp_weights.pth")

# Safe loading for checkpoints containing argparse.Namespace
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Instantiate base model
base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)
base_model.eval()  # freeze by default

# =====================================================
# 2. Encoder Wrapper
# =====================================================
class RetFoundEncoderWrapper(nn.Module):
    def __init__(self, model, latent_dim=1024):
        super().__init__()
        self.model = model
        self.latent_dim = latent_dim
        self.model.eval()

    def forward(self, x):
        """
        x: tensor [B, 3, H, W]
        returns: latent [B, latent_dim] (CLS token)
        """
        x = x.to(next(self.model.parameters()).device)
        with torch.no_grad():
            latent = self.model.forward_encoder(x, mask_ratio=0.75)[0]  # [B, num_patches, latent_dim]
            if latent.dim() == 3:
                latent = latent[:, 0, :]  # take CLS token
        return latent


# =====================================================
# 3. Preprocessing Transform
# =====================================================
retfound_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Instantiate encoder
encoder = RetFoundEncoderWrapper(base_model)
