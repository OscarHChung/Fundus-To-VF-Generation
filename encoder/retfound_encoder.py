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
finetuned_path = os.path.join(current_dir, "best_encoder_finetuned.pth")

# Load base model first
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

base_model = mae_vit_large_patch16_dec512d8b()
base_model.load_state_dict(checkpoint['model'], strict=False)

# Load fine-tuned weights if available
if os.path.exists(finetuned_path):
    print(f"✓ Loading fine-tuned encoder from {finetuned_path}")
    try:
        # Try with weights_only=False for compatibility with numpy types
        finetuned_checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)
        
        # Load only the encoder state (not the prediction head)
        if 'encoder_state' in finetuned_checkpoint:
            base_model.load_state_dict(finetuned_checkpoint['encoder_state'], strict=False)
            print(f"  Fine-tuned MAE: {finetuned_checkpoint['val_mae']:.3f} dB")
            print(f"  Fine-tuned Corr: {finetuned_checkpoint['val_corr']:.3f}")
        else:
            print("  Warning: Could not find 'encoder_state' in checkpoint")
    except Exception as e:
        print(f"  Warning: Could not load fine-tuned weights: {e}")
        print("  Falling back to base RETFound weights")
else:
    print(f"✓ Using base RETFound weights (fine-tuned checkpoint not found)")

base_model.eval()  # Set to eval mode

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
            latent = self.model.forward_encoder(x, mask_ratio=0.0)[0]  # No masking for inference
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
