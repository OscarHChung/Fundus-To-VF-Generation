import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import argparse

# ===========================
# 1. Load RETFound Model
# ===========================
current_dir = os.path.dirname(os.path.abspath(__file__))
retfound_dir = os.path.join(current_dir, 'RETFound_MAE')
sys.path.insert(0, retfound_dir)

from models_mae import mae_vit_large_patch16_dec512d8b

checkpoint_path = os.path.join(current_dir, "RETFound_cfp_weights.pth")

# Safe loading for checkpoints containing argparse.Namespace
with torch.serialization.safe_globals([argparse.Namespace]):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

model = mae_vit_large_patch16_dec512d8b()
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()  # freeze encoder by default

# ===========================
# 2. Encoder Wrapper
# ===========================
class RetFoundEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        """
        x: tensor [B, 3, 224, 224]
        returns: latent [B, 512] (CLS token)
        """
        with torch.no_grad():
            latent = self.model.forward_encoder(x, mask_ratio=0.75)[0]
        return latent

encoder = RetFoundEncoderWrapper(model)

# ===========================
# 3. Preprocessing Transform
# ===========================
retfound_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===========================
# 4. Example: Encode a single image
# ===========================
if __name__ == "__main__":
    img_path = os.path.join(current_dir, "../data/fundus/fundus_example.png")
    img = Image.open(img_path).convert("RGB")
    x = retfound_transform(img).unsqueeze(0)
    latent = encoder(x)
    print("Latent vector shape:", latent.shape)
