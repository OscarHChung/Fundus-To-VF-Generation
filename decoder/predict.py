import torch
from PIL import Image
from decoder import SimpleDecoder
from encoder.retfound_encoder import encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load encoder + decoder
enc = encoder().to(device)
dec = SimpleDecoder().to(device)
dec.load_state_dict(torch.load("decoder.pt", map_location=device))
dec.eval()

def predict_vf(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = enc(img)
        pred_vf = dec(latent)

    return pred_vf.cpu().numpy().tolist()
