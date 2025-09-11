# inference.py
import torch
from torchvision import transforms
from PIL import Image
import os
from model import Generator, StyleEncoder

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_CHECKPOINT = "./checkpoints/gen_epoch_100.pth" # Path to your trained generator
STYLE_ENC_CHECKPOINT = "./checkpoints/style_enc_epoch_100.pth" # Path to your trained style encoder

DAMAGED_IMG_PATH = "./path/to/your/damaged_image.jpg"
MASK_PATH = "./path/to/your/damage_mask.png"
STYLE_REF_PATH = "./path/to/your/style_reference.jpg"
OUTPUT_PATH = "./restored_output.png"

# --- Load Models ---
gen = Generator().to(DEVICE)
gen.load_state_dict(torch.load(GEN_CHECKPOINT, map_location=DEVICE))
gen.eval()

style_enc = StyleEncoder().to(DEVICE)
style_enc.load_state_dict(torch.load(STYLE_ENC_CHECKPOINT, map_location=DEVICE))
style_enc.eval()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- Load and Transform Images ---
damaged_img = transform(Image.open(DAMAGED_IMG_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)
mask = mask_transform(Image.open(MASK_PATH).convert("L")).unsqueeze(0).to(DEVICE)
style_ref = transform(Image.open(STYLE_REF_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)

# --- Perform Restoration ---
print("Restoring image...")
with torch.no_grad():
    style_vector = style_enc(style_ref)
    restored_img = gen(damaged_img, mask, style_vector)

# --- Save Output ---
# De-normalize from [-1, 1] to [0, 1]
restored_img = restored_img * 0.5 + 0.5 
transforms.ToPILImage()(restored_img.squeeze(0).cpu()).save(OUTPUT_PATH)
print(f"✅ Restoration complete! Image saved to {OUTPUT_PATH}")