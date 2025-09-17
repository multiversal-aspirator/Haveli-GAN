# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm

from model import Generator, Discriminator, StyleEncoder
from loss import VGGPerceptualLoss
from dataset import FrescoDataset

# Disable anomaly detection now that errors are fixed (for better performance)
# torch.autograd.set_detect_anomaly(True)

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"
LEARNING_RATE_G = 1e-4  # Reduced learning rate for stability
LEARNING_RATE_D = 1e-4  # Reduced learning rate for stability
BATCH_SIZE = 2  # Reduced batch size to avoid memory issues
IMAGE_SIZE = 256
NUM_EPOCHS = 200  # Extended training for better convergence
LAMBDA_ADV = 1
LAMBDA_L1 = 100
LAMBDA_PERC = 10
LAMBDA_STYLE = 250

# --- Setup ---
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Models ---
gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)
style_enc = StyleEncoder().to(DEVICE)
vgg_loss = VGGPerceptualLoss().to(DEVICE)

# --- Optimizers ---
opt_G = optim.Adam(list(gen.parameters()) + list(style_enc.parameters()), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
opt_D = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

# --- Loss Functions ---
bce_loss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()

# --- Dataloader ---
dataset = FrescoDataset(root_dir=DATA_DIR, image_size=IMAGE_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

print(f"Dataset size: {len(dataset)} images")
print(f"Number of batches: {len(loader)}")

# --- Training Loop ---
gen.train()
disc.train()
style_enc.train()

for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for i, batch in enumerate(loop):
        try:
            damaged = batch["damaged"].to(DEVICE, non_blocking=True)
            gt = batch["gt"].to(DEVICE, non_blocking=True)
            mask = batch["mask"].to(DEVICE, non_blocking=True)
            style_ref = batch["style_ref"].to(DEVICE, non_blocking=True)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            
            # Generate fake images (detached to avoid gradients)
            with torch.no_grad():
                style_vector = style_enc(style_ref)
                fake = gen(damaged, mask, style_vector)

            # Discriminator forward pass
            D_real = disc(gt, damaged)
            D_fake = disc(fake.detach(), damaged)
            
            # Discriminator losses
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            
            # Discriminator backward pass
            D_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            
            # Generate completely new fake images for generator training
            # This ensures no shared computation graphs
            style_vector_gen = style_enc(style_ref.clone())
            fake_gen = gen(damaged.clone(), mask.clone(), style_vector_gen)
            
            # Generator losses
            D_fake_gen = disc(fake_gen, damaged.clone())
            G_adv_loss = bce_loss(D_fake_gen, torch.ones_like(D_fake_gen))
            G_l1_loss = l1_loss(fake_gen, gt)
            G_perc_loss, G_style_loss = vgg_loss(fake_gen, gt)
            
            G_loss = (LAMBDA_ADV * G_adv_loss + 
                      LAMBDA_L1 * G_l1_loss +
                      LAMBDA_PERC * G_perc_loss +
                      LAMBDA_STYLE * G_style_loss)

            # Generator backward pass
            G_loss.backward()
            opt_G.step()
            
            loop.set_postfix(
                D_loss=f"{D_loss.item():.4f}",
                G_adv=f"{G_adv_loss.item():.4f}",
                G_l1=f"{G_l1_loss.item():.4f}",
                G_perc=f"{G_perc_loss.item():.4f}",
                G_style=f"{G_style_loss.item():.4f}",
            )
            
        except RuntimeError as e:
            print(f"Error in batch {i}: {e}")
            # Clear gradients and continue
            opt_G.zero_grad()
            opt_D.zero_grad()
            continue

    # Save samples and checkpoints
    if (epoch + 1) % 5 == 0:
        print(f"Saving samples and checkpoint for epoch {epoch+1}...")
        gen.eval()
        with torch.no_grad():
            try:
                # Use a sample from the current epoch
                sample_batch = next(iter(loader))
                sample_damaged = sample_batch["damaged"][:1].to(DEVICE)
                sample_gt = sample_batch["gt"][:1].to(DEVICE)
                sample_mask = sample_batch["mask"][:1].to(DEVICE)
                sample_style_ref = sample_batch["style_ref"][:1].to(DEVICE)
                
                sample_style_vector = style_enc(sample_style_ref)
                sample_fake = gen(sample_damaged, sample_mask, sample_style_vector)
                
                # Save comparison image
                comparison = torch.cat([sample_damaged * 0.5 + 0.5, 
                                      sample_fake * 0.5 + 0.5, 
                                      sample_gt * 0.5 + 0.5], dim=0)
                save_image(comparison, f"{OUTPUT_DIR}/comparison_epoch_{epoch+1}.png", nrow=1)
                save_image(sample_fake * 0.5 + 0.5, f"{OUTPUT_DIR}/sample_{epoch+1}.png")
                
            except Exception as e:
                print(f"Error saving samples: {e}")
        
        # Save model checkpoints
        torch.save({
            'epoch': epoch + 1,
            'gen_state_dict': gen.state_dict(),
            'style_enc_state_dict': style_enc.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_G_state_dict': opt_G.state_dict(),
            'opt_D_state_dict': opt_D.state_dict(),
        }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth")
        
        gen.train()

print("✅ Training finished!")
for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader, leave=True)
    for i, batch in enumerate(loop):
        damaged = batch["damaged"].to(DEVICE)
        gt = batch["gt"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        style_ref = batch["style_ref"].to(DEVICE)

        # --- Train Discriminator ---
        with torch.no_grad():
            style_vector = style_enc(style_ref)
            fake = gen(damaged, mask, style_vector)

        D_real = disc(gt, damaged)
        D_fake = disc(fake.detach(), damaged)
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
        
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # --- Train Generator ---
        style_vector = style_enc(style_ref)
        fake = gen(damaged, mask, style_vector)
        D_fake_gen = disc(fake, damaged)
        
        G_adv_loss = bce_loss(D_fake_gen, torch.ones_like(D_fake_gen))
        G_l1_loss = l1_loss(fake, gt)
        G_perc_loss, G_style_loss = vgg_loss(fake, gt)
        
        G_loss = (LAMBDA_ADV * G_adv_loss + 
                  LAMBDA_L1 * G_l1_loss +
                  LAMBDA_PERC * G_perc_loss +
                  LAMBDA_STYLE * G_style_loss)

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()
        
        loop.set_postfix(
            epoch=epoch+1,
            D_loss=D_loss.item(),
            G_adv=G_adv_loss.item(),
            G_l1=G_l1_loss.item(),
            G_perc=G_perc_loss.item(),
            G_style=G_style_loss.item(),
        )

    # Save samples and checkpoints
    if (epoch + 1) % 5 == 0:
        save_image(fake * 0.5 + 0.5, f"{OUTPUT_DIR}/sample_{epoch+1}.png")
        torch.save(gen.state_dict(), f"{CHECKPOINT_DIR}/gen_epoch_{epoch+1}.pth")
        torch.save(style_enc.state_dict(), f"{CHECKPOINT_DIR}/style_enc_epoch_{epoch+1}.pth")

print("✅ Training finished!")