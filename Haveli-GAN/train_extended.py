#!/usr/bin/env python3
"""
Resume Training Script for Haveli-GAN
=====================================

This script allows you to resume training from an existing checkpoint
or start fresh training for 200 epochs.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm
import argparse

from model import Generator, Discriminator, StyleEncoder
from loss import VGGPerceptualLoss
from dataset import FrescoDataset

def main():
    parser = argparse.ArgumentParser(description='Resume Haveli-GAN Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., ./checkpoints/checkpoint_epoch_100.pth)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch (automatically set if resuming)')
    
    args = parser.parse_args()
    
    # --- Hyperparameters ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    DATA_DIR = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    OUTPUT_DIR = "./outputs"
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 1e-4
    BATCH_SIZE = 2
    IMAGE_SIZE = 256
    NUM_EPOCHS = args.epochs
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
    opt_G = optim.Adam(list(gen.parameters()) + list(style_enc.parameters()), 
                       lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    
    # --- Loss Functions ---
    bce_loss = torch.nn.BCEWithLogitsLoss()
    l1_loss = torch.nn.L1Loss()
    
    # --- Resume from checkpoint if specified ---
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        style_enc.load_state_dict(checkpoint['style_enc_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed from epoch {start_epoch}")
        
        # Adjust learning rates for extended training
        if start_epoch >= 100:
            new_lr_g = LEARNING_RATE_G * 0.5  # Reduce learning rate for fine-tuning
            new_lr_d = LEARNING_RATE_D * 0.5
            
            for param_group in opt_G.param_groups:
                param_group['lr'] = new_lr_g
            for param_group in opt_D.param_groups:
                param_group['lr'] = new_lr_d
                
            print(f"📉 Reduced learning rates - G: {new_lr_g}, D: {new_lr_d}")
    else:
        print("🚀 Starting fresh training")
    
    # --- Dataloader ---
    dataset = FrescoDataset(root_dir=DATA_DIR, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of batches: {len(loader)}")
    print(f"Training from epoch {start_epoch + 1} to {NUM_EPOCHS}")
    
    # --- Training Loop ---
    gen.train()
    disc.train()
    style_enc.train()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_g_adv = 0
        epoch_g_l1 = 0
        epoch_g_perc = 0
        epoch_g_style = 0
        
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
                
                # Track losses
                epoch_d_loss += D_loss.item()
                epoch_g_loss += G_loss.item()
                epoch_g_adv += G_adv_loss.item()
                epoch_g_l1 += G_l1_loss.item()
                epoch_g_perc += G_perc_loss.item()
                epoch_g_style += G_style_loss.item()
                
                loop.set_postfix(
                    D_loss=f"{D_loss.item():.4f}",
                    G_adv=f"{G_adv_loss.item():.4f}",
                    G_l1=f"{G_l1_loss.item():.4f}",
                    G_perc=f"{G_perc_loss.item():.4f}",
                    G_style=f"{G_style_loss.item():.4f}",
                )
                
            except RuntimeError as e:
                print(f"Error in batch {i}: {e}")
                opt_G.zero_grad()
                opt_D.zero_grad()
                continue

        # Calculate average losses for the epoch
        num_batches = len(loader)
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_l1 = epoch_g_l1 / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  D Loss: {avg_d_loss:.4f}")
        print(f"  G L1:   {avg_g_l1:.4f}")
        
        # Save samples and checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Saving samples and checkpoint for epoch {epoch+1}...")
            gen.eval()
            with torch.no_grad():
                try:
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
                'avg_d_loss': avg_d_loss,
                'avg_g_l1': avg_g_l1,
            }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth")
            
            gen.train()
        
        # Additional checkpoint every 25 epochs for long training
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch + 1,
                'gen_state_dict': gen.state_dict(),
                'style_enc_state_dict': style_enc.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'avg_d_loss': avg_d_loss,
                'avg_g_l1': avg_g_l1,
            }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}_milestone.pth")
            print(f"🎯 Milestone checkpoint saved at epoch {epoch+1}")

    print("✅ Training finished!")
    print(f"🎉 Completed {NUM_EPOCHS} epochs successfully!")

if __name__ == "__main__":
    main()
