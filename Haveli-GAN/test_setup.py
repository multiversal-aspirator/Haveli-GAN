#!/usr/bin/env python3
"""
Test script for Haveli-GAN environment setup
This script tests that all components work correctly with CUDA 12.8
"""
import torch
import torch.nn as nn
from model import Generator, Discriminator, StyleEncoder
from loss import VGGPerceptualLoss

def test_environment():
    print("🔧 Testing Haveli-GAN Environment Setup")
    print("=" * 50)
    
    # Test CUDA
    print(f"✅ Python version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Test model instantiation
    print("\n🏗️  Testing Model Instantiation...")
    try:
        gen = Generator().to(device)
        disc = Discriminator().to(device)
        style_enc = StyleEncoder().to(device)
        vgg_loss = VGGPerceptualLoss().to(device)
        print("✅ All models instantiated successfully")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return False
    
    # Test forward pass with dummy data
    print("\n🚀 Testing Forward Pass...")
    try:
        batch_size = 2
        img_size = 256
        
        # Create dummy tensors
        damaged = torch.randn(batch_size, 3, img_size, img_size).to(device)
        mask = torch.randn(batch_size, 1, img_size, img_size).to(device)
        style_ref = torch.randn(batch_size, 3, img_size, img_size).to(device)
        gt = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        # Test style encoder
        style_vector = style_enc(style_ref)
        print(f"✅ Style encoder output shape: {style_vector.shape}")
        
        # Test generator
        fake = gen(damaged, mask, style_vector)
        print(f"✅ Generator output shape: {fake.shape}")
        
        # Test discriminator
        d_real = disc(gt, damaged)
        d_fake = disc(fake, damaged)
        print(f"✅ Discriminator output shape: {d_real.shape}")
        
        # Test VGG loss
        perc_loss, style_loss = vgg_loss(fake, gt)
        print(f"✅ VGG perceptual loss: {perc_loss.item():.4f}")
        print(f"✅ VGG style loss: {style_loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Your environment is ready for training.")
    return True

if __name__ == "__main__":
    success = test_environment()
    exit(0 if success else 1)
