#!/usr/bin/env python3
"""
Test the prepared Indian Paintings dataset with Haveli-GAN model
"""

import torch
from torch.utils.data import DataLoader
from dataset import FrescoDataset
from model import Generator, Discriminator, StyleEncoder
import matplotlib.pyplot as plt
import numpy as np
import os

def denormalize(tensor):
    """Convert normalized tensor back to image range [0, 255]"""
    return ((tensor * 0.5 + 0.5) * 255).clamp(0, 255).byte()

def test_dataset_loading():
    """Test if the dataset loads correctly"""
    print("Testing dataset loading...")
    
    # Create dataset
    dataset = FrescoDataset('data', image_size=256)
    print(f"Dataset size: {len(dataset)} images")
    
    # Create dataloader  
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test loading a batch
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Damaged images: {batch['damaged'].shape}")
        print(f"  Ground truth:   {batch['gt'].shape}")
        print(f"  Masks:          {batch['mask'].shape}")
        print(f"  Style ref:      {batch['style_ref'].shape}")
        
        # Show sample images
        if i == 0:
            visualize_batch(batch)
        
        if i >= 2:  # Test only first 3 batches
            break
    
    print("✅ Dataset loading test passed!")
    return True

def visualize_batch(batch):
    """Visualize a batch of training data"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(4):
        # Damaged image
        damaged = denormalize(batch['damaged'][i]).permute(1, 2, 0).numpy()
        axes[0, i].imshow(damaged)
        axes[0, i].set_title(f'Damaged {i+1}')
        axes[0, i].axis('off')
        
        # Ground truth
        gt = denormalize(batch['gt'][i]).permute(1, 2, 0).numpy()
        axes[1, i].imshow(gt)
        axes[1, i].set_title(f'Ground Truth {i+1}')
        axes[1, i].axis('off')
        
        # Mask  
        mask = (batch['mask'][i].squeeze() * 255).numpy().astype(np.uint8)
        axes[2, i].imshow(mask, cmap='gray')
        axes[2, i].set_title(f'Mask {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("Sample images saved as 'dataset_samples.png'")
    plt.close()

def test_model_forward_pass():
    """Test forward pass through all models"""
    print("\nTesting model forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    style_encoder = StyleEncoder().to(device)
    
    # Create dummy batch
    batch_size = 2
    damaged = torch.randn(batch_size, 3, 256, 256).to(device)
    gt = torch.randn(batch_size, 3, 256, 256).to(device)
    mask = torch.randn(batch_size, 1, 256, 256).to(device)
    style_ref = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"Input shapes:")
    print(f"  Damaged: {damaged.shape}")
    print(f"  GT:      {gt.shape}")
    print(f"  Mask:    {mask.shape}")
    print(f"  Style:   {style_ref.shape}")
    
    # Test Generator
    with torch.no_grad():
        style_code = style_encoder(style_ref)
        restored = generator(damaged, mask, style_code)
        print(f"Generator output: {restored.shape}")
        
        # Test Discriminator
        disc_real = discriminator(gt, damaged)  # Real image vs damaged
        disc_fake = discriminator(restored, damaged)  # Fake image vs damaged
        print(f"Discriminator real: {disc_real.shape}")
        print(f"Discriminator fake: {disc_fake.shape}")
    
    print("✅ Model forward pass test passed!")
    return True

def test_training_integration():
    """Test integration with actual dataset"""
    print("\nTesting training integration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = FrescoDataset('data', image_size=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize models
    generator = Generator().to(device)
    style_encoder = StyleEncoder().to(device)
    
    # Test with real data
    for batch in dataloader:
        damaged = batch['damaged'].to(device)
        gt = batch['gt'].to(device)
        mask = batch['mask'].to(device)
        style_ref = batch['style_ref'].to(device)
        
        print(f"Real data shapes:")
        print(f"  Damaged: {damaged.shape}")
        print(f"  GT:      {gt.shape}")
        print(f"  Mask:    {mask.shape}")
        print(f"  Style:   {style_ref.shape}")
        
        with torch.no_grad():
            style_code = style_encoder(style_ref)
            restored = generator(damaged, mask, style_code)
            print(f"Restored: {restored.shape}")
        
        break  # Test only first batch
    
    print("✅ Training integration test passed!")
    return True

def main():
    print("=" * 60)
    print("HAVELI-GAN DATASET TESTING")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ Error: data directory not found!")
        print("Please run prepare_indian_paintings_dataset.py first")
        return 1
    
    try:
        # Test dataset loading
        test_dataset_loading()
        
        # Test model forward pass
        test_model_forward_pass()
        
        # Test training integration
        test_training_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("Your Indian Paintings dataset is ready for training!")
        print("Run: python train.py --data_dir data --epochs 100")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
