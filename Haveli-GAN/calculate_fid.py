#!/usr/bin/env python3
"""
FID calculation for Haveli-GAN using a simplified approach
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg
import os
from tqdm import tqdm

from model import Generator, StyleEncoder
from dataset import FrescoDataset

class SimpleFIDCalculator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load models
        self.generator = Generator().to(device)
        self.style_encoder = StyleEncoder().to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(checkpoint['gen_state_dict'])
        self.style_encoder.load_state_dict(checkpoint['style_enc_state_dict'])
        
        # Set to evaluation mode
        self.generator.eval()
        self.style_encoder.eval()
        
        # Load Inception model for FID
        self.inception = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        self.inception.fc = torch.nn.Identity()  # Remove classification layer
        self.inception = self.inception.to(device)
        self.inception.eval()
        
        print("✅ Models loaded successfully!")
    
    def get_inception_features(self, images):
        """Extract features from Inception network"""
        # Resize to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features.cpu().numpy()
    
    def calculate_fid(self, real_features, fake_features):
        """Calculate FID between real and fake features"""
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        # Calculate sqrt of product between cov
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Check for imaginary numbers
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
    def evaluate_fid(self, dataset_path, num_samples=100, batch_size=4):
        """Calculate FID between ground truth and restored images"""
        
        # Load dataset
        dataset = FrescoDataset(root_dir=dataset_path, image_size=256)
        
        if num_samples:
            dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Calculating FID on {len(dataset)} images...")
        
        real_features = []
        fake_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                damaged = batch["damaged"].to(self.device)
                gt = batch["gt"].to(self.device)
                mask = batch["mask"].to(self.device)
                style_ref = batch["style_ref"].to(self.device)
                
                # Generate restored images
                style_vector = self.style_encoder(style_ref)
                restored = self.generator(damaged, mask, style_vector)
                
                # Normalize images to [0, 1] for Inception
                gt_norm = (gt + 1.0) / 2.0
                restored_norm = (restored + 1.0) / 2.0
                
                # Extract features
                real_feat = self.get_inception_features(gt_norm)
                fake_feat = self.get_inception_features(restored_norm)
                
                real_features.append(real_feat)
                fake_features.append(fake_feat)
        
        # Combine all features
        real_features = np.concatenate(real_features, axis=0)
        fake_features = np.concatenate(fake_features, axis=0)
        
        # Calculate FID
        fid_score = self.calculate_fid(real_features, fake_features)
        
        return fid_score, real_features.shape[0]

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate FID for Haveli-GAN')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_epoch_100.pth')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    calculator = SimpleFIDCalculator(args.checkpoint, args.device)
    fid_score, num_images = calculator.evaluate_fid(
        args.data_dir, 
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    print(f"\n🎯 FID CALCULATION RESULTS")
    print(f"=" * 40)
    print(f"FID Score: {fid_score:.2f}")
    print(f"Images evaluated: {num_images}")
    print(f"\n📈 FID INTERPRETATION:")
    print(f"├── Lower FID = Better similarity to real images")
    print(f"├── FID < 10  = Excellent quality")
    print(f"├── FID < 50  = Good quality")
    print(f"├── FID < 100 = Acceptable quality")
    print(f"└── Your FID: {fid_score:.1f} ({'Excellent' if fid_score < 10 else 'Good' if fid_score < 50 else 'Acceptable' if fid_score < 100 else 'Needs improvement'})")
    
    # Save results
    with open("fid_results.txt", "w") as f:
        f.write(f"FID Score: {fid_score:.6f}\n")
        f.write(f"Number of images: {num_images}\n")
    
    print(f"\n💾 FID results saved to 'fid_results.txt'")

if __name__ == "__main__":
    main()
