#!/usr/bin/env python3
"""
Comprehensive evaluation script for Haveli-GAN
Calculates: MSE, PSNR, SSIM, Perceptual Loss, FID, IS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from tqdm import tqdm
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg

from model import Generator, StyleEncoder
from loss import VGGPerceptualLoss
from dataset import FrescoDataset

class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 for FID and IS calculation"""
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        self.model = inception
        self.model.eval()
        
    def forward(self, x):
        # Resize to 299x299 for InceptionV3
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize to [-1, 1] range expected by InceptionV3
        x = (x - 0.5) * 2.0
        
        # Get features before the final classification layer
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = self.model.avgpool(x)
        x = self.model.dropout(x)
        x = torch.flatten(x, 1)
        
        return x

def calculate_fid(real_features, fake_features):
    """Calculate Frechet Inception Distance"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_inception_score(features, splits=10, device='cuda'):
    """Calculate Inception Score"""
    # Get predictions from features
    inception = inception_v3(weights='IMAGENET1K_V1')
    inception.eval()
    inception = inception.to(device)
    
    with torch.no_grad():
        preds = inception.fc(features)
        preds = F.softmax(preds, dim=1)
    
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
        kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0, keepdim=True)))
        kl = torch.mean(torch.sum(kl, dim=1))
        scores.append(torch.exp(kl))
    
    return torch.mean(torch.stack(scores)), torch.std(torch.stack(scores))

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for SSIM/PSNR calculation"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and rearrange dimensions
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    
    return tensor

class ModelEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load models
        self.generator = Generator().to(device)
        self.style_encoder = StyleEncoder().to(device)
        self.vgg_loss = VGGPerceptualLoss().to(device)
        self.inception_extractor = InceptionV3FeatureExtractor().to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(checkpoint['gen_state_dict'])
        self.style_encoder.load_state_dict(checkpoint['style_enc_state_dict'])
        
        # Set to evaluation mode
        self.generator.eval()
        self.style_encoder.eval()
        
        print("✅ Models loaded successfully!")
        
    def evaluate_dataset(self, dataset_path, batch_size=4, num_samples=None):
        """Comprehensive evaluation on dataset"""
        
        # Load dataset
        dataset = FrescoDataset(root_dir=dataset_path, image_size=256)
        
        if num_samples:
            dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Evaluating on {len(dataset)} images...")
        
        # Initialize metrics
        mse_scores = []
        psnr_scores = []
        ssim_scores = []
        perceptual_losses = []
        
        real_features = []
        fake_features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                damaged = batch["damaged"].to(self.device)
                gt = batch["gt"].to(self.device)
                mask = batch["mask"].to(self.device)
                style_ref = batch["style_ref"].to(self.device)
                
                # Generate restored images
                style_vector = self.style_encoder(style_ref)
                restored = self.generator(damaged, mask, style_vector)
                
                # Calculate metrics for each image in batch
                for i in range(gt.size(0)):
                    gt_img = gt[i]
                    restored_img = restored[i]
                    
                    # MSE
                    mse = F.mse_loss(restored_img, gt_img).item()
                    mse_scores.append(mse)
                    
                    # Convert to numpy for SSIM/PSNR
                    gt_np = tensor_to_numpy(gt_img)
                    restored_np = tensor_to_numpy(restored_img)
                    
                    # PSNR
                    psnr_val = psnr(gt_np, restored_np, data_range=1.0)
                    psnr_scores.append(psnr_val)
                    
                    # SSIM (calculate for each channel and average)
                    if gt_np.shape[2] == 3:  # RGB
                        ssim_val = ssim(gt_np, restored_np, multichannel=True, channel_axis=2, data_range=1.0)
                    else:
                        ssim_val = ssim(gt_np, restored_np, data_range=1.0)
                    ssim_scores.append(ssim_val)
                
                # Perceptual Loss
                perc_loss, _ = self.vgg_loss(restored, gt)
                perceptual_losses.extend([perc_loss.item()] * gt.size(0))
                
                # Extract features for FID/IS
                # Denormalize images to [0, 1] for Inception
                gt_norm = (gt + 1.0) / 2.0
                restored_norm = (restored + 1.0) / 2.0
                
                real_feat = self.inception_extractor(gt_norm)
                fake_feat = self.inception_extractor(restored_norm)
                
                real_features.append(real_feat.cpu().numpy())
                fake_features.append(fake_feat.cpu().numpy())
        
        # Combine all features
        real_features = np.concatenate(real_features, axis=0)
        fake_features = np.concatenate(fake_features, axis=0)
        
        # Calculate FID
        fid_score = calculate_fid(real_features, fake_features)
        
        # Calculate IS
        fake_features_tensor = torch.from_numpy(fake_features).to(self.device)
        is_mean, is_std = calculate_inception_score(fake_features_tensor, device=self.device)
        
        # Calculate final metrics
        results = {
            'MSE': {
                'mean': np.mean(mse_scores),
                'std': np.std(mse_scores),
                'values': mse_scores
            },
            'PSNR': {
                'mean': np.mean(psnr_scores),
                'std': np.std(psnr_scores),
                'values': psnr_scores
            },
            'SSIM': {
                'mean': np.mean(ssim_scores),
                'std': np.std(ssim_scores),
                'values': ssim_scores
            },
            'Perceptual_Loss': {
                'mean': np.mean(perceptual_losses),
                'std': np.std(perceptual_losses),
                'values': perceptual_losses
            },
            'FID': fid_score,
            'IS_mean': is_mean.item(),
            'IS_std': is_std.item()
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*60)
        print("🎯 HAVELI-GAN EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 IMAGE QUALITY METRICS:")
        print(f"├── MSE:              {results['MSE']['mean']:.6f} ± {results['MSE']['std']:.6f}")
        print(f"├── PSNR:             {results['PSNR']['mean']:.2f} ± {results['PSNR']['std']:.2f} dB")
        print(f"├── SSIM:             {results['SSIM']['mean']:.4f} ± {results['SSIM']['std']:.4f}")
        print(f"└── Perceptual Loss:  {results['Perceptual_Loss']['mean']:.4f} ± {results['Perceptual_Loss']['std']:.4f}")
        
        print(f"\n🎨 GENERATIVE MODEL METRICS:")
        print(f"├── FID:              {results['FID']:.2f}")
        print(f"└── IS:               {results['IS_mean']:.2f} ± {results['IS_std']:.2f}")
        
        print(f"\n📈 INTERPRETATION:")
        print(f"├── Lower MSE = Better (closer to 0)")
        print(f"├── Higher PSNR = Better (>30 dB is good)")
        print(f"├── Higher SSIM = Better (closer to 1)")
        print(f"├── Lower Perceptual Loss = Better")
        print(f"├── Lower FID = Better (closer to 0)")
        print(f"└── Higher IS = Better (>2 is reasonable)")
        
        # Quality assessment
        print(f"\n🏆 QUALITY ASSESSMENT:")
        psnr_quality = "Excellent" if results['PSNR']['mean'] > 35 else "Good" if results['PSNR']['mean'] > 30 else "Fair"
        ssim_quality = "Excellent" if results['SSIM']['mean'] > 0.9 else "Good" if results['SSIM']['mean'] > 0.8 else "Fair"
        fid_quality = "Excellent" if results['FID'] < 10 else "Good" if results['FID'] < 50 else "Fair"
        
        print(f"├── PSNR Quality:     {psnr_quality}")
        print(f"├── SSIM Quality:     {ssim_quality}")
        print(f"└── FID Quality:      {fid_quality}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Haveli-GAN Model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_epoch_100.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--save_results', type=str, default='evaluation_results.txt',
                        help='File to save results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.checkpoint, args.device)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        args.data_dir, 
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results to file
    if args.save_results:
        with open(args.save_results, 'w') as f:
            f.write("HAVELI-GAN EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            for metric, values in results.items():
                if isinstance(values, dict):
                    f.write(f"{metric}:\n")
                    f.write(f"  Mean: {values['mean']:.6f}\n")
                    f.write(f"  Std:  {values['std']:.6f}\n\n")
                else:
                    f.write(f"{metric}: {values:.6f}\n\n")
        
        print(f"\n💾 Results saved to {args.save_results}")

if __name__ == "__main__":
    main()
