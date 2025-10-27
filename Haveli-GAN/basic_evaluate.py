#!/usr/bin/env python3
"""
Simplified evaluation script for Haveli-GAN focusing on basic metrics
Calculates: MSE, PSNR, SSIM, Perceptual Loss
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from tqdm import tqdm

from model import Generator, StyleEncoder
from loss import VGGPerceptualLoss
from dataset import FrescoDataset

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

class BasicEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load models
        self.generator = Generator().to(device)
        self.style_encoder = StyleEncoder().to(device)
        self.vgg_loss = VGGPerceptualLoss().to(device)
        
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
        """Basic evaluation on dataset"""
        
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
        l1_losses = []
        
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
                    
                    # L1 Loss
                    l1 = F.l1_loss(restored_img, gt_img).item()
                    l1_losses.append(l1)
                    
                    # Convert to numpy for SSIM/PSNR
                    gt_np = tensor_to_numpy(gt_img)
                    restored_np = tensor_to_numpy(restored_img)
                    
                    # PSNR
                    try:
                        psnr_val = psnr(gt_np, restored_np, data_range=1.0)
                        psnr_scores.append(psnr_val)
                    except Exception as e:
                        print(f"PSNR calculation error: {e}")
                        psnr_scores.append(0)
                    
                    # SSIM
                    try:
                        if gt_np.shape[2] == 3:  # RGB
                            ssim_val = ssim(gt_np, restored_np, multichannel=True, channel_axis=2, data_range=1.0)
                        else:
                            ssim_val = ssim(gt_np, restored_np, data_range=1.0)
                        ssim_scores.append(ssim_val)
                    except Exception as e:
                        print(f"SSIM calculation error: {e}")
                        ssim_scores.append(0)
                
                # Perceptual Loss (batch-wise)
                try:
                    perc_loss, _ = self.vgg_loss(restored, gt)
                    perceptual_losses.extend([perc_loss.item()] * gt.size(0))
                except Exception as e:
                    print(f"Perceptual loss calculation error: {e}")
                    perceptual_losses.extend([0.0] * gt.size(0))
        
        # Calculate final metrics
        results = {
            'MSE': {
                'mean': np.mean(mse_scores),
                'std': np.std(mse_scores),
                'values': mse_scores
            },
            'L1': {
                'mean': np.mean(l1_losses),
                'std': np.std(l1_losses),
                'values': l1_losses
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
            }
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*60)
        print("🎯 HAVELI-GAN BASIC EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 IMAGE QUALITY METRICS:")
        print(f"├── MSE:              {results['MSE']['mean']:.6f} ± {results['MSE']['std']:.6f}")
        print(f"├── L1:               {results['L1']['mean']:.6f} ± {results['L1']['std']:.6f}")
        print(f"├── PSNR:             {results['PSNR']['mean']:.2f} ± {results['PSNR']['std']:.2f} dB")
        print(f"├── SSIM:             {results['SSIM']['mean']:.4f} ± {results['SSIM']['std']:.4f}")
        print(f"└── Perceptual Loss:  {results['Perceptual_Loss']['mean']:.4f} ± {results['Perceptual_Loss']['std']:.4f}")
        
        print(f"\n📈 INTERPRETATION:")
        print(f"├── Lower MSE/L1 = Better (closer to 0)")
        print(f"├── Higher PSNR = Better (>30 dB is good, >35 dB is excellent)")
        print(f"├── Higher SSIM = Better (closer to 1, >0.8 is good, >0.9 is excellent)")
        print(f"└── Lower Perceptual Loss = Better")
        
        # Quality assessment
        print(f"\n🏆 QUALITY ASSESSMENT:")
        psnr_quality = "Excellent" if results['PSNR']['mean'] > 35 else "Good" if results['PSNR']['mean'] > 30 else "Fair" if results['PSNR']['mean'] > 25 else "Poor"
        ssim_quality = "Excellent" if results['SSIM']['mean'] > 0.9 else "Good" if results['SSIM']['mean'] > 0.8 else "Fair" if results['SSIM']['mean'] > 0.7 else "Poor"
        
        print(f"├── PSNR Quality:     {psnr_quality}")
        print(f"├── SSIM Quality:     {ssim_quality}")
        
        # Training comparison
        print(f"\n📋 TRAINING COMPARISON:")
        print(f"├── Training L1 Loss: ~0.044 (final epoch)")
        print(f"├── Evaluation L1:    {results['L1']['mean']:.4f}")
        print(f"└── Consistency:      {'✅ Good' if abs(results['L1']['mean'] - 0.044) < 0.02 else '⚠️ Check'}")
        
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Basic Evaluation for Haveli-GAN Model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_epoch_100.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BasicEvaluator(args.checkpoint, args.device)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        args.data_dir, 
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results to file
    with open("basic_evaluation_results.txt", "w") as f:
        f.write("HAVELI-GAN BASIC EVALUATION RESULTS\n")
        f.write("="*45 + "\n\n")
        for metric, values in results.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {values['mean']:.6f}\n")
            f.write(f"  Std:  {values['std']:.6f}\n\n")
    
    print(f"\n💾 Results saved to 'basic_evaluation_results.txt'")
    
    return results

if __name__ == "__main__":
    main()
