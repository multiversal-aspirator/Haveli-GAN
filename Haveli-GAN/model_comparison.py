"""
Model Comparison Framework for Haveli-GAN
Compares Haveli-GAN against Partial Convolutions, EdgeConnect, and MAT models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
from torchvision import models, transforms
from scipy import linalg

# Import existing models
from model import Generator, StyleEncoder
from dataset import get_data_loader

# Import comparison models
from partial_conv_model import PartialConvModel
from edgeconnect_model import EdgeConnect
from mat_model import MAT

class ModelComparisonFramework:
    """Framework for comparing different inpainting models"""
    
    def __init__(self, device='cuda', batch_size=8, image_size=256):
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self.results_dir = "comparison_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.model_names = ['HaveliGAN', 'PartialConv', 'EdgeConnect', 'MAT']
        
        # Evaluation metrics storage
        self.metrics = {name: {'l1_loss': [], 'mse_loss': [], 'psnr': [], 'ssim': [], 
                              'perceptual_loss': [], 'training_time': 0} 
                       for name in self.model_names}
    
    def load_comparison_models(self):
        """Load all comparison models"""
        
        # Load Partial Convolutions
        print("🔄 Loading Partial Convolutions model...")
        partial_conv = PartialConvModel(device=self.device)
        checkpoint_path = "checkpoints/partial_conv_checkpoint.pth"
        if os.path.exists(checkpoint_path):
            partial_conv.load_checkpoint(checkpoint_path)
            print(f"✅ Loaded Partial Convolutions from {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found, using randomly initialized Partial Convolutions")
        self.models['PartialConv'] = partial_conv
        
        # Load EdgeConnect
        print("🔄 Loading EdgeConnect model...")
        edgeconnect = EdgeConnect(device=self.device)
        checkpoint_path = "checkpoints/edgeconnect_checkpoint.pth"
        if os.path.exists(checkpoint_path):
            edgeconnect.load_checkpoint(checkpoint_path)
            print(f"✅ Loaded EdgeConnect from {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found, using randomly initialized EdgeConnect")
        self.models['EdgeConnect'] = edgeconnect
        
        # Load MAT
        print("🔄 Loading MAT model...")
        mat = MAT(device=self.device)
        checkpoint_path = "checkpoints/mat_checkpoint.pth"
        if os.path.exists(checkpoint_path):
            mat.load_checkpoint(checkpoint_path)
            print(f"✅ Loaded MAT from {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found, using randomly initialized MAT")
        self.models['MAT'] = mat
    def load_haveli_gan(self, checkpoint_path=None):
        """Load the existing Haveli-GAN model"""
        generator = Generator().to(self.device)
        style_encoder = StyleEncoder().to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            generator.load_state_dict(checkpoint['generator'])
            style_encoder.load_state_dict(checkpoint['style_encoder'])
            print(f"✅ Loaded Haveli-GAN from {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found, using randomly initialized Haveli-GAN")
        
        self.models['HaveliGAN'] = {'generator': generator, 'style_encoder': style_encoder}
    
    def calculate_metrics(self, pred_img, gt_img):
        """Calculate evaluation metrics between predicted and ground truth images"""
        # Convert to numpy arrays and ensure proper range [0,1]
        if torch.is_tensor(pred_img):
            pred_np = pred_img.detach().cpu().numpy()
            pred_tensor = pred_img.detach()
        else:
            pred_np = np.array(pred_img)
            pred_tensor = torch.from_numpy(pred_np)
        
        if torch.is_tensor(gt_img):
            gt_np = gt_img.detach().cpu().numpy()
            gt_tensor = gt_img.detach()
        else:
            gt_np = np.array(gt_img)
            gt_tensor = torch.from_numpy(gt_np)
        
        # Ensure images are in [0,1] range
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        pred_tensor = torch.clamp(pred_tensor, 0, 1)
        gt_tensor = torch.clamp(gt_tensor, 0, 1)
        
        # Calculate L1 and MSE
        l1_loss = np.mean(np.abs(pred_np - gt_np))
        mse_loss = np.mean((pred_np - gt_np) ** 2)
        
        # For multichannel images, calculate PSNR and SSIM
        if len(pred_np.shape) == 3:  # (H, W, C)
            psnr_val = psnr(gt_np, pred_np, data_range=1.0)
            ssim_val = ssim(gt_np, pred_np, data_range=1.0, channel_axis=2)
        else:  # Grayscale
            psnr_val = psnr(gt_np, pred_np, data_range=1.0)
            ssim_val = ssim(gt_np, pred_np, data_range=1.0)
        
        # Calculate perceptual loss (simplified version)
        try:
            # Add batch dimension if needed
            if len(pred_tensor.shape) == 3:
                pred_tensor = pred_tensor.unsqueeze(0)
                gt_tensor = gt_tensor.unsqueeze(0)
            
            # Simple perceptual loss using MSE in feature space
            with torch.no_grad():
                # Resize for VGG if needed
                if pred_tensor.shape[-1] != 224:
                    pred_resized = F.interpolate(pred_tensor, size=(224, 224), mode='bilinear')
                    gt_resized = F.interpolate(gt_tensor, size=(224, 224), mode='bilinear')
                else:
                    pred_resized = pred_tensor
                    gt_resized = gt_tensor
                
                # Simple feature extraction (can be enhanced with actual VGG)
                pred_features = F.avg_pool2d(pred_resized, 8)
                gt_features = F.avg_pool2d(gt_resized, 8)
                perceptual_loss = F.mse_loss(pred_features, gt_features).item()
        except:
            perceptual_loss = 0.0
        
        return {
            'l1_loss': l1_loss,
            'mse_loss': mse_loss,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'perceptual_loss': perceptual_loss
        }
    
    def save_comparison_image(self, damaged, mask, ground_truth, predictions, filename):
        """Save a comparison image showing all model outputs"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Convert tensors to numpy for visualization
        if torch.is_tensor(damaged):
            damaged = damaged.detach().cpu().permute(1, 2, 0).numpy()
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().squeeze().numpy()
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.detach().cpu().permute(1, 2, 0).numpy()
        
        # Top row: Input, Mask, Ground Truth
        axes[0, 0].imshow(np.clip(damaged, 0, 1))
        axes[0, 0].set_title('Damaged Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.clip(ground_truth, 0, 1))
        axes[0, 2].set_title('Ground Truth')
        axes[0, 2].axis('off')
        
        # Bottom row: Model predictions (show first 3 models)
        model_idx = 0
        for i in range(3):
            if model_idx < len(predictions):
                pred = predictions[list(predictions.keys())[model_idx]]
                if torch.is_tensor(pred):
                    pred = pred.detach().cpu().permute(1, 2, 0).numpy()
                axes[1, i].imshow(np.clip(pred, 0, 1))
                axes[1, i].set_title(list(predictions.keys())[model_idx])
                axes[1, i].axis('off')
                model_idx += 1
            else:
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate_models(self, test_loader, num_samples=None):
        """Evaluate all models on test dataset"""
        print("🔍 Starting model evaluation...")
        
        all_predictions = {name: [] for name in self.model_names if name in self.models}
        
        with torch.no_grad():
            for idx, (damaged, mask, ground_truth, _) in enumerate(test_loader):
                if num_samples and idx >= num_samples:
                    break
                
                damaged = damaged.to(self.device)
                mask = mask.to(self.device)
                ground_truth = ground_truth.to(self.device)
                
                predictions = {}
                
                # Haveli-GAN prediction
                if 'HaveliGAN' in self.models:
                    haveli_gen = self.models['HaveliGAN']['generator']
                    haveli_style = self.models['HaveliGAN']['style_encoder']
                    
                    haveli_gen.eval()
                    haveli_style.eval()
                    
                    with torch.no_grad():
                        style_vector = haveli_style(ground_truth)  # Use GT for style reference
                        pred = haveli_gen(damaged, mask, style_vector)
                        predictions['HaveliGAN'] = pred[0]
                
                # Partial Convolutions prediction
                if 'PartialConv' in self.models:
                    partial_conv = self.models['PartialConv']
                    pred = partial_conv.inference(damaged, mask)
                    predictions['PartialConv'] = pred[0]
                
                # EdgeConnect prediction
                if 'EdgeConnect' in self.models:
                    edgeconnect = self.models['EdgeConnect']
                    pred = edgeconnect.inference(damaged, mask)
                    predictions['EdgeConnect'] = pred[0]
                
                # MAT prediction
                if 'MAT' in self.models:
                    mat = self.models['MAT']
                    pred = mat.inference(damaged, mask)
                    predictions['MAT'] = pred[0]
                
                # Calculate metrics for each model
                for model_name, pred in predictions.items():
                    if len(pred.shape) == 4:  # Remove batch dimension if present
                        pred = pred[0]
                    if len(ground_truth.shape) == 4:
                        gt = ground_truth[0]
                    else:
                        gt = ground_truth
                    
                    # Convert to numpy for metric calculation
                    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
                    gt_np = gt.detach().cpu().permute(1, 2, 0).numpy()
                    
                    metrics = self.calculate_metrics(pred_np, gt_np)
                    
                    self.metrics[model_name]['l1_loss'].append(metrics['l1_loss'])
                    self.metrics[model_name]['mse_loss'].append(metrics['mse_loss'])
                    self.metrics[model_name]['psnr'].append(metrics['psnr'])
                    self.metrics[model_name]['ssim'].append(metrics['ssim'])
                    self.metrics[model_name]['perceptual_loss'].append(metrics['perceptual_loss'])
                
                # Save comparison image every 10 samples
                if idx % 10 == 0:
                    self.save_comparison_image(
                        damaged[0], mask[0], ground_truth[0], 
                        predictions, f'comparison_{idx:04d}.png'
                    )
                
                if idx % 50 == 0:
                    print(f"  Processed {idx} samples...")
        
        # Calculate average metrics
        print("\n📊 Evaluation Results:")
        print("-" * 90)
        print(f"{'Model':<12} | {'L1':<8} | {'MSE':<10} | {'PSNR':<8} | {'SSIM':<8} | {'Perceptual':<10}")
        print("-" * 90)
        for model_name in self.model_names:
            if model_name in self.models and self.metrics[model_name]['psnr']:
                avg_l1 = np.mean(self.metrics[model_name]['l1_loss'])
                avg_mse = np.mean(self.metrics[model_name]['mse_loss'])
                avg_psnr = np.mean(self.metrics[model_name]['psnr'])
                avg_ssim = np.mean(self.metrics[model_name]['ssim'])
                avg_perceptual = np.mean(self.metrics[model_name]['perceptual_loss'])
                
                print(f"{model_name:12} | {avg_l1:.4f} | {avg_mse:.6f} | {avg_psnr:.2f} | {avg_ssim:.4f} | {avg_perceptual:.4f}")
        
        # Save detailed results
        self.save_results()
    
    def save_results(self):
        """Save detailed comparison results"""
        results_file = os.path.join(self.results_dir, 'comparison_results.json')
        
        # Calculate summary statistics
        summary = {}
        for model_name, metrics in self.metrics.items():
            if metrics['psnr']:  # Only include models that have results
                summary[model_name] = {
                    'avg_l1_loss': float(np.mean(metrics['l1_loss'])),
                    'std_l1_loss': float(np.std(metrics['l1_loss'])),
                    'avg_mse_loss': float(np.mean(metrics['mse_loss'])),
                    'std_mse_loss': float(np.std(metrics['mse_loss'])),
                    'avg_psnr': float(np.mean(metrics['psnr'])),
                    'std_psnr': float(np.std(metrics['psnr'])),
                    'avg_ssim': float(np.mean(metrics['ssim'])),
                    'std_ssim': float(np.std(metrics['ssim'])),
                    'avg_perceptual_loss': float(np.mean(metrics['perceptual_loss'])),
                    'std_perceptual_loss': float(np.std(metrics['perceptual_loss'])),
                    'training_time': metrics['training_time'],
                    'num_samples': len(metrics['psnr'])
                }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Detailed results saved to {results_file}")
    
    def plot_metrics_comparison(self):
        """Create visualization of metric comparisons"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        models_with_data = [name for name in self.model_names 
                           if name in self.models and self.metrics[name]['psnr']]
        
        if not models_with_data:
            print("⚠️ No models with evaluation data found")
            return
        
        metrics_to_plot = ['l1_loss', 'mse_loss', 'psnr', 'ssim', 'perceptual_loss']
        metric_labels = ['L1 Loss', 'MSE Loss', 'PSNR (dB)', 'SSIM', 'Perceptual Loss']
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            if i >= len(axes):
                break
                
            values = [np.mean(self.metrics[name][metric]) for name in models_with_data]
            stds = [np.std(self.metrics[name][metric]) for name in models_with_data]
            
            bars = axes[i].bar(models_with_data, values, yerr=stds, capsize=5)
            axes[i].set_title(f'{label} Comparison')
            axes[i].set_ylabel(label)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Color bars differently
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for bar, color in zip(bars, colors[:len(bars)]):
                bar.set_color(color)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}' if val < 1 else f'{val:.2f}', 
                           ha='center', va='bottom')
        
        # Hide unused subplot
        if len(axes) > len(metrics_to_plot):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'comprehensive_metrics_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Comprehensive metrics comparison plot saved to {self.results_dir}/comprehensive_metrics_comparison.png")

def main():
    """Main comparison function"""
    print("🚀 Starting Haveli-GAN Model Comparison")
    
    # Initialize comparison framework
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    comparison = ModelComparisonFramework(device=device, batch_size=4)
    
    # Load Haveli-GAN (existing model)
    checkpoint_path = "checkpoints/checkpoint_epoch_200.pth"
    comparison.load_haveli_gan(checkpoint_path)
    
    # Load all comparison models
    comparison.load_comparison_models()
    
    # Load test dataset
    print("📁 Loading test dataset...")
    try:
        test_loader = get_data_loader(
            damaged_dir="data/train_damaged",
            mask_dir="data/train_masks", 
            ground_truth_dir="data/train_ground_truth",
            batch_size=comparison.batch_size,
            shuffle=False
        )
        print(f"✅ Test dataset loaded with {len(test_loader)} batches")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Run evaluation (start with just Haveli-GAN for now)
    comparison.evaluate_models(test_loader, num_samples=100)
    
    # Create visualizations
    comparison.plot_metrics_comparison()
    
    print(f"\n🎉 Comparison complete! Results saved in '{comparison.results_dir}' directory")

if __name__ == "__main__":
    main()