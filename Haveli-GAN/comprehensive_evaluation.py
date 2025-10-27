"""
Comprehensive Model Evaluation and Comparison Script
Evaluates Haveli-GAN against Partial Convolutions, EdgeConnect, and MAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import argparse
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import seaborn as sns
from torchvision import models, transforms
from scipy import linalg

# Import models and comparison framework
from dataset import FrescoDataset
from torch.utils.data import DataLoader

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG16"""
    
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.ModuleList()
        
        # Use layers: relu1_2, relu2_2, relu3_3, relu4_3
        layer_indices = [3, 8, 15, 22]
        current_layer = 0
        
        for i, layer in enumerate(vgg):
            self.vgg_layers.append(layer)
            if i in layer_indices:
                break
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        self.vgg_layers = self.vgg_layers.to(device)
        
        # Normalization for ImageNet pre-trained model
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, pred, target):
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_features = []
        target_features = []
        
        x_pred = pred_norm
        x_target = target_norm
        
        for layer in self.vgg_layers:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if isinstance(layer, nn.ReLU):
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # Calculate L1 loss between features
        perceptual_loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            perceptual_loss += F.l1_loss(pred_feat, target_feat)
        
        return perceptual_loss / len(pred_features)

class FIDCalculator:
    """Frechet Inception Distance calculator"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load pre-trained Inception v3
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.inception.eval()
        self.inception.to(device)
        
        # Freeze parameters
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_features(self, images):
        """Extract features from images using Inception v3"""
        if images.size(1) == 1:  # Grayscale to RGB
            images = images.repeat(1, 3, 1, 1)
        
        # Preprocess images
        preprocessed = []
        for img in images:
            preprocessed.append(self.preprocess(img))
        images = torch.stack(preprocessed)
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features.cpu().numpy()
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Frechet distance between two multivariate Gaussians"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + 
                np.trace(sigma2) - 2 * tr_covmean)
    
    def calculate_fid(self, real_images, fake_images):
        """Calculate FID score between real and fake images"""
        # Get features
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # Calculate statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        fid_score = self.calculate_frechet_distance(
            mu_real, sigma_real, mu_fake, sigma_fake
        )
        
        return fid_score

class ComprehensiveEvaluator:
    """Comprehensive evaluation of all models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = config.get('results_dir', 'evaluation_results')
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'sample_outputs'), exist_ok=True)
        
        # Initialize comparison framework
        self.comparison_framework = ModelComparisonFramework(
            device=self.device,
            batch_size=config.get('batch_size', 4),
            image_size=config.get('image_size', 256)
        )
        
        # Initialize metric calculators
        self.perceptual_loss = PerceptualLoss(device=self.device)
        self.fid_calculator = FIDCalculator(device=self.device)
        
        # Evaluation results storage
        self.detailed_results = {}
        self.summary_results = {}
        
        # Model performance tracking
        self.inference_times = {}
        self.memory_usage = {}
        
        # Storage for FID calculation
        self.real_images_for_fid = []
        self.fake_images_for_fid = {}
    
    def load_all_models(self):
        """Load all models with their checkpoints"""
        print("🔄 Loading all models...")
        
        # Load Haveli-GAN
        haveli_checkpoint = self.config.get('haveli_checkpoint', 'checkpoints/checkpoint_epoch_200.pth')
        self.comparison_framework.load_haveli_gan(haveli_checkpoint)
        
        # Load comparison models
        self.comparison_framework.load_comparison_models()
        
        print(f"✅ Loaded {len(self.comparison_framework.models)} models")
    
    def setup_test_data(self):
        """Setup test dataset"""
        print("📁 Setting up test dataset...")
        
        self.test_loader = get_data_loader(
            damaged_dir=self.config.get('test_damaged_dir', 'data/train_damaged'),
            mask_dir=self.config.get('test_mask_dir', 'data/train_masks'),
            ground_truth_dir=self.config.get('test_ground_truth_dir', 'data/train_ground_truth'),
            batch_size=self.comparison_framework.batch_size,
            shuffle=False
        )
        
        print(f"✅ Test dataset loaded: {len(self.test_loader)} batches")
    
    def measure_inference_time(self, model_name, model, damaged, mask, num_runs=5):
        """Measure inference time for a model"""
        times = []
        
        # Warm up
        if model_name == 'HaveliGAN':
            with torch.no_grad():
                style_vector = model['style_encoder'](damaged)  # Use damaged as style reference
                _ = model['generator'](damaged, mask, style_vector)
        else:
            with torch.no_grad():
                _ = model.inference(damaged, mask)
        
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        for _ in range(num_runs):
            start_time = time.time()
            
            if model_name == 'HaveliGAN':
                with torch.no_grad():
                    style_vector = model['style_encoder'](damaged)
                    _ = model['generator'](damaged, mask, style_vector)
            else:
                with torch.no_grad():
                    _ = model.inference(damaged, mask)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)
        
        return np.mean(times), np.std(times)
    
    def measure_memory_usage(self, model_name, model):
        """Measure GPU memory usage"""
        if not torch.cuda.is_available():
            return 0
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        dummy_mask = torch.ones(1, 1, 256, 256).to(self.device)
        
        if model_name == 'HaveliGAN':
            with torch.no_grad():
                style_vector = model['style_encoder'](dummy_input)
                _ = model['generator'](dummy_input, dummy_mask, style_vector)
        else:
            with torch.no_grad():
                _ = model.inference(dummy_input, dummy_mask)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        torch.cuda.empty_cache()
        
        return peak_memory
    
    def calculate_advanced_metrics(self, pred_img, gt_img, mask):
        """Calculate all requested evaluation metrics: L1, MSE, PSNR, SSIM, Perceptual Loss"""
        # Convert to numpy for traditional metrics
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
        
        if torch.is_tensor(mask):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        # Ensure correct range [0,1]
        pred_np = np.clip(pred_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        pred_tensor = torch.clamp(pred_tensor, 0, 1)
        gt_tensor = torch.clamp(gt_tensor, 0, 1)
        
        # Add batch dimension if needed for perceptual loss
        if len(pred_tensor.shape) == 3:
            pred_tensor = pred_tensor.unsqueeze(0)
            gt_tensor = gt_tensor.unsqueeze(0)
        
        # Move to device for perceptual loss calculation
        pred_tensor = pred_tensor.to(self.device)
        gt_tensor = gt_tensor.to(self.device)
        
        # L1 Loss (Mean Absolute Error)
        l1_loss = np.mean(np.abs(pred_np - gt_np))
        
        # MSE Loss
        mse_loss = np.mean((pred_np - gt_np) ** 2)
        
        # Convert to proper format for PSNR/SSIM (H, W, C)
        if len(pred_np.shape) == 3 and pred_np.shape[0] == 3:  # (C, H, W)
            pred_np_hwc = pred_np.transpose(1, 2, 0)
            gt_np_hwc = gt_np.transpose(1, 2, 0)
        else:
            pred_np_hwc = pred_np
            gt_np_hwc = gt_np
        
        # PSNR and SSIM
        psnr_val = psnr(gt_np_hwc, pred_np_hwc, data_range=1.0)
        ssim_val = ssim(gt_np_hwc, pred_np_hwc, data_range=1.0, 
                       channel_axis=2 if len(gt_np_hwc.shape) == 3 else None)
        
        # Perceptual Loss
        try:
            with torch.no_grad():
                perceptual_loss_val = self.perceptual_loss(pred_tensor, gt_tensor).item()
        except Exception as e:
            print(f"Warning: Perceptual loss calculation failed: {e}")
            perceptual_loss_val = 0.0
        
        # Hole-specific metrics (only in missing regions)
        if len(mask_np.shape) == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]  # Remove channel dimension
        
        hole_mask = (mask_np < 0.5)  # 0 indicates holes
        
        if np.any(hole_mask):
            hole_l1 = np.mean(np.abs(pred_np_hwc[hole_mask] - gt_np_hwc[hole_mask]))
            hole_mse = np.mean((pred_np_hwc[hole_mask] - gt_np_hwc[hole_mask]) ** 2)
        else:
            hole_l1 = hole_mse = 0
        
        # Valid region metrics
        valid_mask = (mask_np >= 0.5)
        if np.any(valid_mask):
            valid_l1 = np.mean(np.abs(pred_np_hwc[valid_mask] - gt_np_hwc[valid_mask]))
            valid_mse = np.mean((pred_np_hwc[valid_mask] - gt_np_hwc[valid_mask]) ** 2)
        else:
            valid_l1 = valid_mse = 0
        
        return {
            'l1_loss': l1_loss,
            'mse_loss': mse_loss,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'perceptual_loss': perceptual_loss_val,
            'hole_l1': hole_l1,
            'hole_mse': hole_mse,
            'valid_l1': valid_l1,
            'valid_mse': valid_mse
        }
    
    def evaluate_single_sample(self, sample_idx, damaged, mask, ground_truth):
        """Evaluate all models on a single sample"""
        sample_results = {}
        sample_predictions = {}
        
        for model_name, model in self.comparison_framework.models.items():
            try:
                # Measure inference time
                inf_time_mean, inf_time_std = self.measure_inference_time(
                    model_name, model, damaged, mask
                )
                
                # Generate prediction
                if model_name == 'HaveliGAN':
                    model['generator'].eval()
                    model['style_encoder'].eval()
                    with torch.no_grad():
                        style_vector = model['style_encoder'](ground_truth)  # Use GT for style
                        pred = model['generator'](damaged, mask, style_vector)
                        prediction = pred[0]
                else:
                    prediction = model.inference(damaged, mask)[0]
                
                sample_predictions[model_name] = prediction
                
                # Store images for FID calculation
                if model_name not in self.fake_images_for_fid:
                    self.fake_images_for_fid[model_name] = []
                
                # Convert prediction to proper format for FID
                if torch.is_tensor(prediction):
                    pred_for_fid = prediction.unsqueeze(0)  # Add batch dimension
                else:
                    pred_for_fid = torch.from_numpy(prediction).unsqueeze(0)
                
                self.fake_images_for_fid[model_name].append(pred_for_fid.cpu())
                
                # Calculate metrics
                metrics = self.calculate_advanced_metrics(
                    prediction, ground_truth[0], mask[0]
                )
                
                # Add timing information
                metrics['inference_time_mean'] = inf_time_mean
                metrics['inference_time_std'] = inf_time_std
                
                sample_results[model_name] = metrics
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name} on sample {sample_idx}: {e}")
                sample_results[model_name] = None
        
        return sample_results, sample_predictions
    
    def save_sample_comparison(self, sample_idx, damaged, mask, ground_truth, predictions):
        """Save visual comparison of all model outputs"""
        num_models = len(predictions)
        fig, axes = plt.subplots(2, max(3, num_models), figsize=(4*max(3, num_models), 8))
        
        # Convert tensors to numpy for visualization
        if torch.is_tensor(damaged):
            damaged_np = damaged[0].detach().cpu().permute(1, 2, 0).numpy()
        if torch.is_tensor(mask):
            mask_np = mask[0].detach().cpu().squeeze().numpy()
        if torch.is_tensor(ground_truth):
            gt_np = ground_truth[0].detach().cpu().permute(1, 2, 0).numpy()
        
        # First row: Input, Mask, Ground Truth
        axes[0, 0].imshow(np.clip(damaged_np, 0, 1))
        axes[0, 0].set_title('Damaged Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.clip(gt_np, 0, 1))
        axes[0, 2].set_title('Ground Truth')
        axes[0, 2].axis('off')
        
        # Hide remaining axes in first row
        for i in range(3, axes.shape[1]):
            axes[0, i].axis('off')
        
        # Second row: Model predictions
        for i, (model_name, pred) in enumerate(predictions.items()):
            if i < axes.shape[1]:
                if torch.is_tensor(pred):
                    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
                else:
                    pred_np = pred
                
                axes[1, i].imshow(np.clip(pred_np, 0, 1))
                axes[1, i].set_title(model_name)
                axes[1, i].axis('off')
        
        # Hide remaining axes in second row
        for i in range(len(predictions), axes.shape[1]):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, 'sample_outputs', f'comparison_sample_{sample_idx:04d}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation on all models"""
        print("🔍 Starting comprehensive evaluation...")
        
        all_results = {model_name: [] for model_name in self.comparison_framework.models.keys()}
        
        # Measure memory usage for each model
        print("📊 Measuring memory usage...")
        for model_name, model in self.comparison_framework.models.items():
            memory_usage = self.measure_memory_usage(model_name, model)
            self.memory_usage[model_name] = memory_usage
            print(f"  {model_name}: {memory_usage:.2f} GB")
        
        num_samples = self.config.get('num_eval_samples', 100)
        save_every = max(1, num_samples // 20)  # Save 20 comparison images
        
        with torch.no_grad():
            for batch_idx, (damaged, mask, ground_truth, _) in enumerate(self.test_loader):
                if batch_idx >= num_samples:
                    break
                
                damaged = damaged.to(self.device)
                mask = mask.to(self.device)
                ground_truth = ground_truth.to(self.device)
                
                # Evaluate single sample
                sample_results, sample_predictions = self.evaluate_single_sample(
                    batch_idx, damaged, mask, ground_truth
                )
                
                # Store results
                for model_name, results in sample_results.items():
                    if results is not None:
                        all_results[model_name].append(results)
                
                # Save visual comparison
                if batch_idx % save_every == 0:
                    self.save_sample_comparison(
                        batch_idx, damaged, mask, ground_truth, sample_predictions
                    )
                
                # Store real images for FID calculation
                if len(self.real_images_for_fid) < 1000:  # Limit to 1000 images for FID
                    real_for_fid = ground_truth[0].unsqueeze(0).cpu()  # Add batch dimension
                    self.real_images_for_fid.append(real_for_fid)
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{num_samples} samples...")
        
        # Calculate FID scores
        print("📊 Calculating FID scores...")
        self.calculate_fid_scores()
        
        # Calculate summary statistics
        self.calculate_summary_statistics(all_results)
        
        # Generate visualizations
        self.generate_comparison_plots()
        
        # Save detailed results
        self.save_detailed_results(all_results)
        
        print("✅ Comprehensive evaluation completed!")
    
    def calculate_fid_scores(self):
        """Calculate FID scores for all models"""
        if not self.real_images_for_fid:
            print("⚠️ No real images available for FID calculation")
            return
        
        # Concatenate real images
        real_images = torch.cat(self.real_images_for_fid, dim=0)
        print(f"  Real images for FID: {real_images.shape[0]}")
        
        self.fid_scores = {}
        
        for model_name, fake_images_list in self.fake_images_for_fid.items():
            if not fake_images_list:
                continue
            
            try:
                # Concatenate fake images (limit to same number as real images)
                num_real = real_images.shape[0]
                fake_images_list = fake_images_list[:num_real]
                fake_images = torch.cat(fake_images_list, dim=0)
                
                print(f"  Calculating FID for {model_name}: {fake_images.shape[0]} images")
                
                # Calculate FID in batches to avoid memory issues
                batch_size = 50
                fid_score = self.calculate_fid_in_batches(
                    real_images, fake_images, batch_size
                )
                
                self.fid_scores[model_name] = fid_score
                print(f"  {model_name} FID: {fid_score:.2f}")
                
            except Exception as e:
                print(f"❌ Error calculating FID for {model_name}: {e}")
                self.fid_scores[model_name] = float('inf')
    
    def calculate_fid_in_batches(self, real_images, fake_images, batch_size=50):
        """Calculate FID in batches to handle memory constraints"""
        real_features = []
        fake_features = []
        
        # Process real images in batches
        for i in range(0, real_images.shape[0], batch_size):
            batch = real_images[i:i+batch_size].to(self.device)
            features = self.fid_calculator.get_features(batch)
            real_features.append(features)
        
        # Process fake images in batches
        for i in range(0, fake_images.shape[0], batch_size):
            batch = fake_images[i:i+batch_size].to(self.device)
            features = self.fid_calculator.get_features(batch)
            fake_features.append(features)
        
        # Concatenate features
        real_features = np.concatenate(real_features, axis=0)
        fake_features = np.concatenate(fake_features, axis=0)
        
        # Calculate statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        fid_score = self.fid_calculator.calculate_frechet_distance(
            mu_real, sigma_real, mu_fake, sigma_fake
        )
        
        return fid_score

    def calculate_summary_statistics(self, all_results):
        """Calculate summary statistics for all models"""
        self.summary_results = {}
        
        for model_name, results in all_results.items():
            if not results:
                continue
            
            # Calculate means and standard deviations
            metrics_summary = {}
            for metric in results[0].keys():
                values = [r[metric] for r in results if metric in r]
                if values:
                    metrics_summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
            
            # Add model metadata
            metrics_summary['num_samples'] = len(results)
            metrics_summary['memory_usage_gb'] = self.memory_usage.get(model_name, 0)
            metrics_summary['fid_score'] = self.fid_scores.get(model_name, float('inf'))
            
            self.summary_results[model_name] = metrics_summary
        
        # Print summary
        print("\\n📈 Evaluation Summary:")
        print("-" * 100)
        print(f"{'Model':<12} | {'L1':<8} | {'MSE':<10} | {'PSNR':<8} | {'SSIM':<8} | {'Perceptual':<10} | {'FID':<8} | {'Time(s)':<8} | {'Memory(GB)':<10}")
        print("-" * 100)
        
        for model_name, summary in self.summary_results.items():
            l1_mean = summary.get('l1_loss', {}).get('mean', 0)
            mse_mean = summary.get('mse_loss', {}).get('mean', 0)
            psnr_mean = summary.get('psnr', {}).get('mean', 0)
            ssim_mean = summary.get('ssim', {}).get('mean', 0)
            perceptual_mean = summary.get('perceptual_loss', {}).get('mean', 0)
            fid_score = summary.get('fid_score', float('inf'))
            time_mean = summary.get('inference_time_mean', {}).get('mean', 0)
            memory = summary.get('memory_usage_gb', 0)
            
            print(f"{model_name:<12} | {l1_mean:<8.4f} | {mse_mean:<10.6f} | {psnr_mean:<8.2f} | {ssim_mean:<8.4f} | {perceptual_mean:<10.4f} | {fid_score:<8.1f} | {time_mean:<8.4f} | {memory:<10.2f}")
    
    def generate_comparison_plots(self):
        """Generate comprehensive comparison plots"""
        print("📊 Generating comparison plots...")
        
        # Metrics comparison bar plot
        self.plot_metrics_comparison()
        
        # Performance vs quality trade-off
        self.plot_performance_trade_off()
        
        # Detailed metrics heatmap
        self.plot_metrics_heatmap()
        
        # Box plots for metric distributions
        self.plot_metric_distributions()
    
    def plot_metrics_comparison(self):
        """Plot side-by-side metrics comparison"""
        models = list(self.summary_results.keys())
        metrics = ['l1_loss', 'mse_loss', 'psnr', 'ssim', 'perceptual_loss', 'fid_score']
        metric_labels = ['L1 Loss', 'MSE Loss', 'PSNR (dB)', 'SSIM', 'Perceptual Loss', 'FID Score']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = []
            errors = []
            model_names = []
            
            for model in models:
                if metric == 'fid_score':
                    # FID is stored directly, not as mean/std
                    fid_val = self.summary_results[model].get('fid_score', float('inf'))
                    if fid_val != float('inf'):
                        values.append(fid_val)
                        errors.append(0)  # No std for FID
                        model_names.append(model)
                elif metric in self.summary_results[model]:
                    values.append(self.summary_results[model][metric]['mean'])
                    errors.append(self.summary_results[model][metric]['std'])
                    model_names.append(model)
            
            if values:
                bars = axes[i].bar(model_names, values, yerr=errors, capsize=5, alpha=0.8)
                axes[i].set_title(f'{label} Comparison')
                axes[i].set_ylabel(label)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Color bars
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for bar, color in zip(bars, colors[:len(bars)]):
                    bar.set_color(color)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.3f}' if val < 100 else f'{val:.1f}', 
                               ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'visualizations', 'comprehensive_metrics_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_trade_off(self):
        """Plot performance vs quality trade-off"""
        models = list(self.summary_results.keys())
        
        psnr_values = []
        time_values = []
        memory_values = []
        
        for model in models:
            if 'psnr' in self.summary_results[model]:
                psnr_values.append(self.summary_results[model]['psnr']['mean'])
                time_values.append(self.summary_results[model]['inference_time_mean']['mean'])
                memory_values.append(self.summary_results[model]['memory_usage_gb'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PSNR vs Inference Time
        scatter1 = ax1.scatter(time_values, psnr_values, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (time_values[i], psnr_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Inference Time (seconds)')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Quality vs Speed Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # PSNR vs Memory Usage
        scatter2 = ax2.scatter(memory_values, psnr_values, s=100, alpha=0.7, color='orange')
        for i, model in enumerate(models):
            ax2.annotate(model, (memory_values[i], psnr_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Memory Usage (GB)')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Quality vs Memory Trade-off')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'visualizations', 'performance_trade_off.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_heatmap(self):
        """Plot comprehensive metrics heatmap"""
        models = list(self.summary_results.keys())
        metrics = ['l1_loss', 'mse_loss', 'psnr', 'ssim', 'perceptual_loss', 'fid_score', 'inference_time_mean']
        metric_labels = ['L1 Loss', 'MSE Loss', 'PSNR', 'SSIM', 'Perceptual', 'FID', 'Time(s)']
        
        # Create data matrix
        data_matrix = []
        for model in models:
            row = []
            for metric in metrics:
                if metric == 'fid_score':
                    value = self.summary_results[model].get('fid_score', float('inf'))
                    if value == float('inf'):
                        value = np.nan
                elif metric in self.summary_results[model]:
                    value = self.summary_results[model][metric]['mean']
                else:
                    value = np.nan
                row.append(value)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Normalize each metric to [0, 1] for better visualization
        # For metrics where lower is better (L1, MSE, Perceptual, FID, Time), invert the normalization
        normalized_data = np.zeros_like(data_matrix)
        lower_is_better = [True, True, False, False, True, True, True]  # L1, MSE, PSNR, SSIM, Perceptual, FID, Time
        
        for j in range(data_matrix.shape[1]):
            col = data_matrix[:, j]
            if not np.all(np.isnan(col)):
                col_min, col_max = np.nanmin(col), np.nanmax(col)
                if col_max > col_min:
                    if lower_is_better[j]:
                        # Invert for metrics where lower is better
                        normalized_data[:, j] = 1 - (col - col_min) / (col_max - col_min)
                    else:
                        normalized_data[:, j] = (col - col_min) / (col_max - col_min)
        
        plt.figure(figsize=(12, 8))
        
        # Use try-catch for seaborn import
        try:
            import seaborn as sns
            sns.heatmap(normalized_data, 
                       xticklabels=metric_labels,
                       yticklabels=models,
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlGn',  # Red-Yellow-Green colormap
                       cbar_kws={'label': 'Normalized Score (Higher = Better)'})
        except ImportError:
            # Fallback to matplotlib imshow
            im = plt.imshow(normalized_data, cmap='RdYlGn', aspect='auto')
            plt.colorbar(im, label='Normalized Score (Higher = Better)')
            plt.xticks(range(len(metric_labels)), metric_labels, rotation=45)
            plt.yticks(range(len(models)), models)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(metrics)):
                    plt.text(j, i, f'{normalized_data[i, j]:.3f}', 
                           ha='center', va='center', color='black')
        
        plt.title('Comprehensive Metrics Heatmap (Normalized - Higher = Better)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'visualizations', 'comprehensive_metrics_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metric_distributions(self):
        """Plot box plots for metric distributions"""
        # This would require storing all individual results, which we do in detailed_results
        # For now, we'll create a placeholder
        print("📊 Metric distribution plots would require individual sample data...")
    
    def save_detailed_results(self, all_results):
        """Save detailed results to JSON"""
        # Save summary results with FID scores
        summary_file = os.path.join(self.results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'evaluation_config': self.config,
                'evaluation_date': datetime.now().isoformat(),
                'summary_results': self.summary_results,
                'memory_usage': self.memory_usage,
                'fid_scores': getattr(self, 'fid_scores', {})
            }, f, indent=2)
        
        # Save detailed results (first 50 samples to avoid huge files)
        detailed_sample = {}
        for model_name, results in all_results.items():
            detailed_sample[model_name] = results[:50]  # First 50 samples
        
        detailed_file = os.path.join(self.results_dir, 'detailed_results_sample.json')
        with open(detailed_file, 'w') as f:
            json.dump(detailed_sample, f, indent=2)
        
        print(f"💾 Results saved to {self.results_dir}")
        print(f"  - Summary: {summary_file}")
        print(f"  - Detailed sample: {detailed_file}")

def get_default_evaluation_config():
    """Get default evaluation configuration"""
    return {
        'batch_size': 4,
        'image_size': 256,
        'num_eval_samples': 100,
        'results_dir': 'comprehensive_evaluation_results',
        'test_damaged_dir': 'data/train_damaged',
        'test_mask_dir': 'data/train_masks',
        'test_ground_truth_dir': 'data/train_ground_truth',
        'haveli_checkpoint': 'checkpoints/checkpoint_epoch_200.pth'
    }

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--results_dir', type=str, default='comprehensive_evaluation_results', 
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_evaluation_config()
    
    # Override with command line arguments
    if args.num_samples:
        config['num_eval_samples'] = args.num_samples
    if args.results_dir:
        config['results_dir'] = args.results_dir
    
    print("🚀 Starting Comprehensive Model Evaluation")
    print(f"📊 Evaluating on {config['num_eval_samples']} samples")
    print(f"💾 Results will be saved to: {config['results_dir']}")
    print("-" * 60)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(config)
    
    # Load models
    evaluator.load_all_models()
    
    # Setup test data
    evaluator.setup_test_data()
    
    # Run evaluation
    evaluator.run_comprehensive_evaluation()
    
    print(f"\\n🎉 Evaluation completed! Check results in: {config['results_dir']}")

if __name__ == "__main__":
    main()