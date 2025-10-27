#!/usr/bin/env python3
"""
Comprehensive inference comparison with all available models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import FrescoDataset
import os

# Import all models
from model import Generator, StyleEncoder
from partial_conv_model import PartialConvModel
from edgeconnect_model import EdgeConnect
from mat_model import MAT

def comprehensive_model_comparison():
    """Compare all available models"""
    print("🚀 Comprehensive Model Comparison")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = FrescoDataset(root_dir='data')
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    models = {}
    
    # 1. Load HaveliGAN (epoch 200)
    print("\n📦 Loading HaveliGAN (epoch 200)...")
    try:
        generator = Generator().to(device)
        style_encoder = StyleEncoder().to(device)
        
        checkpoint = torch.load('checkpoints/checkpoint_epoch_200.pth', map_location=device)
        generator.load_state_dict(checkpoint['gen_state_dict'])
        style_encoder.load_state_dict(checkpoint['style_enc_state_dict'])
        
        generator.eval()
        style_encoder.eval()
        
        models['HaveliGAN'] = {'generator': generator, 'style_encoder': style_encoder}
        print("✅ HaveliGAN loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load HaveliGAN: {e}")
        models['HaveliGAN'] = None
    
    # 2. Load trained PartialConv (5 epochs)
    print("\n📦 Loading trained PartialConv (5 epochs)...")
    try:
        model = PartialConvModel(device=device, lr=0.0002)
        model.load_checkpoint('partialconv_5epoch_checkpoints/PartialConv_5epochs_final.pth')
        model.model.eval()
        models['PartialConv'] = model
        print("✅ PartialConv loaded (5 epochs trained)")
    except Exception as e:
        print(f"❌ Failed to load PartialConv: {e}")
        models['PartialConv'] = None
    
    # 3. Load EdgeConnect (200 epochs)
    print("\n📦 Loading EdgeConnect (200 epochs)...")
    try:
        model = EdgeConnect(device=device, lr=0.0002)
        model.load_checkpoint('sequential_checkpoints_200epochs/EdgeConnect_final.pth')
        model.edge_generator.eval()
        model.inpaint_generator.eval()
        models['EdgeConnect'] = model
        print("✅ EdgeConnect loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load EdgeConnect: {e}")
        models['EdgeConnect'] = None
    
    # 4. Load MAT (200 epochs)
    print("\n📦 Loading MAT (200 epochs)...")
    try:
        model = MAT(device=device, lr=0.0002)
        model.load_checkpoint('sequential_checkpoints_200epochs/MAT_final.pth')
        model.generator.eval()
        models['MAT'] = model
        print("✅ MAT loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load MAT: {e}")
        models['MAT'] = None
    
    successful_models = [name for name, model in models.items() if model is not None]
    print(f"\n📊 Successfully loaded {len(successful_models)}/4 models: {successful_models}")
    
    # Run inference on first sample
    print("\n🔄 Running inference on test sample...")
    sample = next(iter(test_loader))
    damaged = sample['damaged'].to(device)
    mask = sample['mask'].to(device)
    gt = sample['gt'].to(device)
    
    print(f"Input shapes:")
    print(f"  Damaged: {damaged.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Ground truth: {gt.shape}")
    
    results = {}
    
    with torch.no_grad():
        # HaveliGAN
        if models['HaveliGAN']:
            try:
                generator = models['HaveliGAN']['generator']
                style_encoder = models['HaveliGAN']['style_encoder']
                
                style_features = style_encoder(gt)
                output = generator(damaged, mask, style_features)
                results['HaveliGAN'] = output.cpu()
                print("✅ HaveliGAN inference complete")
            except Exception as e:
                print(f"❌ HaveliGAN inference failed: {e}")
                results['HaveliGAN'] = None
        
        # PartialConv
        if models['PartialConv']:
            try:
                output = models['PartialConv'].inference(damaged, mask)
                results['PartialConv'] = output.cpu()
                print("✅ PartialConv inference complete (5 epochs trained)")
            except Exception as e:
                print(f"❌ PartialConv inference failed: {e}")
                results['PartialConv'] = None
        
        # EdgeConnect
        if models['EdgeConnect']:
            try:
                output = models['EdgeConnect'].inference(damaged, mask)
                results['EdgeConnect'] = output.cpu()
                print("✅ EdgeConnect inference complete")
            except Exception as e:
                print(f"❌ EdgeConnect inference failed: {e}")
                results['EdgeConnect'] = None
        
        # MAT
        if models['MAT']:
            try:
                output = models['MAT'].inference(damaged, mask)
                results['MAT'] = output.cpu()
                print("✅ MAT inference complete")
            except Exception as e:
                print(f"❌ MAT inference failed: {e}")
                results['MAT'] = None
    
    # Create comparison visualization
    working_results = {k: v for k, v in results.items() if v is not None}
    num_models = len(working_results)
    
    if num_models > 0:
        fig, axes = plt.subplots(2, max(3, num_models), figsize=(4*max(3, num_models), 8))
        if num_models == 1:
            axes = axes.reshape(2, -1)
        
        # Top row: Input, mask, ground truth
        damaged_np = damaged[0].cpu().permute(1, 2, 0).numpy()
        mask_np = mask[0, 0].cpu().numpy()
        gt_np = gt[0].cpu().permute(1, 2, 0).numpy()
        
        damaged_np = np.clip(damaged_np, 0, 1)
        gt_np = np.clip(gt_np, 0, 1)
        
        axes[0, 0].imshow(damaged_np)
        axes[0, 0].set_title('Damaged Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gt_np)
        axes[0, 2].set_title('Ground Truth')
        axes[0, 2].axis('off')
        
        # Hide extra top row axes
        for i in range(3, max(3, num_models)):
            if i < axes.shape[1]:
                axes[0, i].axis('off')
        
        # Bottom row: Model outputs
        for idx, (model_name, output) in enumerate(working_results.items()):
            if idx < axes.shape[1]:
                output_np = output[0].permute(1, 2, 0).numpy()
                output_np = np.clip(output_np, 0, 1)
                
                axes[1, idx].imshow(output_np)
                title = f'{model_name}'
                if model_name == 'PartialConv':
                    title += '\n(5 epochs)'
                elif model_name in ['EdgeConnect', 'MAT']:
                    title += '\n(200 epochs)'
                elif model_name == 'HaveliGAN':
                    title += '\n(200 epochs)'
                axes[1, idx].set_title(title)
                axes[1, idx].axis('off')
        
        # Hide extra bottom row axes
        for i in range(num_models, max(3, num_models)):
            if i < axes.shape[1]:
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n📸 Comprehensive comparison saved to: comprehensive_model_comparison.png")
        
        # Calculate metrics for trained models
        print(f"\n📊 Performance Metrics:")
        for model_name, output in working_results.items():
            if output is not None:
                gt_tensor = gt.cpu()
                mse = torch.nn.functional.mse_loss(output, gt_tensor).item()
                
                # Calculate PSNR
                psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse)))
                
                status = ""
                if model_name == 'PartialConv':
                    status = " (5 epochs)"
                elif model_name in ['EdgeConnect', 'MAT']:
                    status = " (200 epochs)"
                elif model_name == 'HaveliGAN':
                    status = " (200 epochs)"
                
                print(f"  {model_name:12}: MSE={mse:.6f}, PSNR={psnr:.2f} dB{status}")
    
    print(f"\n🎉 Comprehensive comparison complete!")
    print(f"📸 Results saved to: comprehensive_model_comparison.png")
    
    return results

if __name__ == "__main__":
    results = comprehensive_model_comparison()