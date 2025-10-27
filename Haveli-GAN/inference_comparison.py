#!/usr/bin/env python3
"""
Inference Script for Model Comparison
Compares outputs from HaveliGAN (epoch 200), PartialConv, EdgeConnect, and MAT
"""

import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import FrescoDataset

# Import models
from model import Generator, StyleEncoder
from partial_conv_model import PartialConvModel
from edgeconnect_model import EdgeConnect
from mat_model import MAT

class InferenceComparison:
    """Compare outputs from all 4 inpainting models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        self.models = {}
        self.load_all_models()
        
    def load_all_models(self):
        """Load all trained models"""
        print("🔄 Loading all models...")
        
        # 1. Load HaveliGAN from epoch 200 checkpoint
        print("📦 Loading HaveliGAN (epoch 200)...")
        try:
            checkpoint = torch.load('checkpoints/checkpoint_epoch_200.pth', map_location=self.device)
            
            generator = Generator().to(self.device)
            style_encoder = StyleEncoder().to(self.device)
            
            # Load state dicts
            generator.load_state_dict(checkpoint['gen_state_dict'])
            style_encoder.load_state_dict(checkpoint['style_enc_state_dict'])
            
            generator.eval()
            style_encoder.eval()
            
            self.models['HaveliGAN'] = {
                'generator': generator,
                'style_encoder': style_encoder
            }
            print("✅ HaveliGAN loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load HaveliGAN: {e}")
            self.models['HaveliGAN'] = None
        
        # 2. Load PartialConv (5 epochs trained)
        print("📦 Loading PartialConv...")
        try:
            model = PartialConvModel(device=self.device, lr=0.0002)
            model.load_checkpoint('sequential_checkpoints_200epochs/PartialConv_final.pth')
            model.model.eval()
            self.models['PartialConv'] = model
            print("✅ PartialConv loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load PartialConv: {e}")
            self.models['PartialConv'] = None
        
        # 3. Load EdgeConnect
        print("📦 Loading EdgeConnect...")
        try:
            model = EdgeConnect(device=self.device, lr=0.0002)
            model.load_checkpoint('sequential_checkpoints_200epochs/EdgeConnect_final.pth')
            model.edge_generator.eval()
            model.inpaint_generator.eval()
            self.models['EdgeConnect'] = model
            print("✅ EdgeConnect loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load EdgeConnect: {e}")
            self.models['EdgeConnect'] = None
        
        # 4. Load MAT
        print("📦 Loading MAT...")
        try:
            model = MAT(device=self.device, lr=0.0002)
            model.load_checkpoint('sequential_checkpoints_200epochs/MAT_final.pth')
            model.generator.eval()
            self.models['MAT'] = model
            print("✅ MAT loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load MAT: {e}")
            self.models['MAT'] = None
        
        print(f"📊 Successfully loaded {sum(1 for m in self.models.values() if m is not None)}/4 models")
    
    def run_inference(self, damaged, mask, gt):
        """Run inference on all models"""
        results = {}
        
        print("\n🔄 Running inference on all models...")
        
        with torch.no_grad():
            # HaveliGAN
            if self.models['HaveliGAN']:
                try:
                    generator = self.models['HaveliGAN']['generator']
                    style_encoder = self.models['HaveliGAN']['style_encoder']
                    
                    style_features = style_encoder(gt)
                    output = generator(damaged, mask, style_features)
                    results['HaveliGAN'] = output.cpu()
                    print("✅ HaveliGAN inference complete")
                except Exception as e:
                    print(f"❌ HaveliGAN inference failed: {e}")
                    results['HaveliGAN'] = None
            
            # PartialConv
            if self.models['PartialConv']:
                try:
                    output = self.models['PartialConv'].inference(damaged, mask)
                    results['PartialConv'] = output.cpu()
                    print("✅ PartialConv inference complete")
                except Exception as e:
                    print(f"❌ PartialConv inference failed: {e}")
                    results['PartialConv'] = None
            
            # EdgeConnect
            if self.models['EdgeConnect']:
                try:
                    output = self.models['EdgeConnect'].inference(damaged, mask)
                    results['EdgeConnect'] = output.cpu()
                    print("✅ EdgeConnect inference complete")
                except Exception as e:
                    print(f"❌ EdgeConnect inference failed: {e}")
                    results['EdgeConnect'] = None
            
            # MAT
            if self.models['MAT']:
                try:
                    output = self.models['MAT'].inference(damaged, mask)
                    results['MAT'] = output.cpu()
                    print("✅ MAT inference complete")
                except Exception as e:
                    print(f"❌ MAT inference failed: {e}")
                    results['MAT'] = None
        
        return results
    
    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension
        
        numpy_image = tensor.permute(1, 2, 0).numpy()
        
        # Convert to PIL Image
        image = Image.fromarray((numpy_image * 255).astype(np.uint8))
        return image
    
    def create_comparison_grid(self, damaged, mask, gt, results, save_path='inference_comparison.png'):
        """Create a comparison grid showing all model outputs"""
        
        # Convert inputs to images
        damaged_img = self.tensor_to_image(damaged.cpu().squeeze(0))
        mask_img = self.tensor_to_image(mask.cpu().squeeze(0).repeat(3, 1, 1))  # Convert grayscale to RGB
        gt_img = self.tensor_to_image(gt.cpu().squeeze(0))
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Model Comparison: Fresco Inpainting Results', fontsize=16, fontweight='bold')
        
        # First row: inputs and ground truth
        axes[0, 0].imshow(damaged_img)
        axes[0, 0].set_title('Damaged Input', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_img, cmap='gray')
        axes[0, 1].set_title('Mask', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gt_img)
        axes[0, 2].set_title('Ground Truth', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].axis('off')  # Empty space
        
        # Second row: model outputs
        model_names = ['HaveliGAN', 'PartialConv', 'EdgeConnect', 'MAT']
        
        for i, model_name in enumerate(model_names):
            if results.get(model_name) is not None:
                result_img = self.tensor_to_image(results[model_name])
                axes[1, i].imshow(result_img)
                axes[1, i].set_title(f'{model_name}', fontweight='bold')
            else:
                axes[1, i].text(0.5, 0.5, f'{model_name}\n(Failed)', 
                               ha='center', va='center', transform=axes[1, i].transAxes,
                               fontsize=12, color='red')
                axes[1, i].set_facecolor('lightgray')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📸 Comparison saved to: {save_path}")
        
        return save_path
    
    def compare_models(self, sample_idx=0):
        """Compare all models on a single sample"""
        print(f"\n🎯 Running model comparison on sample {sample_idx}")
        
        # Load dataset
        dataset = FrescoDataset(root_dir='data')
        
        if sample_idx >= len(dataset):
            sample_idx = 0
            print(f"⚠️ Sample index too large, using sample 0")
        
        # Get sample
        sample = dataset[sample_idx]
        damaged = sample['damaged'].unsqueeze(0).to(self.device)
        mask = sample['mask'].unsqueeze(0).to(self.device)
        gt = sample['gt'].unsqueeze(0).to(self.device)
        
        print(f"📊 Sample info:")
        print(f"  Damaged: {damaged.shape}")
        print(f"  Mask: {mask.shape}")
        print(f"  Ground truth: {gt.shape}")
        
        # Run inference
        results = self.run_inference(damaged, mask, gt)
        
        # Create comparison
        comparison_path = self.create_comparison_grid(damaged, mask, gt, results, 
                                                    f'inference_comparisons_sample_{sample_idx}.png')
        
        # Print summary
        print(f"\n📋 Results Summary:")
        working_models = [name for name, result in results.items() if result is not None]
        failed_models = [name for name, result in results.items() if result is None]
        
        print(f"✅ Working models ({len(working_models)}): {', '.join(working_models)}")
        if failed_models:
            print(f"❌ Failed models ({len(failed_models)}): {', '.join(failed_models)}")
        
        return results, comparison_path

def main():
    print("🚀 Starting Model Inference Comparison")
    print("="*60)
    
    # Initialize comparison
    comparator = InferenceComparison()
    
    # Run comparison on first sample
    results, comparison_path = comparator.compare_models(sample_idx=0)
    
    print(f"\n🎉 Model comparison complete!")
    print(f"📸 Comparison image saved: {comparison_path}")
    
    return results

if __name__ == "__main__":
    main()