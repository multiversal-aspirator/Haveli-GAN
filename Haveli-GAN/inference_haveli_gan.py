#!/usr/bin/env python3
"""
Haveli-GAN Inference Script
===========================

This script performs fresco restoration inference using the trained Haveli-GAN model.
It can restore damaged Indian paintings while preserving their authentic artistic styles.

Usage:
    python inference_haveli_gan.py --input path/to/damaged/image.jpg --mask path/to/mask.jpg --output path/to/restored/image.jpg
    python inference_haveli_gan.py --input_dir path/to/damaged/images/ --output_dir path/to/restored/images/
    python inference_haveli_gan.py --demo  # Run demo with sample images

Features:
- Single image restoration
- Batch processing of multiple images
- Automatic mask generation for damaged regions
- Multiple checkpoint loading options
- High-quality output with style preservation
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import argparse
import os
import glob
from pathlib import Path
import time

from model import Generator, StyleEncoder
from dataset import FrescoDataset

class HaveliGANInference:
    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize the Haveli-GAN inference engine.
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.style_encoder = StyleEncoder().to(self.device)
        
        # Load checkpoint
        if checkpoint_path is None:
            # Try to find the latest checkpoint
            checkpoint_path = self.find_latest_checkpoint()
        
        self.load_checkpoint(checkpoint_path)
        
        # Set models to evaluation mode
        self.generator.eval()
        self.style_encoder.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Changed from 512 to 256 for consistency
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.to_pil = transforms.ToPILImage()
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint in the checkpoints directory."""
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(script_dir, "checkpoints")
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoints directory found at: {checkpoint_dir}")
        
        # Look for checkpoint files with the correct naming pattern
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
        if not checkpoints:
            # Try alternative patterns
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "gen_epoch_*.pth"))
            if checkpoints:
                # Legacy format - we need both generator and style encoder
                latest_gen = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                epoch_num = latest_gen.split('_')[-1].split('.')[0]
                style_checkpoint = os.path.join(checkpoint_dir, f"style_enc_epoch_{epoch_num}.pth")
                if os.path.exists(style_checkpoint):
                    print(f"Found legacy checkpoints: {latest_gen} and {style_checkpoint}")
                    return latest_gen  # We'll handle this specially in load_checkpoint
                else:
                    raise FileNotFoundError("Style encoder checkpoint not found for legacy format!")
            else:
                raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")
        
        # Sort by epoch number
        def extract_epoch(checkpoint_path):
            # Extract epoch number, handling milestone files
            filename = os.path.basename(checkpoint_path)
            # Split by '_' and find the part with epoch number
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if part == 'epoch' and i + 1 < len(parts):
                    epoch_part = parts[i + 1]
                    # Handle milestone files: "200_milestone.pth" -> "200"
                    epoch_num = epoch_part.split('.')[0]
                    if epoch_num != 'milestone':
                        return int(epoch_num)
            return 0  # fallback
        
        checkpoints.sort(key=extract_epoch)
        latest_checkpoint = checkpoints[-1]
        print(f"Using latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    def load_checkpoint(self, checkpoint_path):
        """Load the trained model weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Check if this is a legacy checkpoint (separate files)
        if "gen_epoch_" in checkpoint_path:
            # Legacy format - load generator and style encoder separately
            gen_checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.generator.load_state_dict(gen_checkpoint)
            
            # Find corresponding style encoder checkpoint
            epoch_num = checkpoint_path.split('_')[-1].split('.')[0]
            style_checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), f"style_enc_epoch_{epoch_num}.pth")
            style_checkpoint = torch.load(style_checkpoint_path, map_location=self.device, weights_only=False)
            self.style_encoder.load_state_dict(style_checkpoint)
            
            print(f"Loaded legacy format models from epoch: {epoch_num}")
        else:
            # New unified format
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'gen_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['gen_state_dict'])
                self.style_encoder.load_state_dict(checkpoint['style_enc_state_dict'])
            else:
                # Alternative format
                self.generator.load_state_dict(checkpoint['generator'])
                self.style_encoder.load_state_dict(checkpoint['style_encoder'])
            
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded model from epoch: {epoch}")
    
    def load_image(self, image_path):
        """Load and preprocess an image."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def load_mask(self, mask_path):
        """Load and preprocess a mask."""
        mask = Image.open(mask_path).convert('L')
        mask = self.mask_transform(mask).unsqueeze(0).to(self.device)
        return mask
    
    def generate_automatic_mask(self, damaged_image_tensor):
        """
        Generate an automatic mask for damaged regions.
        This is a simple implementation - you might want to use more sophisticated methods.
        """
        # Convert to numpy for processing
        img_np = damaged_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Create mask for very dark or very bright regions (potential damage)
        mask1 = (gray < 30).astype(np.uint8) * 255  # Very dark regions
        mask2 = (gray > 225).astype(np.uint8) * 255  # Very bright regions
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert back to tensor
        mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
        
        return mask
    
    def restore_image(self, damaged_image_path, mask_path=None, reference_image_path=None):
        """
        Restore a single damaged image.
        
        Args:
            damaged_image_path (str): Path to the damaged image
            mask_path (str): Path to the damage mask (optional - will auto-generate if None)
            reference_image_path (str): Path to reference image for style (optional - will use damaged image if None)
        
        Returns:
            PIL.Image: Restored image
        """
        print(f"Restoring image: {damaged_image_path}")
        
        # Load damaged image
        damaged_img = self.load_image(damaged_image_path)
        
        # Load or generate mask
        if mask_path and os.path.exists(mask_path):
            mask = self.load_mask(mask_path)
            print(f"Using provided mask: {mask_path}")
        else:
            mask = self.generate_automatic_mask(damaged_img)
            print("Generated automatic mask for damaged regions")
        
        # Load reference image for style (use damaged image if no reference provided)
        if reference_image_path and os.path.exists(reference_image_path):
            reference_img = self.load_image(reference_image_path)
            print(f"Using reference image for style: {reference_image_path}")
        else:
            reference_img = damaged_img
            print("Using damaged image as style reference")
        
        # Perform inference
        with torch.no_grad():
            # Extract style from reference image
            style_vector = self.style_encoder(reference_img)
            
            # Generate restored image
            restored_img = self.generator(damaged_img, mask, style_vector)
            
            # Post-process
            restored_img = torch.clamp(restored_img, -1, 1)
            restored_img = (restored_img + 1) / 2  # Normalize to [0, 1]
            
            # Convert to PIL image
            restored_pil = self.to_pil(restored_img.squeeze().cpu())
            
        return restored_pil
    
    def restore_batch(self, input_dir, output_dir, mask_dir=None):
        """
        Restore all images in a directory.
        
        Args:
            input_dir (str): Directory containing damaged images
            output_dir (str): Directory to save restored images
            mask_dir (str): Directory containing masks (optional)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to restore")
        
        for i, image_path in enumerate(image_files):
            start_time = time.time()
            
            # Get corresponding mask path if mask directory is provided
            mask_path = None
            if mask_dir:
                image_name = os.path.basename(image_path)
                mask_name = os.path.splitext(image_name)[0] + '_mask.png'
                mask_path = os.path.join(mask_dir, mask_name)
            
            # Restore image
            try:
                restored_img = self.restore_image(image_path, mask_path)
                
                # Save restored image
                output_path = os.path.join(output_dir, f"restored_{os.path.basename(image_path)}")
                restored_img.save(output_path, quality=95)
                
                elapsed_time = time.time() - start_time
                print(f"[{i+1}/{len(image_files)}] Restored {os.path.basename(image_path)} -> {os.path.basename(output_path)} ({elapsed_time:.2f}s)")
                
            except Exception as e:
                print(f"Error restoring {image_path}: {str(e)}")
    
    def create_comparison_image(self, damaged_path, restored_path, output_path):
        """Create a side-by-side comparison image."""
        damaged_img = Image.open(damaged_path)
        restored_img = Image.open(restored_path)
        
        # Resize images to same size
        size = (512, 512)
        damaged_img = damaged_img.resize(size)
        restored_img = restored_img.resize(size)
        
        # Create comparison
        comparison = Image.new('RGB', (1024, 512))
        comparison.paste(damaged_img, (0, 0))
        comparison.paste(restored_img, (512, 0))
        
        comparison.save(output_path, quality=95)
        print(f"Comparison saved: {output_path}")
    
    def run_demo(self):
        """Run a demo using sample images from the training data."""
        print("Running Haveli-GAN Demo...")
        
        # Create demo output directory
        demo_dir = "./demo_restoration"
        os.makedirs(demo_dir, exist_ok=True)
        
        # Find some sample images from training data
        sample_damaged = glob.glob("./data/train_damaged/*.jpg")[:5]
        sample_ground_truth = glob.glob("./data/train_ground_truth/*.jpg")[:5]
        
        if not sample_damaged:
            print("No sample images found in ./data/train_damaged/")
            return
        
        print(f"Demo: Restoring {len(sample_damaged)} sample images...")
        
        for i, damaged_path in enumerate(sample_damaged):
            try:
                # Restore the image
                restored_img = self.restore_image(damaged_path)
                
                # Save restored image
                image_name = os.path.basename(damaged_path)
                restored_path = os.path.join(demo_dir, f"demo_restored_{image_name}")
                restored_img.save(restored_path, quality=95)
                
                # Create comparison if ground truth exists
                gt_name = os.path.basename(damaged_path)
                gt_path = os.path.join("./data/train_ground_truth", gt_name)
                if os.path.exists(gt_path):
                    comparison_path = os.path.join(demo_dir, f"demo_comparison_{image_name}")
                    self.create_comparison_with_gt(damaged_path, restored_path, gt_path, comparison_path)
                
                print(f"Demo [{i+1}/{len(sample_damaged)}]: {image_name} -> restored")
                
            except Exception as e:
                print(f"Demo error with {damaged_path}: {str(e)}")
        
        print(f"Demo completed! Results saved in {demo_dir}")
    
    def create_comparison_with_gt(self, damaged_path, restored_path, gt_path, output_path):
        """Create a three-way comparison: damaged | restored | ground truth."""
        damaged_img = Image.open(damaged_path).resize((512, 512))
        restored_img = Image.open(restored_path).resize((512, 512))
        gt_img = Image.open(gt_path).resize((512, 512))
        
        # Create three-way comparison
        comparison = Image.new('RGB', (1536, 512))
        comparison.paste(damaged_img, (0, 0))
        comparison.paste(restored_img, (512, 0))
        comparison.paste(gt_img, (1024, 0))
        
        comparison.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description='Haveli-GAN Fresco Restoration Inference')
    parser.add_argument('--input', type=str, help='Path to input damaged image')
    parser.add_argument('--mask', type=str, help='Path to damage mask (optional)')
    parser.add_argument('--reference', type=str, help='Path to reference image for style (optional)')
    parser.add_argument('--output', type=str, help='Path to save restored image')
    parser.add_argument('--input_dir', type=str, help='Directory containing damaged images')
    parser.add_argument('--output_dir', type=str, help='Directory to save restored images')
    parser.add_argument('--mask_dir', type=str, help='Directory containing masks (optional)')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample images')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference_engine = HaveliGANInference(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing inference engine: {str(e)}")
        return
    
    # Run demo
    if args.demo:
        inference_engine.run_demo()
        return
    
    # Single image restoration
    if args.input and args.output:
        try:
            restored_img = inference_engine.restore_image(
                args.input, 
                args.mask, 
                args.reference
            )
            restored_img.save(args.output, quality=95)
            print(f"Restored image saved: {args.output}")
            
            # Create comparison image
            base_name = os.path.splitext(args.output)[0]
            ext = os.path.splitext(args.output)[1]
            comparison_path = f"{base_name}_comparison{ext}"
            inference_engine.create_comparison_image(args.input, args.output, comparison_path)
            
        except Exception as e:
            print(f"Error during single image restoration: {str(e)}")
        return
    
    # Batch restoration
    if args.input_dir and args.output_dir:
        try:
            inference_engine.restore_batch(
                args.input_dir,
                args.output_dir,
                args.mask_dir
            )
        except Exception as e:
            print(f"Error during batch restoration: {str(e)}")
        return
    
    # If no valid arguments provided, show help
    parser.print_help()


if __name__ == "__main__":
    main()
