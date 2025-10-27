#!/usr/bin/env python3
"""
Interactive Haveli-GAN Inference
================================

Interactive script to run inference with user-selected images.
Allows browsing and selecting specific damaged images for restoration.
"""

import torch
from torchvision import transforms
from PIL import Image
import os
import glob
from model import Generator, StyleEncoder

class InteractiveInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {self.device}")
        
        # Load MAT model
        self.load_model()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Data directories
        self.damaged_dir = "./data/train_damaged"
        self.mask_dir = "./data/train_masks"
        self.gt_dir = "./data/train_ground_truth"
        self.output_dir = "./interactive_results"
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_models(self):
        """Load the trained models"""
        # Find latest checkpoint
        checkpoint_path = "./checkpoints/checkpoint_epoch_200.pth"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "./checkpoints/checkpoint_epoch_100.pth"
        
        print(f"📂 Loading models from: {checkpoint_path}")
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.style_encoder = StyleEncoder().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint['gen_state_dict'])
        self.style_encoder.load_state_dict(checkpoint['style_enc_state_dict'])
        
        # Set to evaluation mode
        self.generator.eval()
        self.style_encoder.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Models loaded successfully from epoch {epoch}")
    
    def get_available_images(self):
        """Get list of available images with complete sets (damaged, mask, ground truth)"""
        if not os.path.exists(self.damaged_dir):
            print(f"❌ Damaged images directory not found: {self.damaged_dir}")
            return []
        
        damaged_files = glob.glob(os.path.join(self.damaged_dir, "*.jpg"))
        available_images = []
        
        for damaged_path in damaged_files:
            filename = os.path.basename(damaged_path)
            mask_path = os.path.join(self.mask_dir, filename)
            gt_path = os.path.join(self.gt_dir, filename)
            
            if os.path.exists(mask_path) and os.path.exists(gt_path):
                available_images.append({
                    'name': filename,
                    'damaged': damaged_path,
                    'mask': mask_path,
                    'ground_truth': gt_path
                })
        
        return available_images
    
    def display_image_list(self, images):
        """Display numbered list of available images"""
        print("\n" + "="*60)
        print("🎨 AVAILABLE IMAGES FOR RESTORATION")
        print("="*60)
        
        # Group by art style for better organization
        styles = {}
        for i, img in enumerate(images):
            style = img['name'].split('.')[0]
            # Extract art style from filename
            for art_style in ['gond', 'kalighat', 'kangra', 'kerala', 'madhubani', 'mandana', 'pichwai', 'warli']:
                if art_style in style.lower():
                    if art_style not in styles:
                        styles[art_style] = []
                    styles[art_style].append((i, img))
                    break
            else:
                if 'other' not in styles:
                    styles['other'] = []
                styles['other'].append((i, img))
        
        for style, style_images in styles.items():
            print(f"\n🎭 {style.upper()} Style:")
            for idx, img in style_images:
                print(f"  {idx + 1:2d}. {img['name']}")
    
    def get_user_choice(self, images):
        """Get user's image selection"""
        while True:
            try:
                choice = input(f"\n🎯 Select image number (1-{len(images)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(images):
                    return images[choice_num - 1]
                else:
                    print(f"❌ Please enter a number between 1 and {len(images)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
    
    def restore_image(self, image_info):
        """Restore the selected image"""
        print(f"\n🎨 Restoring: {image_info['name']}")
        
        try:
            # Load images
            damaged_img = Image.open(image_info['damaged']).convert("RGB")
            mask_img = Image.open(image_info['mask']).convert("L")
            gt_img = Image.open(image_info['ground_truth']).convert("RGB")
            
            # Transform images
            damaged_tensor = self.transform(damaged_img).unsqueeze(0).to(self.device)
            mask_tensor = self.mask_transform(mask_img).unsqueeze(0).to(self.device)
            style_tensor = self.transform(gt_img).unsqueeze(0).to(self.device)
            
            # Perform restoration
            with torch.no_grad():
                style_vector = self.style_encoder(style_tensor)
                restored_tensor = self.generator(damaged_tensor, mask_tensor, style_vector)
            
            # Convert back to PIL image
            restored_tensor = torch.clamp(restored_tensor, -1, 1)
            restored_tensor = (restored_tensor + 1) / 2  # Denormalize to [0, 1]
            restored_img = transforms.ToPILImage()(restored_tensor.squeeze(0).cpu())
            
            # Save results
            base_name = os.path.splitext(image_info['name'])[0]
            
            # Save individual images
            restored_path = os.path.join(self.output_dir, f"restored_{base_name}.png")
            restored_img.save(restored_path, quality=95)
            
            # Create comparison image
            comparison_img = self.create_comparison(damaged_img, restored_img, gt_img)
            comparison_path = os.path.join(self.output_dir, f"comparison_{base_name}.png")
            comparison_img.save(comparison_path, quality=95)
            
            print(f"✅ Restoration complete!")
            print(f"   📁 Restored image: {restored_path}")
            print(f"   📊 Comparison: {comparison_path}")
            
            return restored_img, comparison_img
            
        except Exception as e:
            print(f"❌ Error during restoration: {e}")
            return None, None
    
    def create_comparison(self, damaged, restored, ground_truth):
        """Create a side-by-side comparison image"""
        # Resize all images to same size for comparison
        size = (256, 256)
        damaged = damaged.resize(size)
        restored = restored.resize(size)
        ground_truth = ground_truth.resize(size)
        
        # Create comparison image
        comparison_width = size[0] * 3 + 40  # 3 images + margins
        comparison_height = size[1] + 60  # Image height + text space
        
        comparison = Image.new('RGB', (comparison_width, comparison_height), 'white')
        
        # Paste images
        comparison.paste(damaged, (10, 40))
        comparison.paste(restored, (size[0] + 20, 40))
        comparison.paste(ground_truth, (size[0] * 2 + 30, 40))
        
        # Add labels (would need PIL.ImageDraw for text)
        return comparison
    
    def run_interactive_session(self):
        """Run the interactive inference session"""
        print("🎨 Welcome to Haveli-GAN Interactive Inference!")
        print("This tool helps you restore damaged Indian heritage paintings.")
        
        # Get available images
        images = self.get_available_images()
        
        if not images:
            print("❌ No images found! Please check your data directory structure.")
            return
        
        print(f"📊 Found {len(images)} complete image sets for restoration")
        
        while True:
            # Display available images
            self.display_image_list(images)
            
            # Get user choice
            selected_image = self.get_user_choice(images)
            
            if selected_image is None:
                print("👋 Thanks for using Haveli-GAN! Goodbye!")
                break
            
            # Restore the selected image
            restored_img, comparison_img = self.restore_image(selected_image)
            
            if restored_img is not None:
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Restore another image? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("👋 Thanks for using Haveli-GAN! Goodbye!")
                    break
            else:
                print("❌ Restoration failed. Please try another image.")
    
    def run_batch_inference(self):
        """Run inference on all available images and save results with comparison"""
        images = self.get_available_images()
        if not images:
            print("❌ No images found! Please check your data directory structure.")
            return
        print(f"📊 Found {len(images)} complete image sets for batch restoration")
        for idx, image_info in enumerate(images):
            print(f"\n[{idx+1}/{len(images)}] Restoring: {image_info['name']}")
            restored_img, comparison_img = self.restore_image(image_info)
            if restored_img is None:
                print(f"❌ Failed to restore {image_info['name']}")
        print("\n🎉 Batch inference complete! All results saved in:", self.output_dir)

def main():
    """Main function"""
    try:
        runner = InteractiveInference()
        runner.run_interactive_session()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()