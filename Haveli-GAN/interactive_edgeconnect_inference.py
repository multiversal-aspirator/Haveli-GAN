#!/usr/bin/env python3
"""
Interactive EdgeConnect Model Inference
=======================================

Interactive script to run inference using only the EdgeConnect model.
Allows browsing and selecting specific damaged images for restoration using EdgeConnect.
"""

import torch
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from edgeconnect_model import EdgeConnect

class InteractiveEdgeConnect:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 EdgeConnect Interactive Inference")
        print(f"📱 Using device: {self.device}")
        
        # Load EdgeConnect model
        self.load_edgeconnect_model()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Data directories
        self.damaged_dir = "./data/train_damaged"
        self.mask_dir = "./data/train_masks"
        self.gt_dir = "./data/train_ground_truth"
        self.output_dir = "./edgeconnect_interactive_results"
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_edgeconnect_model(self):
        """Load the trained EdgeConnect model"""
        print("\n📦 Loading EdgeConnect model...")
        
        try:
            self.edgeconnect_model = EdgeConnect(device=self.device, lr=0.0002)
            checkpoint_path = "sequential_checkpoints_200epochs/EdgeConnect_final.pth"
            
            if os.path.exists(checkpoint_path):
                self.edgeconnect_model.load_checkpoint(checkpoint_path)
                self.edgeconnect_model.edge_generator.eval()
                self.edgeconnect_model.inpaint_generator.eval()
                print(f"✅ EdgeConnect model loaded from: {checkpoint_path}")
            else:
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"EdgeConnect checkpoint not found")
                
        except Exception as e:
            print(f"❌ Failed to load EdgeConnect model: {e}")
            raise
    
    def get_available_images(self):
        """Get list of available damaged images"""
        if not os.path.exists(self.damaged_dir):
            print(f"❌ Damaged images directory not found: {self.damaged_dir}")
            return []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(self.damaged_dir, ext)))
            images.extend(glob.glob(os.path.join(self.damaged_dir, ext.upper())))
        
        return sorted(images)
    
    def display_image_menu(self, images):
        """Display menu of available images"""
        print(f"\n📋 Available Images ({len(images)} found):")
        print("-" * 50)
        
        for i, img_path in enumerate(images):
            filename = os.path.basename(img_path)
            print(f"{i+1:3d}. {filename}")
        
        print("-" * 50)
        print(f"{len(images)+1:3d}. Process ALL images (batch mode)")
        print(f"{len(images)+2:3d}. Exit")
        
        return images
    
    def load_image_and_mask(self, image_path):
        """Load damaged image and corresponding mask"""
        try:
            # Load damaged image
            damaged_img = Image.open(image_path).convert('RGB')
            
            # Find corresponding mask
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
            
            if not os.path.exists(mask_path):
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                    alt_mask_path = os.path.join(self.mask_dir, f"{base_name}{ext}")
                    if os.path.exists(alt_mask_path):
                        mask_path = alt_mask_path
                        break
            
            if not os.path.exists(mask_path):
                print(f"⚠️ Mask not found for {base_name}, creating default mask")
                # Create a default mask (center hole)
                mask_img = Image.new('L', damaged_img.size, 255)
                # Create a hole in the center
                w, h = damaged_img.size
                hole_size = min(w, h) // 4
                x1, y1 = w//2 - hole_size//2, h//2 - hole_size//2
                x2, y2 = x1 + hole_size, y1 + hole_size
                
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask_img)
                draw.rectangle([x1, y1, x2, y2], fill=0)
            else:
                mask_img = Image.open(mask_path).convert('L')
            
            # Apply transforms
            damaged_tensor = self.transform(damaged_img).unsqueeze(0).to(self.device)
            mask_tensor = self.mask_transform(mask_img).unsqueeze(0).to(self.device)
            
            # Convert mask to binary (0 for holes, 1 for valid regions)
            mask_tensor = (mask_tensor > 0.5).float()
            
            return damaged_tensor, mask_tensor, damaged_img, mask_img
            
        except Exception as e:
            print(f"❌ Error loading image/mask: {e}")
            return None, None, None, None
    
    def run_edgeconnect_inference(self, damaged_tensor, mask_tensor):
        """Run EdgeConnect inference using the built-in inference method"""
        print("🔄 Running EdgeConnect inference...")
        
        try:
            with torch.no_grad():
                # Use the built-in inference method
                output = self.edgeconnect_model.inference(damaged_tensor, mask_tensor)
                
            print("✅ EdgeConnect inference complete")
            return output
            
        except Exception as e:
            print(f"❌ EdgeConnect inference failed: {e}")
            return None
    
    def save_results(self, damaged_img, mask_img, result_tensor, base_name):
        """Save inference results"""
        try:
            # Convert result tensor to PIL image
            result_np = result_tensor[0].cpu().permute(1, 2, 0).numpy()
            result_np = np.clip(result_np, 0, 1)
            result_img = Image.fromarray((result_np * 255).astype(np.uint8))
            
            # Create comparison
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(damaged_img)
            axes[0].set_title('Damaged Input')
            axes[0].axis('off')
            
            axes[1].imshow(mask_img, cmap='gray')
            axes[1].set_title('Mask')
            axes[1].axis('off')
            
            axes[2].imshow(result_img)
            axes[2].set_title('EdgeConnect Output')
            axes[2].axis('off')
            
            # Load ground truth if available
            gt_path = os.path.join(self.gt_dir, f"{base_name}.png")
            if not os.path.exists(gt_path):
                for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                    alt_gt_path = os.path.join(self.gt_dir, f"{base_name}{ext}")
                    if os.path.exists(alt_gt_path):
                        gt_path = alt_gt_path
                        break
            
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('RGB')
                gt_img = gt_img.resize(damaged_img.size, Image.LANCZOS)
                axes[3].imshow(gt_img)
                axes[3].set_title('Ground Truth')
            else:
                axes[3].text(0.5, 0.5, 'Ground Truth\nNot Available', 
                           ha='center', va='center', transform=axes[3].transAxes)
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save files
            comparison_path = os.path.join(self.output_dir, f"{base_name}_edgeconnect_comparison.png")
            result_path = os.path.join(self.output_dir, f"{base_name}_edgeconnect_result.png")
            
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            result_img.save(result_path)
            
            print(f"💾 Results saved:")
            print(f"   📸 Comparison: {comparison_path}")
            print(f"   🖼️ Result: {result_path}")
            
            return comparison_path, result_path
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None, None
    
    def run_batch_inference(self, images):
        """Run EdgeConnect inference on all images"""
        print(f"\n🚀 Starting batch processing of {len(images)} images...")
        print("=" * 60)
        
        successful_count = 0
        failed_count = 0
        
        for idx, image_path in enumerate(images):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"\n[{idx+1}/{len(images)}] Processing: {os.path.basename(image_path)}")
            
            try:
                # Check if results already exist
                comparison_path = os.path.join(self.output_dir, f"{base_name}_edgeconnect_comparison.png")
                result_path = os.path.join(self.output_dir, f"{base_name}_edgeconnect_result.png")
                
                if os.path.exists(comparison_path) and os.path.exists(result_path):
                    print(f"⏭️  Results already exist, skipping...")
                    successful_count += 1
                    continue
                
                # Load image and mask
                damaged_tensor, mask_tensor, damaged_img, mask_img = self.load_image_and_mask(image_path)
                
                if damaged_tensor is None:
                    print(f"❌ Failed to load {base_name}")
                    failed_count += 1
                    continue
                
                # Run EdgeConnect inference
                result_tensor = self.run_edgeconnect_inference(damaged_tensor, mask_tensor)
                
                if result_tensor is None:
                    print(f"❌ Inference failed for {base_name}")
                    failed_count += 1
                    continue
                
                # Save results
                comp_path, res_path = self.save_results(damaged_img, mask_img, result_tensor, base_name)
                
                if comp_path:
                    print(f"✅ Successfully processed {base_name}")
                    successful_count += 1
                else:
                    print(f"❌ Failed to save results for {base_name}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing {base_name}: {e}")
                failed_count += 1
                continue
        
        print(f"\n🎉 Batch processing complete!")
        print(f"📊 Results Summary:")
        print(f"   ✅ Successful: {successful_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   📁 Results saved to: {self.output_dir}")
        
        return successful_count, failed_count
    
    def run_interactive_session(self):
        """Main interactive session"""
        print("\n🎯 EdgeConnect Interactive Inference Session")
        print("=" * 50)
        
        while True:
            # Get available images
            images = self.get_available_images()
            
            if not images:
                print("❌ No images found in damaged directory")
                break
            
            # Display menu
            self.display_image_menu(images)
            
            try:
                choice = input(f"\n🔍 Select image (1-{len(images)+2}): ").strip()
                
                if not choice.isdigit():
                    print("⚠️ Please enter a valid number")
                    continue
                
                choice = int(choice)
                
                if choice == len(images) + 2:
                    print("👋 Goodbye!")
                    break
                
                if choice == len(images) + 1:
                    # Batch processing mode
                    print(f"\n🔄 You selected batch processing mode")
                    confirm = input(f"📝 This will process {len(images)} images. Continue? (y/n): ").strip().lower()
                    
                    if confirm in ['y', 'yes']:
                        successful, failed = self.run_batch_inference(images)
                        input("\n📋 Press Enter to return to main menu...")
                    else:
                        print("❌ Batch processing cancelled")
                    continue
                
                if choice < 1 or choice > len(images):
                    print(f"⚠️ Please enter a number between 1 and {len(images)+2}")
                    continue
                
                # Process selected image
                selected_image = images[choice - 1]
                base_name = os.path.splitext(os.path.basename(selected_image))[0]
                
                print(f"\n🖼️ Processing: {os.path.basename(selected_image)}")
                
                # Load image and mask
                damaged_tensor, mask_tensor, damaged_img, mask_img = self.load_image_and_mask(selected_image)
                
                if damaged_tensor is None:
                    print("❌ Failed to load image, trying next...")
                    continue
                
                # Run EdgeConnect inference
                result_tensor = self.run_edgeconnect_inference(damaged_tensor, mask_tensor)
                
                if result_tensor is None:
                    print("❌ Inference failed, trying next...")
                    continue
                
                # Save results
                comparison_path, result_path = self.save_results(damaged_img, mask_img, result_tensor, base_name)
                
                if comparison_path:
                    print(f"✅ Processing complete for {base_name}")
                
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Continue with another image? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '']:
                    print("👋 Session ended")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error during processing: {e}")
                continue

def main():
    """Main function"""
    try:
        interactive_edgeconnect = InteractiveEdgeConnect()
        interactive_edgeconnect.run_interactive_session()
    except Exception as e:
        print(f"❌ Failed to initialize EdgeConnect interactive inference: {e}")

if __name__ == "__main__":
    main()