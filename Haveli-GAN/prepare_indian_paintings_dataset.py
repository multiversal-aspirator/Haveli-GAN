#!/usr/bin/env python3
"""
Prepare Indian Paintings Dataset for Haveli-GAN Training

This script processes the Indian_Paintings_Dataset and creates training data
by applying artificial damage to good-condition paintings from all categories.
"""

import os
import cv2
import numpy as np
import random
from PIL import Image
import argparse
from tqdm import tqdm
import shutil

class ArtificialDamageGenerator:
    """Generate realistic damage patterns on paintings"""
    
    def __init__(self):
        self.damage_types = [
            'scratches', 'holes', 'stains', 'color_fade', 
            'cracks', 'dirt', 'missing_parts'
        ]
    
    def add_scratches(self, image, intensity=0.3):
        """Add scratch-like damage"""
        h, w = image.shape[:2]
        damaged = image.copy()
        
        # Number of scratches based on intensity
        num_scratches = int(5 + intensity * 15)
        
        for _ in range(num_scratches):
            # Random scratch parameters
            start_x = random.randint(0, w-1)
            start_y = random.randint(0, h-1)
            length = random.randint(20, min(w, h) // 4)
            angle = random.uniform(0, 2 * np.pi)
            thickness = random.randint(1, 4)
            
            # Calculate end point
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            end_x = max(0, min(w-1, end_x))
            end_y = max(0, min(h-1, end_y))
            
            # Draw scratch (darker line)
            color = tuple(max(0, c - random.randint(30, 80)) for c in [128, 128, 128])
            cv2.line(damaged, (start_x, start_y), (end_x, end_y), color, thickness)
            
        return damaged
    
    def add_holes(self, image, intensity=0.2):
        """Add hole-like damage"""
        h, w = image.shape[:2]
        damaged = image.copy()
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Number of holes based on intensity
        num_holes = int(1 + intensity * 8)
        
        for _ in range(num_holes):
            # Random hole parameters
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
            radius = random.randint(5, min(w, h) // 20)
            
            # Create hole (black circle)
            cv2.circle(damaged, (center_x, center_y), radius, (0, 0, 0), -1)
            cv2.circle(mask, (center_x, center_y), radius, 0, -1)
            
        return damaged, mask
    
    def add_stains(self, image, intensity=0.25):
        """Add stain-like discoloration"""
        h, w = image.shape[:2]
        damaged = image.copy().astype(np.float32)
        
        # Number of stains based on intensity
        num_stains = int(2 + intensity * 8)
        
        for _ in range(num_stains):
            # Random stain parameters
            center_x = random.randint(0, w-1)
            center_y = random.randint(0, h-1)
            radius = random.randint(15, min(w, h) // 8)
            
            # Create stain mask
            stain_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(stain_mask, (center_x, center_y), radius, 1.0, -1)
            
            # Apply gaussian blur for natural stain edge
            stain_mask = cv2.GaussianBlur(stain_mask, (21, 21), 0)
            
            # Random stain color (brownish/yellowish)
            stain_color = np.array([
                random.uniform(0.6, 0.9),  # Blue channel reduction
                random.uniform(0.7, 0.95), # Green channel reduction  
                random.uniform(0.8, 1.0)   # Red channel preservation
            ])
            
            # Apply stain
            for c in range(3):
                damaged[:, :, c] *= (1 - stain_mask * (1 - stain_color[c]))
                
        return np.clip(damaged, 0, 255).astype(np.uint8)
    
    def add_color_fade(self, image, intensity=0.2):
        """Add color fading effect"""
        damaged = image.copy().astype(np.float32)
        
        # Reduce color saturation
        hsv = cv2.cvtColor(damaged, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] *= (1 - intensity * 0.5)  # Reduce saturation
        hsv[:, :, 2] *= (1 - intensity * 0.3)  # Slightly reduce brightness
        
        damaged = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return np.clip(damaged, 0, 255).astype(np.uint8)
    
    def add_cracks(self, image, intensity=0.3):
        """Add crack-like damage"""
        h, w = image.shape[:2]
        damaged = image.copy()
        
        # Number of crack systems
        num_cracks = int(2 + intensity * 6)
        
        for _ in range(num_cracks):
            # Start from random edge
            if random.choice([True, False]):
                start_x = random.choice([0, w-1])
                start_y = random.randint(0, h-1)
            else:
                start_x = random.randint(0, w-1)
                start_y = random.choice([0, h-1])
            
            # Create branching crack
            self._draw_crack_branch(damaged, start_x, start_y, 
                                  random.uniform(0, 2*np.pi), 
                                  random.randint(30, min(w, h)//3), 
                                  depth=0, max_depth=2)
        
        return damaged
    
    def _draw_crack_branch(self, image, x, y, angle, length, depth, max_depth):
        """Draw a single crack branch with potential sub-branches"""
        h, w = image.shape[:2]
        
        # Draw main crack line
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        end_x = max(0, min(w-1, end_x))
        end_y = max(0, min(h-1, end_y))
        
        # Dark crack line
        color = tuple(max(0, c - random.randint(40, 100)) for c in [64, 64, 64])
        cv2.line(image, (x, y), (end_x, end_y), color, 1)
        
        # Add sub-branches
        if depth < max_depth and random.random() < 0.6:
            branch_point_x = int(x + (end_x - x) * random.uniform(0.3, 0.7))
            branch_point_y = int(y + (end_y - y) * random.uniform(0.3, 0.7))
            
            # Create 1-2 sub-branches
            for _ in range(random.randint(1, 2)):
                branch_angle = angle + random.uniform(-np.pi/3, np.pi/3)
                branch_length = length // 2
                self._draw_crack_branch(image, branch_point_x, branch_point_y,
                                      branch_angle, branch_length, depth+1, max_depth)
    
    def add_dirt_accumulation(self, image, intensity=0.2):
        """Add dirt and grime accumulation"""
        h, w = image.shape[:2]
        damaged = image.copy().astype(np.float32)
        
        # Create dirt pattern using noise
        dirt = np.random.random((h, w)) * intensity
        dirt = cv2.GaussianBlur(dirt, (15, 15), 0)
        
        # Apply dirt (darkening effect)
        for c in range(3):
            damaged[:, :, c] *= (1 - dirt)
            
        return np.clip(damaged, 0, 255).astype(np.uint8)
    
    def generate_damage(self, image, damage_level='medium'):
        """Generate comprehensive damage on an image"""
        # Define damage intensity based on level
        intensity_map = {
            'light': 0.2,
            'medium': 0.4, 
            'heavy': 0.7
        }
        base_intensity = intensity_map.get(damage_level, 0.4)
        
        damaged = image.copy()
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Randomly select and apply damage types
        selected_damages = random.sample(self.damage_types, 
                                       random.randint(3, len(self.damage_types)))
        
        for damage_type in selected_damages:
            intensity = base_intensity * random.uniform(0.5, 1.5)
            
            if damage_type == 'scratches':
                damaged = self.add_scratches(damaged, intensity)
            elif damage_type == 'holes':
                damaged, hole_mask = self.add_holes(damaged, intensity * 0.5)
                mask = cv2.bitwise_and(mask, hole_mask)
            elif damage_type == 'stains':
                damaged = self.add_stains(damaged, intensity)
            elif damage_type == 'color_fade':
                damaged = self.add_color_fade(damaged, intensity)
            elif damage_type == 'cracks':
                damaged = self.add_cracks(damaged, intensity)
            elif damage_type == 'dirt':
                damaged = self.add_dirt_accumulation(damaged, intensity)
            # 'missing_parts' is handled by holes
        
        return damaged, mask

def process_image(image_path, output_damaged_dir, output_gt_dir, output_mask_dir, 
                 damage_generator, target_size=(256, 256)):
    """Process a single image to create training triplet"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load {image_path}")
            return False
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Generate damage
        damaged_image, mask = damage_generator.generate_damage(
            image, damage_level=random.choice(['light', 'medium', 'heavy'])
        )
        
        # Create output filename
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_name = f"{name}.jpg"  # Standardize to .jpg
        
        # Save images
        cv2.imwrite(os.path.join(output_damaged_dir, output_name), damaged_image)
        cv2.imwrite(os.path.join(output_gt_dir, output_name), image)
        cv2.imwrite(os.path.join(output_mask_dir, output_name), mask)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def create_training_dataset(source_dir, output_dir, max_images_per_category=None):
    """Create training dataset from Indian paintings"""
    
    # Categories to process
    categories = [
        'gond painting', 'kalighat painting', 'kangra painting', 'kerala mural',
        'madhubani painting', 'mandana art drawing', 'pichwai painting', 'warli painting'
    ]
    
    # Create output directories
    output_damaged_dir = os.path.join(output_dir, 'train_damaged')
    output_gt_dir = os.path.join(output_dir, 'train_ground_truth')
    output_mask_dir = os.path.join(output_dir, 'train_masks')
    
    for dir_path in [output_damaged_dir, output_gt_dir, output_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize damage generator
    damage_generator = ArtificialDamageGenerator()
    
    total_processed = 0
    category_counts = {}
    
    print(f"Processing Indian Paintings Dataset from: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Categories to process: {categories}")
    
    for category in categories:
        category_dir = os.path.join(source_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory not found: {category_dir}")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.JPG', '.JPEG']:
            image_files.extend([f for f in os.listdir(category_dir) if f.endswith(ext)])
        
        if max_images_per_category:
            image_files = image_files[:max_images_per_category]
        
        print(f"\nProcessing {category}: {len(image_files)} images")
        category_processed = 0
        
        for image_file in tqdm(image_files, desc=f"Processing {category}"):
            image_path = os.path.join(category_dir, image_file)
            
            # Create unique filename with category prefix
            name, ext = os.path.splitext(image_file)
            category_short = category.replace(' ', '_').replace(' painting', '').replace(' art drawing', '')
            unique_name = f"{category_short}_{name}"
            
            # Process image
            if process_image(image_path, output_damaged_dir, output_gt_dir, output_mask_dir,
                           damage_generator, target_size=(256, 256)):
                category_processed += 1
                total_processed += 1
        
        category_counts[category] = category_processed
        print(f"Successfully processed {category_processed} images from {category}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("DATASET CREATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {total_processed}")
    print("\nImages per category:")
    for category, count in category_counts.items():
        print(f"  {category:25}: {count:3d} images")
    
    print(f"\nTraining data saved to:")
    print(f"  Damaged images: {output_damaged_dir}")
    print(f"  Ground truth:   {output_gt_dir}")
    print(f"  Masks:          {output_mask_dir}")
    
    return total_processed

def main():
    parser = argparse.ArgumentParser(description='Prepare Indian Paintings Dataset for Haveli-GAN')
    parser.add_argument('--source', default='/home/amansh/SOP/Haveli-GAN/Indian_Paintings_Dataset',
                       help='Path to Indian_Paintings_Dataset directory')
    parser.add_argument('--output', default='/home/amansh/SOP/Haveli-GAN/Haveli-GAN/data',
                       help='Output directory for training data')
    parser.add_argument('--max_per_category', type=int, default=None,
                       help='Maximum images per category (default: use all)')
    parser.add_argument('--test_run', action='store_true',
                       help='Process only 5 images per category for testing')
    
    args = parser.parse_args()
    
    if args.test_run:
        args.max_per_category = 5
        print("Test run mode: processing 5 images per category")
    
    # Verify source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory does not exist: {args.source}")
        return 1
    
    # Create training dataset
    total_images = create_training_dataset(
        source_dir=args.source,
        output_dir=args.output,
        max_images_per_category=args.max_per_category
    )
    
    if total_images > 0:
        print(f"\n✅ Successfully created training dataset with {total_images} images!")
        print(f"You can now train the Haveli-GAN model using: python train.py")
    else:
        print(f"\n❌ No images were processed. Please check the source directory.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
