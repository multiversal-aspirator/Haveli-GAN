#!/usr/bin/env python3
"""
Data augmentation script to create artificial damage for training
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random

def create_artificial_damage(image_path, output_damaged_path, output_mask_path):
    """Create artificial damage and corresponding mask"""
    # Load image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Create mask for damage
    mask = np.zeros((h, w), dtype=np.uint8)
    damaged_img = img.copy()
    
    # Add random scratches
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        thickness = random.randint(2, 8)
        cv2.line(damaged_img, (x1, y1), (x2, y2), (0, 0, 0), thickness)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    
    # Add random spots/holes
    for _ in range(random.randint(3, 10)):
        x, y = random.randint(0, w), random.randint(0, h)
        radius = random.randint(5, 20)
        cv2.circle(damaged_img, (x, y), radius, (0, 0, 0), -1)
        cv2.circle(mask, (x, y), radius, 255, -1)
    
    # Add color degradation
    degradation = np.random.normal(0.8, 0.1, damaged_img.shape)
    degradation = np.clip(degradation, 0.3, 1.0)
    damaged_img = (damaged_img * degradation).astype(np.uint8)
    
    # Save results
    cv2.imwrite(output_damaged_path, damaged_img)
    cv2.imwrite(output_mask_path, mask)

def process_good_frescoes(input_dir, output_dir):
    """Process all good frescoes to create training data"""
    os.makedirs(f"{output_dir}/train_damaged", exist_ok=True)
    os.makedirs(f"{output_dir}/train_ground_truth", exist_ok=True)
    os.makedirs(f"{output_dir}/train_masks", exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            
            # Copy original as ground truth
            gt_path = os.path.join(f"{output_dir}/train_ground_truth", filename)
            os.system(f"cp '{input_path}' '{gt_path}'")
            
            # Create damaged version and mask
            damaged_path = os.path.join(f"{output_dir}/train_damaged", filename)
            mask_path = os.path.join(f"{output_dir}/train_masks", filename)
            
            create_artificial_damage(input_path, damaged_path, mask_path)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    # Usage example
    input_directory = "./good_frescoes"  # Your good fresco images
    output_directory = "./data"          # Training data output
    
    print("Creating artificial damage for training...")
    process_good_frescoes(input_directory, output_directory)
    print("Done! Training data created.")
