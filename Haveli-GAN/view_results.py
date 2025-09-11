#!/usr/bin/env python3
"""
Result Viewer for Haveli-GAN
============================

A simple script to display and compare restoration results.
"""

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def view_demo_results():
    """Display demo restoration results."""
    demo_dir = "./demo_restoration"
    
    if not os.path.exists(demo_dir):
        print("Demo results not found. Run 'python inference_haveli_gan.py --demo' first.")
        return
    
    # Find comparison images
    comparison_files = [f for f in os.listdir(demo_dir) if f.startswith("demo_comparison_")]
    
    if not comparison_files:
        print("No comparison images found in demo directory.")
        return
    
    print(f"Found {len(comparison_files)} demo results:")
    
    # Create a grid of results
    fig, axes = plt.subplots(len(comparison_files), 1, figsize=(15, 5 * len(comparison_files)))
    if len(comparison_files) == 1:
        axes = [axes]
    
    for i, comp_file in enumerate(comparison_files):
        comp_path = os.path.join(demo_dir, comp_file)
        image_name = comp_file.replace("demo_comparison_", "").replace(".jpg", "")
        
        # Load and display comparison image
        img = Image.open(comp_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Restoration Result: {image_name}", fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
        # Add labels
        width = img.width
        height = img.height
        third_width = width // 3
        
        # Add text annotations
        axes[i].text(third_width//2, -20, "Damaged", ha='center', va='top', fontsize=12, fontweight='bold')
        axes[i].text(third_width + third_width//2, -20, "Ground Truth", ha='center', va='top', fontsize=12, fontweight='bold')
        axes[i].text(2*third_width + third_width//2, -20, "Restored", ha='center', va='top', fontsize=12, fontweight='bold')
        
        # Add dividing lines
        axes[i].axvline(x=third_width, color='white', linewidth=2, alpha=0.8)
        axes[i].axvline(x=2*third_width, color='white', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig("./demo_results_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Results overview saved as 'demo_results_overview.png'")

def view_single_result(image_path):
    """Display a single restoration result."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    img = Image.open(image_path)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(f"Restoration Result: {os.path.basename(image_path)}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def compare_before_after(damaged_path, restored_path):
    """Compare damaged and restored images side by side."""
    if not os.path.exists(damaged_path):
        print(f"Damaged image not found: {damaged_path}")
        return
    if not os.path.exists(restored_path):
        print(f"Restored image not found: {restored_path}")
        return
    
    damaged_img = Image.open(damaged_path)
    restored_img = Image.open(restored_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(damaged_img)
    ax1.set_title("Damaged Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(restored_img)
    ax2.set_title("Restored Image", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle("Haveli-GAN Restoration Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def list_available_results():
    """List all available restoration results."""
    print("=== Available Restoration Results ===\n")
    
    # Demo results
    demo_dir = "./demo_restoration"
    if os.path.exists(demo_dir):
        demo_files = [f for f in os.listdir(demo_dir) if f.startswith("demo_restored_")]
        print(f"Demo Results ({len(demo_files)} images):")
        for f in demo_files:
            print(f"  - {f}")
        print()
    
    # Batch results
    batch_dir = "./batch_restorations"
    if os.path.exists(batch_dir):
        batch_files = [f for f in os.listdir(batch_dir) if f.endswith(".jpg")]
        print(f"Batch Results ({len(batch_files)} images):")
        for f in batch_files:
            print(f"  - {f}")
        print()
    
    # Single results
    single_files = [f for f in os.listdir(".") if f.startswith("test_restoration")]
    if single_files:
        print(f"Single Results ({len(single_files)} images):")
        for f in single_files:
            print(f"  - {f}")

def main():
    """Main function with menu."""
    print("Haveli-GAN Result Viewer")
    print("=" * 25)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "demo":
            view_demo_results()
        elif command == "list":
            list_available_results()
        elif command == "compare" and len(sys.argv) == 4:
            compare_before_after(sys.argv[2], sys.argv[3])
        elif command == "view" and len(sys.argv) == 3:
            view_single_result(sys.argv[2])
        else:
            print("Invalid command or arguments")
            print_usage()
    else:
        print_usage()

def print_usage():
    """Print usage instructions."""
    print("\nUsage:")
    print("  python view_results.py demo                    # View demo results overview")
    print("  python view_results.py list                    # List all available results")
    print("  python view_results.py view <image_path>       # View a single result")
    print("  python view_results.py compare <before> <after> # Compare before and after")
    print("\nExamples:")
    print("  python view_results.py demo")
    print("  python view_results.py view test_restoration.jpg")
    print("  python view_results.py compare ./data/train_damaged/gond11.jpg test_restoration.jpg")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install matplotlib pillow")
    except Exception as e:
        print(f"Error: {e}")
