#!/usr/bin/env python3
"""
Download pre-trained models and datasets for Haveli-GAN
========================================================

This script downloads the necessary model checkpoints and dataset
for running Haveli-GAN inference and training.
"""

import os
import urllib.request
from pathlib import Path
import zipfile
import tarfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def create_directories():
    """Create necessary directories"""
    directories = [
        "checkpoints",
        "data/train_damaged",
        "data/train_masks", 
        "data/train_ground_truth",
        "outputs",
        "inference_results",
        "demo_restoration"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_sample_models():
    """Download sample pre-trained models"""
    print("📦 Downloading sample pre-trained models...")
    
    # Note: Replace these URLs with actual model URLs when available
    models = {
        "haveli_gan_200epochs.pth": "https://example.com/models/haveli_gan_200epochs.pth",
        "partialconv_5epochs.pth": "https://example.com/models/partialconv_5epochs.pth", 
        "edgeconnect_200epochs.pth": "https://example.com/models/edgeconnect_200epochs.pth",
        "mat_200epochs.pth": "https://example.com/models/mat_200epochs.pth"
    }
    
    checkpoint_dir = Path("checkpoints")
    
    for model_name, url in models.items():
        model_path = checkpoint_dir / model_name
        
        if model_path.exists():
            print(f"⚠️ {model_name} already exists, skipping...")
            continue
            
        print(f"📥 Downloading {model_name}...")
        try:
            # For now, create dummy files since URLs are placeholders
            print(f"⚠️ Model URLs not yet available - creating placeholder for {model_name}")
            model_path.touch()
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

def download_sample_dataset():
    """Download sample Indian paintings dataset"""
    print("🎨 Setting up sample dataset...")
    
    # For now, create sample directory structure
    sample_styles = [
        "gond", "kalighat", "kangra", "kerala", 
        "madhubani", "mandana", "pichwai", "warli"
    ]
    
    data_dirs = ["train_damaged", "train_masks", "train_ground_truth"]
    
    for data_dir in data_dirs:
        dir_path = Path("data") / data_dir
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample placeholder files
        for style in sample_styles:
            for i in range(1, 4):  # 3 samples per style
                sample_file = dir_path / f"{style}{i}.jpg"
                if not sample_file.exists():
                    sample_file.touch()
                    
    print("✅ Sample dataset structure created")
    print("📝 Note: Replace placeholder files with actual images for training")

def setup_config_files():
    """Create default configuration files if they don't exist"""
    print("⚙️ Setting up configuration files...")
    
    # Create a simple config.py if it doesn't exist
    config_content = '''# Haveli-GAN Configuration
# Default settings for training and inference

# Model settings
DEVICE = "cuda"  # or "cpu"
BATCH_SIZE = 2
IMAGE_SIZE = 256
LEARNING_RATE = 0.0002

# Training settings  
NUM_EPOCHS = 200
SAVE_EVERY = 5

# Paths
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"

# Loss weights
LAMBDA_ADV = 1
LAMBDA_L1 = 100
LAMBDA_PERC = 10
LAMBDA_STYLE = 250
'''
    
    config_path = Path("config.py")
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ Created config.py")
    else:
        print("⚠️ config.py already exists, skipping...")

def main():
    """Main download and setup function"""
    print("=" * 60)
    print("Haveli-GAN Model and Dataset Downloader")
    print("=" * 60)
    print()
    
    # Create directories
    create_directories()
    print()
    
    # Download models
    download_sample_models()
    print()
    
    # Download dataset
    download_sample_dataset()
    print()
    
    # Setup config files
    setup_config_files()
    print()
    
    print("=" * 60)
    print("✅ Download and setup completed!")
    print("=" * 60)
    print()
    print("📝 Next steps:")
    print("1. Replace placeholder model files with actual trained models")
    print("2. Add your Indian painting images to the data directories:")
    print("   - data/train_damaged/     (damaged paintings)")
    print("   - data/train_masks/       (damage masks)")
    print("   - data/train_ground_truth/ (original paintings)")
    print("3. Run 'python verify_installation.py' to check setup")
    print("4. Start training with 'python train.py' or inference with 'python inference_haveli_gan.py --demo'")

if __name__ == "__main__":
    main()