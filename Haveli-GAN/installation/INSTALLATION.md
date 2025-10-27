# Haveli-GAN Installation Guide

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Manual Installation](#manual-installation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Getting Started](#getting-started)

## 🖥️ System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3060 or better)
- **VRAM**: Minimum 6GB GPU memory (8GB+ recommended)
- **RAM**: Minimum 16GB system RAM (32GB recommended)
- **Storage**: At least 10GB free space

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS
- **CUDA**: Version 12.1 or higher (up to 13.x supported)
- **Python**: 3.10 (required)
- **Git**: For cloning the repository

## 🚀 Quick Installation

### Option 1: Using the Installation Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/multiversal-aspirator/Haveli-GAN.git
cd Haveli-GAN

# Navigate to installation directory
cd installation

# Make installation script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

### Option 2: Using Conda Environment File

```bash
# Clone the repository
git clone https://github.com/multiversal-aspirator/Haveli-GAN.git
cd Haveli-GAN

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate haveli-gan

# Verify installation
python installation/verify_installation.py
```

## 🔧 Manual Installation

### Step 1: Install Miniconda/Anaconda

**Linux/macOS:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

**Windows:**
Download and install from: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

### Step 2: Clone Repository
```bash
git clone https://github.com/multiversal-aspirator/Haveli-GAN.git
cd Haveli-GAN
```

### Step 3: Create Python Environment
```bash
conda create -n haveli-gan python=3.10
conda activate haveli-gan
```

### Step 4: Install CUDA-enabled PyTorch
```bash
# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CUDA 12.4+ (if you have newer CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 5: Install Core Dependencies
```bash
# Install from conda
conda install numpy scipy matplotlib opencv scikit-image seaborn tk mkl mkl-service blas libjpeg-turbo -c conda-forge

# Install from pip
pip install tqdm Pillow opencv-python argparse jupyter ipython notebook
```

### Step 6: Download Pre-trained Models (Optional)
```bash
# Download the dataset and pre-trained checkpoints
python download_models.py
```

## ✅ Verification

Run the verification script to ensure everything is properly installed:

```bash
conda activate haveli-gan
python verify_installation.py
```

Expected output:
```
✅ Python 3.10 detected
✅ PyTorch 2.5.0+ installed with CUDA support
✅ All required packages available
✅ GPU detected: [Your GPU Name]
✅ CUDA version: 12.x
✅ Installation verified successfully!
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Version Mismatch
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 2: SSL Certificate Errors
```bash
# Update certificates
conda update ca-certificates
conda update certifi
```

#### Issue 3: MKL Library Not Found
```bash
# Install MKL libraries
conda install mkl mkl-service blas
```

#### Issue 4: Import Errors
```bash
# Reinstall problematic packages
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.12.0.88
```

#### Issue 5: Memory Issues During Training
- Reduce batch size in training scripts
- Close other GPU-intensive applications
- Use CPU-only mode if necessary:
```python
# In training scripts, change:
DEVICE = "cpu"  # instead of "cuda"
```

### Environment Variables (if needed)
```bash
# Add to your ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 🎯 Getting Started

### 1. Prepare Your Data
```bash
# Place your damaged Indian paintings in:
mkdir -p data/train_damaged
mkdir -p data/train_masks  
mkdir -p data/train_ground_truth

# Or use the sample dataset
python prepare_indian_paintings_dataset.py
```

### 2. Train the Model (Quick Test)
```bash
# Run 1-epoch training to test setup
python train.py  # Will train for 1 epoch by default

# For full training (200 epochs)
python train_extended.py
```

### 3. Run Inference
```bash
# Interactive inference
python interactive_inference.py

# Batch inference
python inference_haveli_gan.py --demo

# GUI application
python gui_haveli_gan.py
```

### 4. Evaluate Models
```bash
# Basic evaluation
python basic_evaluate.py

# Comprehensive comparison
python model_comparison.py
```

## 📁 Project Structure
```
Haveli-GAN/
├── README.md                 # Project overview
├── INSTALLATION.md          # This file
├── environment.yml          # Conda environment
├── requirements.txt         # Pip requirements
├── install.sh              # Installation script
├── verify_installation.py   # Verification script
├── download_models.py       # Model downloader
├── model.py                 # Core Haveli-GAN model
├── train.py                 # Training script
├── dataset.py               # Dataset handling
├── inference_haveli_gan.py  # Inference script
├── gui_haveli_gan.py        # GUI application
├── data/                    # Training data
├── checkpoints/             # Model checkpoints
└── outputs/                 # Generated results
```

## 🔗 Additional Resources
- **Paper**: [Link to research paper]
- **Dataset**: Indian Paintings Dataset
- **Issues**: Report problems on GitHub Issues
- **Documentation**: See README.md for detailed usage

## 📧 Support
If you encounter issues not covered here, please:
1. Check the GitHub Issues page
2. Create a new issue with detailed error messages
3. Include your system specifications and installation logs

---

**Note**: This project requires significant computational resources for training. For demonstration purposes, pre-trained models are available for download.