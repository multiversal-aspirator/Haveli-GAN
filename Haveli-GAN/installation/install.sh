#!/bin/bash

# Haveli-GAN Installation Script
# This script sets up the complete environment for the Haveli-GAN project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Linux detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_success "macOS detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        print_error "This script supports Linux and macOS only"
        exit 1
    fi
}

# Check if conda is installed
check_conda() {
    print_status "Checking for Conda installation..."
    if command -v conda &> /dev/null; then
        print_success "Conda found: $(conda --version)"
        return 0
    else
        print_warning "Conda not found. Installing Miniconda..."
        install_miniconda
        return 1
    fi
}

# Install Miniconda
install_miniconda() {
    print_status "Installing Miniconda..."
    
    if [[ "$OS" == "linux" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OS" == "macos" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
    
    # Download and install
    wget $MINICONDA_URL -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Add to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init
    
    print_success "Miniconda installed successfully"
    print_warning "Please restart your terminal or run 'source ~/.bashrc' to use conda"
}

# Check CUDA installation
check_cuda() {
    print_status "Checking CUDA installation..."
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
        print_success "NVIDIA GPU detected with CUDA $CUDA_VERSION"
        return 0
    else
        print_warning "NVIDIA GPU/CUDA not detected. Will install CPU-only version."
        return 1
    fi
}

# Create conda environment
create_environment() {
    print_status "Creating Haveli-GAN conda environment..."
    
    # Check if environment already exists
    if conda env list | grep -q "haveli-gan"; then
        print_warning "Environment 'haveli-gan' already exists"
        read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n haveli-gan
            print_status "Removed existing environment"
        else
            print_status "Using existing environment"
            return 0
        fi
    fi
    
    # Create environment from file if it exists, otherwise create manually
    if [[ -f "environment.yml" ]]; then
        print_status "Creating environment from environment.yml..."
        conda env create -f environment.yml
    else
        print_status "Creating environment manually..."
        conda create -n haveli-gan python=3.10 -y
    fi
    
    print_success "Environment created successfully"
}

# Install PyTorch with appropriate CUDA support
install_pytorch() {
    print_status "Installing PyTorch..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate haveli-gan
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/')
        
        if [[ $CUDA_VERSION -ge 121 ]]; then
            print_status "Installing PyTorch with CUDA 12.1+ support..."
            conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y || {
                print_warning "Conda installation failed, trying pip..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            }
        else
            print_status "Installing PyTorch with CUDA 11.8 support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        print_status "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_success "PyTorch installed successfully"
}

# Install additional dependencies
install_dependencies() {
    print_status "Installing additional dependencies..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate haveli-gan
    
    # Install conda packages
    print_status "Installing conda packages..."
    conda install -y numpy scipy matplotlib opencv scikit-image seaborn tk mkl mkl-service blas libjpeg-turbo -c conda-forge || {
        print_warning "Some conda packages failed, continuing with pip..."
    }
    
    # Install pip packages
    print_status "Installing pip packages..."
    pip install tqdm Pillow opencv-python argparse jupyter ipython notebook glob2 pyyaml requests json5
    
    print_success "Dependencies installed successfully"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/train_damaged
    mkdir -p data/train_masks
    mkdir -p data/train_ground_truth
    mkdir -p checkpoints
    mkdir -p outputs
    mkdir -p inference_results
    mkdir -p demo_restoration
    
    print_success "Directories created successfully"
}

# Fix library issues (create symlinks)
fix_libraries() {
    print_status "Fixing library issues..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate haveli-gan
    
    # Get conda environment path
    ENV_PATH=$(conda info --base)/envs/haveli-gan/lib
    
    if [[ -d "$ENV_PATH" ]]; then
        cd "$ENV_PATH"
        
        # Create MKL symlinks
        [[ -f "libmkl_intel_lp64.so.2" ]] && ln -sf libmkl_intel_lp64.so.2 libmkl_intel_lp64.so
        [[ -f "libmkl_core.so.2" ]] && ln -sf libmkl_core.so.2 libmkl_core.so
        [[ -f "libmkl_intel_thread.so.2" ]] && ln -sf libmkl_intel_thread.so.2 libmkl_intel_thread.so
        [[ -f "libmkl_gnu_thread.so.2" ]] && ln -sf libmkl_gnu_thread.so.2 libmkl_gnu_thread.so
        
        # Create JPEG symlinks
        [[ -f "libjpeg.so.9.5.0" ]] && ln -sf libjpeg.so.9.5.0 libjpeg.so.9
        [[ -f "libjpeg.so.8.2.2" ]] && ln -sf libjpeg.so.8.2.2 libjpeg.so.8
        
        cd - > /dev/null
        print_success "Library symlinks created"
    else
        print_warning "Could not find conda environment lib directory"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate haveli-gan
    
    if [[ -f "verify_installation.py" ]]; then
        python verify_installation.py
    else
        print_status "Running basic verification..."
        python -c "
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print('✅ Python version:', torch.version.python)
print('✅ PyTorch version:', torch.__version__)
print('✅ Torchvision version:', torchvision.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA version:', torch.version.cuda)
    print('✅ GPU count:', torch.cuda.device_count())
    print('✅ GPU name:', torch.cuda.get_device_name(0))

print('✅ OpenCV version:', cv2.__version__)
print('✅ NumPy version:', np.__version__)
print('✅ PIL version:', Image.__version__)
print('✅ All imports successful!')
"
    fi
    
    print_success "Installation verification completed"
}

# Main installation process
main() {
    echo "========================================="
    echo "    Haveli-GAN Installation Script"
    echo "========================================="
    echo
    
    # Check requirements
    check_os
    
    # Install conda if needed
    if ! check_conda; then
        print_warning "Please restart your terminal and run this script again"
        exit 0
    fi
    
    # Check CUDA
    check_cuda
    
    # Create environment
    create_environment
    
    # Install PyTorch
    install_pytorch
    
    # Install other dependencies
    install_dependencies
    
    # Create directories
    create_directories
    
    # Fix library issues
    fix_libraries
    
    # Verify installation
    verify_installation
    
    echo
    echo "========================================="
    print_success "Installation completed successfully!"
    echo "========================================="
    echo
    print_status "To activate the environment, run:"
    echo "    conda activate haveli-gan"
    echo
    print_status "To test the installation, run:"
    echo "    python train.py  # For 1-epoch training test"
    echo "    python inference_haveli_gan.py --demo  # For inference demo"
    echo
    print_status "For detailed usage instructions, see README.md"
}

# Run main function
main "$@"