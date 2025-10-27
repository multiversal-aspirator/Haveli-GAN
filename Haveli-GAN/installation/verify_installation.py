#!/usr/bin/env python3
"""
Haveli-GAN Installation Verification Script
===========================================

This script verifies that all required packages and dependencies 
are properly installed and working correctly.
"""

import sys
import importlib
import subprocess
import os
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="INFO"):
    colors = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "ERROR": Colors.RED,
        "WARNING": Colors.YELLOW
    }
    color = colors.get(status, Colors.BLUE)
    print(f"{color}[{status}]{Colors.END} {message}")

def check_python_version():
    """Check if Python version is 3.10"""
    print_status("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor == 10:
        print_status(f"✅ Python {version.major}.{version.minor}.{version.micro} detected", "SUCCESS")
        return True
    else:
        print_status(f"❌ Python {version.major}.{version.minor}.{version.micro} detected, but 3.10 is required", "ERROR")
        return False

def check_package_import(package_name, import_name=None, version_attr=None):
    """Check if a package can be imported and optionally check version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Check version if specified
        version_info = ""
        if version_attr:
            try:
                version = getattr(module, version_attr)
                version_info = f" (v{version})"
            except AttributeError:
                version_info = " (version unknown)"
        
        print_status(f"✅ {package_name}{version_info}", "SUCCESS")
        return True, module
    except ImportError as e:
        print_status(f"❌ {package_name} - Import failed: {e}", "ERROR")
        return False, None

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    print_status("Checking PyTorch installation...")
    
    success, torch = check_package_import("PyTorch", "torch", "__version__")
    if not success:
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        
        print_status(f"✅ CUDA {cuda_version} available", "SUCCESS")
        print_status(f"✅ {gpu_count} GPU(s) detected: {gpu_name}", "SUCCESS")
        
        # Test GPU memory
        try:
            device = torch.device('cuda:0')
            x = torch.randn(100, 100, device=device)
            print_status("✅ GPU memory test passed", "SUCCESS")
        except Exception as e:
            print_status(f"⚠️ GPU memory test failed: {e}", "WARNING")
    else:
        print_status("⚠️ CUDA not available - will use CPU", "WARNING")
    
    return True

def check_core_packages():
    """Check all core packages required for the project"""
    print_status("Checking core packages...")
    
    packages = [
        ("NumPy", "numpy", "__version__"),
        ("Pillow/PIL", "PIL", "__version__"),
        ("OpenCV", "cv2", "__version__"),
        ("Matplotlib", "matplotlib", "__version__"),
        ("SciPy", "scipy", "__version__"),
        ("Scikit-image", "skimage", "__version__"),
        ("Seaborn", "seaborn", "__version__"),
        ("tqdm", "tqdm", "__version__"),
    ]
    
    all_success = True
    for package_name, import_name, version_attr in packages:
        success, _ = check_package_import(package_name, import_name, version_attr)
        if not success:
            all_success = False
    
    return all_success

def check_optional_packages():
    """Check optional packages"""
    print_status("Checking optional packages...")
    
    optional_packages = [
        ("Jupyter", "jupyter", None),
        ("IPython", "IPython", "__version__"),
        ("tkinter (GUI support)", "tkinter", None),
    ]
    
    for package_name, import_name, version_attr in optional_packages:
        check_package_import(package_name, import_name, version_attr)

def check_model_files():
    """Check if model files and directories exist"""
    print_status("Checking project structure...")
    
    required_files = [
        "model.py",
        "train.py", 
        "dataset.py",
        "loss.py",
        "inference_haveli_gan.py",
        "environment.yml"
    ]
    
    required_dirs = [
        "data",
        "checkpoints",
        "outputs"
    ]
    
    all_exist = True
    
    # Check files
    for file_name in required_files:
        if os.path.exists(file_name):
            print_status(f"✅ {file_name} found", "SUCCESS")
        else:
            print_status(f"❌ {file_name} missing", "ERROR")
            all_exist = False
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print_status(f"✅ {dir_name}/ directory found", "SUCCESS")
        else:
            print_status(f"⚠️ {dir_name}/ directory missing - will be created when needed", "WARNING")
    
    return all_exist

def check_pretrained_models():
    """Check for pre-trained model checkpoints"""
    print_status("Checking for pre-trained models...")
    
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            print_status(f"✅ Found {len(checkpoints)} checkpoint file(s)", "SUCCESS")
            for checkpoint in checkpoints[:3]:  # Show first 3
                print_status(f"   - {checkpoint.name}", "INFO")
            if len(checkpoints) > 3:
                print_status(f"   ... and {len(checkpoints) - 3} more", "INFO")
        else:
            print_status("⚠️ No checkpoint files found - you'll need to train models first", "WARNING")
    else:
        print_status("⚠️ Checkpoints directory not found", "WARNING")

def test_basic_functionality():
    """Test basic model loading and tensor operations"""
    print_status("Testing basic functionality...")
    
    try:
        import torch
        import numpy as np
        from PIL import Image
        
        # Test tensor operations
        x = torch.randn(2, 3, 256, 256)
        y = torch.nn.functional.relu(x)
        print_status("✅ Tensor operations working", "SUCCESS")
        
        # Test PIL image creation
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        print_status("✅ PIL image operations working", "SUCCESS")
        
        # Test model import (if available)
        try:
            from model import Generator, StyleEncoder
            gen = Generator()
            style_enc = StyleEncoder()
            print_status("✅ Model classes can be imported", "SUCCESS")
        except ImportError as e:
            print_status(f"⚠️ Model import failed: {e}", "WARNING")
        
        return True
        
    except Exception as e:
        print_status(f"❌ Basic functionality test failed: {e}", "ERROR")
        return False

def check_disk_space():
    """Check available disk space"""
    print_status("Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free // (1024**3)
        
        if free_gb >= 10:
            print_status(f"✅ {free_gb}GB free space available", "SUCCESS")
        elif free_gb >= 5:
            print_status(f"⚠️ {free_gb}GB free space - recommend at least 10GB", "WARNING")
        else:
            print_status(f"❌ Only {free_gb}GB free space - need at least 5GB", "ERROR")
            
    except Exception as e:
        print_status(f"⚠️ Could not check disk space: {e}", "WARNING")

def run_verification():
    """Run complete verification process"""
    print("=" * 60)
    print(f"{Colors.BOLD}Haveli-GAN Installation Verification{Colors.END}")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_pytorch),
        ("Core Packages", check_core_packages),
        ("Optional Packages", check_optional_packages),
        ("Project Structure", check_model_files),
        ("Pre-trained Models", check_pretrained_models),
        ("Basic Functionality", test_basic_functionality),
        ("Disk Space", check_disk_space),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{Colors.BOLD}--- {check_name} ---{Colors.END}")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_status(f"❌ {check_name} check failed with exception: {e}", "ERROR")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}Verification Summary{Colors.END}")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len([r for r in results if r[1] is not None])
    
    for check_name, result in results:
        if result is True:
            print_status(f"✅ {check_name}", "SUCCESS")
        elif result is False:
            print_status(f"❌ {check_name}", "ERROR")
        else:
            print_status(f"⚠️ {check_name}", "WARNING")
    
    print(f"\n{Colors.BOLD}Result: {passed}/{total} checks passed{Colors.END}")
    
    if passed == total:
        print_status("🎉 Installation verification completed successfully!", "SUCCESS")
        print_status("You can now run the Haveli-GAN project!", "SUCCESS")
        return True
    else:
        print_status("⚠️ Some checks failed. Please review the issues above.", "WARNING")
        print_status("The project may still work, but with limited functionality.", "WARNING")
        return False

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)