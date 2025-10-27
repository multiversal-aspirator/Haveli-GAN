# Haveli-GAN Quick Setup Guide

## 🚀 One-Command Setup

For the fastest installation on a new machine:

```bash
curl -sSL https://raw.githubusercontent.com/multiversal-aspirator/Haveli-GAN/main/install.sh | bash
```

## 📋 Manual Setup (5 Minutes)

### 1. Clone Repository
```bash
git clone https://github.com/multiversal-aspirator/Haveli-GAN.git
cd Haveli-GAN
```

### 2. Create Environment
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate haveli-gan

# OR using pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python verify_installation.py
```

### 4. Quick Test
```bash
# Test training (1 epoch)
python train.py

# Test inference
python inference_haveli_gan.py --demo
```

## 🎯 What You Get

After installation, you'll have:

- ✅ **Complete ML Environment**: PyTorch + CUDA support
- ✅ **4 Restoration Models**: Haveli-GAN, PartialConv, EdgeConnect, MAT
- ✅ **Training Pipeline**: Ready-to-use training scripts
- ✅ **Inference Tools**: Interactive and batch processing
- ✅ **GUI Application**: User-friendly interface
- ✅ **Evaluation Metrics**: PSNR, SSIM, FID scoring
- ✅ **Sample Dataset**: Indian paintings for testing

## 🛠️ System Requirements

- **GPU**: NVIDIA with 6GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB+ system memory
- **CUDA**: Version 12.1+ (supports up to 13.x)
- **Python**: 3.10 (required)
- **Storage**: 10GB+ free space

## 📚 Usage Examples

```bash
# Activate environment
conda activate haveli-gan

# Train for 1 epoch (quick test)
python train.py

# Train all models sequentially
python sequential_training.py

# Interactive restoration
python interactive_inference.py

# GUI application
python gui_haveli_gan.py

# Batch processing
python inference_haveli_gan.py --input_dir damaged/ --output_dir restored/

# Model comparison
python model_comparison.py
```

## 🐛 Troubleshooting

**CUDA Issues:**
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Import Errors:**
```bash
# Reinstall problematic packages
pip install --force-reinstall opencv-python Pillow
```

**Memory Issues:**
- Reduce batch size in training scripts
- Use CPU mode: change `DEVICE = "cpu"` in scripts

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/multiversal-aspirator/Haveli-GAN/issues)
- **Documentation**: See `INSTALLATION.md` for detailed setup
- **Examples**: Check `examples/` directory

---

**Ready to restore Indian heritage paintings with AI!** 🎨