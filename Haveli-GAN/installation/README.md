# 📦 Haveli-GAN Installation Package

This directory contains all the files needed to set up Haveli-GAN on a new machine.

## 📁 Installation Files

### **🚀 Quick Setup**
- **`install.sh`** - Automated installation script (Linux/macOS)
  ```bash
  chmod +x install.sh && ./install.sh
  ```

### **📚 Documentation**
- **`INSTALLATION.md`** - Comprehensive installation guide with troubleshooting
- **`QUICKSTART.md`** - 5-minute setup guide for experienced users
- **`PROJECT_STRUCTURE.md`** - Complete project overview and file organization
- **`README_NEW.md`** - Professional project README for GitHub

### **🔧 Setup Tools**
- **`verify_installation.py`** - Installation verification and testing script
- **`download_models.py`** - Pre-trained model and dataset downloader

## 🎯 Recommended Setup Process

### For New Users (Recommended):
1. **Download the installation package**
2. **Run the automated installer:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```
3. **Verify installation:**
   ```bash
   python verify_installation.py
   ```

### For Advanced Users:
1. **Follow QUICKSTART.md** for manual setup
2. **Use PROJECT_STRUCTURE.md** for understanding the codebase
3. **Refer to INSTALLATION.md** for detailed configuration

## 📋 What Gets Installed

After running the installation:

✅ **Python Environment**: Conda environment with Python 3.10  
✅ **PyTorch Ecosystem**: PyTorch 2.5.0+ with CUDA support  
✅ **Dependencies**: All required packages (OpenCV, PIL, NumPy, etc.)  
✅ **Project Structure**: Complete directory tree  
✅ **Pre-trained Models**: Downloaded model checkpoints  
✅ **Sample Data**: Example datasets for testing  
✅ **Configuration**: Optimized settings and library fixes  

## 🖥️ System Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended)
- **CUDA**: Version 12.1+ (supports up to 13.x)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space

## 🐛 Troubleshooting

If installation fails:
1. **Check system requirements** in INSTALLATION.md
2. **Review error messages** - most issues are dependency-related
3. **Try manual installation** following QUICKSTART.md
4. **Run verification script** to identify specific problems

## 📞 Support

- **Detailed Guide**: See `INSTALLATION.md`
- **Quick Reference**: See `QUICKSTART.md`
- **Project Info**: See `PROJECT_STRUCTURE.md`
- **Issues**: Create GitHub issue with error logs

---

**Ready to restore Indian heritage paintings with AI!** 🎨