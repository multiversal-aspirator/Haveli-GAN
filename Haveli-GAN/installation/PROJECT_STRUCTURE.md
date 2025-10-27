# Haveli-GAN Project Setup

## 📁 Project Structure
```
Haveli-GAN/
├── 📄 README.md                    # Project overview and documentation
├── ⚙️ INSTALLATION.md             # Detailed installation guide  
├── 🚀 QUICKSTART.md               # Quick setup guide
├── 🛠️ install.sh                  # Automated installation script
├── ✅ verify_installation.py       # Installation verification
├── 📦 download_models.py           # Model and dataset downloader
├── 🐍 environment.yml             # Conda environment specification
├── 📋 requirements.txt            # Pip requirements
├── ⚙️ config.py                   # Configuration settings
│
├── 🧠 Core Models/                 
│   ├── model.py                   # Haveli-GAN architecture
│   ├── partial_conv_model.py      # PartialConv model
│   ├── edgeconnect_model.py       # EdgeConnect model
│   ├── mat_model.py               # MAT (Mask-Aware Transformer)
│   ├── loss.py                    # Loss functions
│   └── dataset.py                 # Data loading and preprocessing
│
├── 🏋️ Training/
│   ├── train.py                   # Main training script
│   ├── train_extended.py          # Extended training options
│   ├── sequential_training.py     # Sequential model training
│   └── create_training_data.py    # Data preparation utilities
│
├── 🔮 Inference/
│   ├── inference_haveli_gan.py    # Main inference script
│   ├── inference_updated.py       # Updated inference with options
│   ├── interactive_inference.py   # Interactive restoration
│   ├── interactive_*_inference.py # Model-specific interactive tools
│   └── inference_comparison.py    # Multi-model comparison
│
├── 🖼️ GUI Applications/
│   ├── gui_haveli_gan.py          # Main GUI application
│   └── gui.py                     # Alternative GUI interface
│
├── 📊 Evaluation/
│   ├── evaluate_model.py          # Model evaluation metrics
│   ├── basic_evaluate.py          # Basic evaluation script
│   ├── model_comparison.py        # Comprehensive model comparison
│   ├── comprehensive_evaluation.py # Detailed evaluation suite
│   └── calculate_fid.py           # FID score calculation
│
├── 📂 Data Directories/
│   ├── data/
│   │   ├── train_damaged/         # Damaged painting images
│   │   ├── train_masks/           # Damage masks
│   │   └── train_ground_truth/    # Original paintings
│   ├── checkpoints/               # Trained model weights
│   ├── outputs/                   # Training outputs
│   ├── inference_results/         # Inference outputs
│   └── evaluation_results/        # Evaluation reports
│
└── 🔧 Utilities/
    ├── prepare_indian_paintings_dataset.py  # Dataset preparation
    ├── test_setup.py              # Setup testing
    └── view_results.py            # Result visualization
```

## 🎯 Key Components

### **Installation Files**
- `install.sh` - Automated setup script for Linux/macOS
- `environment.yml` - Complete conda environment specification
- `verify_installation.py` - Comprehensive installation verification
- `download_models.py` - Pre-trained model downloader

### **Core Models**
- **Haveli-GAN**: Main restoration model with style preservation
- **PartialConv**: Partial convolution-based inpainting
- **EdgeConnect**: Two-stage edge-guided restoration  
- **MAT**: Mask-aware transformer architecture

### **Training Pipeline**
- Single model training with `train.py`
- Sequential training to avoid memory issues
- Comprehensive data preparation tools
- Automatic checkpoint saving and resuming

### **Inference Options**
- Batch processing for multiple images
- Interactive selection and restoration
- Real-time GUI applications
- Multi-model comparison tools

### **Evaluation Suite**
- PSNR, SSIM, FID metrics
- Visual quality assessment
- Comprehensive model benchmarking
- Automated report generation

## 🚀 Getting Started

### 1. **Quick Installation**
```bash
git clone https://github.com/multiversal-aspirator/Haveli-GAN.git
cd Haveli-GAN
./install.sh
```

### 2. **Verify Setup**
```bash
conda activate haveli-gan
python verify_installation.py
```

### 3. **Download Resources**
```bash
python download_models.py
```

### 4. **Test Installation**
```bash
# Quick training test (1 epoch)
python train.py

# Inference demo
python inference_haveli_gan.py --demo

# GUI application
python gui_haveli_gan.py
```

## 📋 Usage Workflows

### **Training Workflow**
1. Prepare dataset in `data/` directories
2. Run `python train.py` for quick test
3. Use `python sequential_training.py` for full training
4. Monitor progress in `outputs/` directory

### **Inference Workflow**  
1. Place damaged images in appropriate directory
2. Run inference: `python inference_haveli_gan.py --input damaged.jpg --output restored.jpg`
3. Use GUI for interactive restoration
4. Compare models with comparison tools

### **Evaluation Workflow**
1. Ensure ground truth images are available
2. Run `python evaluate_model.py` for single model
3. Use `python model_comparison.py` for benchmarking
4. Review results in `evaluation_results/`

## 🔧 Configuration

Key settings in `config.py`:
- Device selection (CUDA/CPU)
- Batch sizes and learning rates
- Model hyperparameters
- File paths and directories

## 📚 Documentation

- `INSTALLATION.md` - Comprehensive setup guide
- `QUICKSTART.md` - 5-minute setup 
- `README.md` - Project overview and usage
- Code comments and docstrings throughout

## 🎨 Example Outputs

The project generates:
- **Restored Images**: High-quality painting restoration
- **Comparison Grids**: Side-by-side before/after views
- **Training Visualizations**: Loss curves and sample outputs
- **Evaluation Reports**: Quantitative quality metrics
- **Interactive Galleries**: Web-based result viewers

---

This structure provides a complete, production-ready setup for Indian heritage painting restoration using state-of-the-art deep learning models.