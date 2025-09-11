# Haveli-GAN: AI-Powered Fresco Restoration System

A state-of-the-art Generative Adversarial Network (GAN) for restoring damaged traditional Indian paintings and frescoes. This system specializes in cultural heritage preservation using deep learning techniques.

## 🎨 Overview

Haveli-GAN is designed to restore damaged artwork from various traditional Indian painting styles including:
- **Gond** - Tribal art from Central India
- **Kalighat** - Bengali folk art
- **Kangra** - Pahari miniature paintings
- **Kerala Mural** - Traditional temple art
- **Madhubani** - Folk art from Bihar
- **Mandana** - Rajasthani floor paintings
- **Pichwai** - Devotional cloth paintings
- **Warli** - Maharashtra tribal art

## 🏗️ Architecture

The system consists of three main components:

1. **Generator**: GatedConv2d-based architecture for inpainting damaged regions
2. **Discriminator**: Multi-scale discriminator for realistic output generation
3. **Style Encoder**: Extracts style features from reference images

### Loss Functions
- **Adversarial Loss**: Ensures realistic generation
- **L1 Loss**: Pixel-level reconstruction accuracy
- **Perceptual Loss**: VGG-based feature matching
- **Style Loss**: Gram matrix-based style preservation

## 📋 Requirements

### System Requirements
- CUDA-compatible GPU (recommended)
- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install pillow opencv-python matplotlib
pip install tqdm argparse
pip install tkinter  # For GUI (usually included with Python)
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n haveli-gan python=3.10
conda activate haveli-gan

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional dependencies
pip install pillow opencv-python matplotlib tqdm
```

### 2. Training (if needed)
```bash
# Ensure dataset is properly organized in ./data/
python train.py
```

### 3. Inference Options

#### Command-Line Interface
```bash
# Single image restoration
python inference_haveli_gan.py --input damaged_image.jpg --output restored_image.jpg

# Batch processing
python inference_haveli_gan.py --input_dir ./damaged_images/ --output_dir ./restored_images/

# Demo with sample images
python inference_haveli_gan.py --demo

# With custom mask
python inference_haveli_gan.py --input damaged.jpg --mask damage_mask.jpg --output restored.jpg
```

#### Graphical User Interface
```bash
python gui_haveli_gan.py
```

## 📁 Project Structure

```
Haveli-GAN/
├── model.py              # GAN architecture (Generator, Discriminator, StyleEncoder)
├── loss.py               # Loss functions (VGG Perceptual, Style Loss)
├── dataset.py            # Data loading and preprocessing
├── train.py              # Training script
├── inference_haveli_gan.py   # Command-line inference
├── gui_haveli_gan.py     # Graphical user interface
├── view_results.py       # Result visualization utility
├── data/                 # Training dataset
│   ├── train_damaged/    # Damaged paintings
│   ├── train_ground_truth/   # Original paintings
│   └── train_masks/      # Damage masks
├── checkpoints/          # Trained model weights
├── outputs/              # Training outputs
└── demo_restoration/     # Demo results
```

## 🎯 Usage Examples

### 1. Demo Mode
The easiest way to see the system in action:
```bash
python inference_haveli_gan.py --demo
```
This processes 5 sample images and creates before/after comparisons.

### 2. Single Image Restoration
```bash
python inference_haveli_gan.py \
    --input ./data/train_damaged/gond11.jpg \
    --output ./my_restoration.jpg
```

### 3. Batch Processing
```bash
python inference_haveli_gan.py \
    --input_dir ./my_damaged_paintings/ \
    --output_dir ./my_restorations/
```

### 4. GUI Mode
```bash
python gui_haveli_gan.py
```
Provides a user-friendly interface with:
- Image browsing and selection
- Real-time processing status
- Side-by-side result comparison
- Easy saving of results

### 5. View Results
```bash
# View demo results overview
python view_results.py demo

# Compare before and after
python view_results.py compare damaged.jpg restored.jpg

# List all available results
python view_results.py list
```

## 🔧 Advanced Configuration

### Model Checkpoints
The system automatically uses the latest checkpoint from `./checkpoints/`. To use a specific checkpoint:
```bash
python inference_haveli_gan.py \
    --input damaged.jpg \
    --output restored.jpg \
    --checkpoint ./checkpoints/checkpoint_epoch_50.pth
```

### Automatic Mask Generation
If no mask is provided, the system automatically detects damaged regions using computer vision techniques:
- Edge detection for cracks and tears
- Color analysis for fading and discoloration
- Texture analysis for missing areas

### Style Transfer
The system can use reference images for style guidance:
```bash
python inference_haveli_gan.py \
    --input damaged.jpg \
    --reference style_reference.jpg \
    --output restored.jpg
```

## 🎨 Training Your Own Model

### Dataset Preparation
1. Organize images in the following structure:
```
data/
├── train_damaged/      # Artificially damaged or naturally damaged images
├── train_ground_truth/ # Clean, original images
└── train_masks/        # Binary masks indicating damaged areas
```

2. Ensure corresponding files have the same names across directories.

### Training Process
```bash
python train.py
```

### Training Parameters (in train.py)
- `BATCH_SIZE = 2` - Adjust based on GPU memory
- `LEARNING_RATE_G = 1e-4` - Generator learning rate
- `LEARNING_RATE_D = 1e-4` - Discriminator learning rate
- `NUM_EPOCHS = 100` - Training duration
- `IMAGE_SIZE = 256` - Input image resolution

## 📊 Model Performance

### Training Metrics (100 epochs)
- **Discriminator Loss**: 0.5503
- **Generator Adversarial Loss**: 0.8028
- **L1 Reconstruction Loss**: 0.0438 (excellent)
- **Perceptual Loss**: 0.7809
- **Style Loss**: 0.0102

### Inference Speed
- **GPU (CUDA)**: ~0.13-0.31 seconds per image
- **CPU**: ~2-5 seconds per image (depending on hardware)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training
   - Use CPU mode: `--device cpu`

2. **Missing Dependencies**
   ```bash
   pip install torch torchvision pillow opencv-python matplotlib tqdm
   ```

3. **Checkpoint Loading Errors**
   - Ensure checkpoint file exists
   - Check file permissions
   - Verify model architecture matches

4. **Poor Restoration Quality**
   - Try different epochs (earlier checkpoints sometimes work better)
   - Ensure input image quality is reasonable
   - Check if the art style is represented in training data

### Performance Optimization
- Use GPU for faster inference
- Process images in batches for efficiency
- Resize very large images before processing

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for additional art styles
- Enhanced mask generation algorithms
- Mobile/web deployment
- Real-time processing optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Traditional Indian artists whose work inspired this project
- Cultural heritage preservation organizations
- The computer vision and deep learning communities

## 📚 Technical Details

### Model Architecture
- **Generator**: U-Net with Gated Convolutions
- **Input Resolution**: 256x256 pixels
- **Output Channels**: RGB (3 channels)
- **Training Data**: 400 paintings across 8 traditional styles

### File Formats Supported
- **Input**: JPG, JPEG, PNG, BMP, TIFF
- **Output**: JPG (default), PNG
- **Quality**: 95% JPEG quality for optimal results

### System Requirements
- **Minimum RAM**: 8GB
- **Recommended RAM**: 16GB+
- **GPU VRAM**: 4GB+ for training, 2GB+ for inference
- **Storage**: 2GB for model and dependencies

---

**Note**: This system is designed for cultural heritage preservation and should be used respectfully with appropriate permissions when working with historical artwork.
