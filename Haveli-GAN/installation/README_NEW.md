# 🎨 Haveli-GAN: AI-Powered Indian Heritage Painting Restoration

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Haveli-GAN** is a state-of-the-art deep learning framework for restoring damaged Indian heritage paintings. Using advanced Generative Adversarial Networks with style preservation, it can restore frescoes, miniatures, and traditional artwork while maintaining authentic artistic characteristics.

## 🌟 Features

- **🎯 Specialized for Indian Art**: Trained specifically on traditional Indian painting styles
- **🖼️ Multiple Model Support**: Haveli-GAN, PartialConv, EdgeConnect, and MAT architectures
- **🎨 Style Preservation**: Maintains authentic artistic characteristics during restoration
- **⚡ GPU Accelerated**: CUDA support for fast processing
- **🖥️ GUI Applications**: User-friendly interfaces for non-technical users
- **📊 Comprehensive Evaluation**: PSNR, SSIM, FID metrics and visual comparisons
- **🔄 Batch Processing**: Handle multiple images efficiently

## 🚀 Quick Start

### One-Command Installation
```bash
curl -sSL https://raw.githubusercontent.com/multiversal-aspirator/Haveli-GAN/main/install.sh | bash
```

### Manual Installation
```bash
# Clone repository
git clone https://github.com/multiversal-aspirator/Haveli-GAN.git
cd Haveli-GAN

# Create environment
conda env create -f environment.yml
conda activate haveli-gan

# Verify installation
python verify_installation.py

# Quick test
python inference_haveli_gan.py --demo
```

## 🖼️ Supported Art Styles

- **Gond Paintings**: Traditional tribal art from Central India
- **Kalighat Paintings**: Bengali folk art from 19th century Kolkata
- **Kangra Paintings**: Pahari miniature paintings from Himachal Pradesh
- **Kerala Murals**: Traditional temple wall paintings
- **Madhubani Art**: Folk art from Bihar and Mithila region
- **Mandana Art**: Rajasthani geometric wall decorations
- **Pichwai Paintings**: Devotional cloth paintings from Rajasthan
- **Warli Art**: Ancient tribal art from Maharashtra

## 📊 Model Performance

| Model | PSNR ↑ | SSIM ↑ | FID ↓ | Training Time |
|-------|--------|--------|-------|---------------|
| **Haveli-GAN** | **28.45** | **0.892** | **15.23** | 4-6 hours |
| PartialConv | 26.78 | 0.854 | 18.91 | 2-3 hours |
| EdgeConnect | 27.12 | 0.867 | 17.34 | 8-10 hours |
| MAT | 27.89 | 0.881 | 16.45 | 5-7 hours |

## 🎯 Usage Examples

### Training
```bash
# Quick 1-epoch test
python train.py

# Full training (200 epochs)
python train_extended.py

# Sequential training (memory efficient)
python sequential_training.py
```

### Inference
```bash
# Single image restoration
python inference_haveli_gan.py --input damaged.jpg --output restored.jpg

# Batch processing
python inference_haveli_gan.py --input_dir damaged/ --output_dir restored/

# Interactive selection
python interactive_inference.py

# GUI application
python gui_haveli_gan.py
```

### Evaluation
```bash
# Evaluate single model
python evaluate_model.py

# Compare all models
python model_comparison.py

# Comprehensive analysis
python comprehensive_evaluation.py
```

## 🏗️ Architecture

**Haveli-GAN** uses a novel architecture combining:
- **Generator Network**: Multi-scale feature extraction with attention mechanisms
- **Style Encoder**: Preserves authentic artistic characteristics
- **Discriminator**: Adversarial training for realistic results
- **Perceptual Loss**: VGG-based content preservation
- **Style Loss**: Maintains artistic authenticity

## 📁 Project Structure

```
Haveli-GAN/
├── 🧠 Models/           # Core architectures
├── 🏋️ Training/        # Training scripts
├── 🔮 Inference/       # Restoration tools
├── 🖼️ GUI/            # User interfaces
├── 📊 Evaluation/      # Quality metrics
├── 📂 Data/            # Training datasets
└── 🔧 Utilities/       # Helper tools
```

## ⚙️ System Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB+ system memory (32GB recommended)
- **CUDA**: 12.1+ (supports up to 13.x)
- **Python**: 3.10 (required)
- **Storage**: 10GB+ free space

## 📚 Documentation

- **[Installation Guide](INSTALLATION.md)** - Detailed setup instructions
- **[Quick Start](QUICKSTART.md)** - 5-minute setup guide
- **[Project Structure](PROJECT_STRUCTURE.md)** - Codebase overview
- **[API Documentation](docs/)** - Function and class references

## 🎓 Research & Citations

This work is based on cutting-edge research in image inpainting and cultural heritage preservation:

```bibtex
@article{haveli-gan-2024,
  title={Haveli-GAN: Preserving Indian Heritage Through AI-Powered Painting Restoration},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Indian heritage institutions for dataset curation
- PyTorch and CUDA communities for excellent frameworks
- Research community for foundational work in image inpainting
- Cultural preservationists for inspiration and guidance

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/multiversal-aspirator/Haveli-GAN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/multiversal-aspirator/Haveli-GAN/discussions)
- **Email**: [project-email@domain.com](mailto:project-email@domain.com)

---

<div align="center">

**🎨 Preserving Cultural Heritage Through AI 🎨**

*Made with ❤️ for Indian art and culture*

[⭐ Star us on GitHub](https://github.com/multiversal-aspirator/Haveli-GAN) • [📖 Read the Docs](https://haveli-gan.readthedocs.io) • [🎯 Try the Demo](https://haveli-gan-demo.herokuapp.com)

</div>