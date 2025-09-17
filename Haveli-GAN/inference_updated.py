# inference_updated.py - Complete inference script for 200-epoch model
import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
from model import Generator, StyleEncoder

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Use the latest checkpoint from 200 epochs
CHECKPOINT_PATH = "./checkpoints/checkpoint_epoch_200.pth"
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = "./checkpoints/checkpoint_epoch_100.pth"
    print("Warning: 200-epoch checkpoint not found, using 100-epoch checkpoint")

# Example paths - can be overridden by command line arguments
DEFAULT_DAMAGED_IMG = "./data/train_damaged/gond11.jpg"
DEFAULT_MASK = "./data/train_masks/gond11.jpg"
DEFAULT_STYLE_REF = "./data/train_ground_truth/gond11.jpg"
DEFAULT_OUTPUT = "./inference_output.png"

def load_models(checkpoint_path):
    """Load Generator and StyleEncoder from unified checkpoint"""
    print(f"Loading models from {checkpoint_path}")
    
    gen = Generator().to(DEVICE)
    style_enc = StyleEncoder().to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        style_enc.load_state_dict(checkpoint['style_enc_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Loaded models from epoch {epoch}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None
    
    gen.eval()
    style_enc.eval()
    return gen, style_enc

def perform_inference(gen, style_enc, damaged_path, mask_path, style_ref_path, output_path):
    """Perform restoration inference"""
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Load and transform images
    try:
        print(f"Loading images:")
        print(f"  Damaged: {damaged_path}")
        print(f"  Mask: {mask_path}")
        print(f"  Style reference: {style_ref_path}")
        
        damaged_img = transform(Image.open(damaged_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        mask = mask_transform(Image.open(mask_path).convert("L")).unsqueeze(0).to(DEVICE)
        style_ref = transform(Image.open(style_ref_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return False
    
    # Perform restoration
    print("🎨 Performing restoration...")
    with torch.no_grad():
        style_vector = style_enc(style_ref)
        restored_img = gen(damaged_img, mask, style_vector)
    
    # Save output
    restored_img = restored_img * 0.5 + 0.5  # De-normalize from [-1, 1] to [0, 1]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    transforms.ToPILImage()(restored_img.squeeze(0).cpu()).save(output_path)
    print(f"✅ Restoration complete! Image saved to {output_path}")
    return True

def run_batch_inference(gen, style_enc, num_samples=5):
    """Run inference on multiple test samples"""
    import random
    
    # Get available images
    damaged_dir = "./data/train_damaged"
    mask_dir = "./data/train_masks" 
    gt_dir = "./data/train_ground_truth"
    
    if not all(os.path.exists(d) for d in [damaged_dir, mask_dir, gt_dir]):
        print("❌ Data directories not found")
        return
    
    # Get list of available images
    damaged_files = [f for f in os.listdir(damaged_dir) if f.endswith('.jpg')]
    available_files = [f for f in damaged_files if 
                      os.path.exists(os.path.join(mask_dir, f)) and 
                      os.path.exists(os.path.join(gt_dir, f))]
    
    print(f"Found {len(available_files)} complete image sets")
    
    # Select random samples
    test_files = random.sample(available_files, min(num_samples, len(available_files)))
    
    # Create output directory
    output_dir = "./inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    for i, filename in enumerate(test_files):
        print(f"\n--- Sample {i+1}/{len(test_files)}: {filename} ---")
        
        damaged_path = os.path.join(damaged_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        style_ref_path = os.path.join(gt_dir, filename)
        
        # Create output filename
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"restored_{base_name}.png")
        
        if perform_inference(gen, style_enc, damaged_path, mask_path, style_ref_path, output_path):
            successful += 1
    
    print(f"\n🎉 Batch inference complete! {successful}/{len(test_files)} samples processed successfully")
    print(f"Results saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Haveli-GAN Inference')
    parser.add_argument('--damaged', default=DEFAULT_DAMAGED_IMG, help='Path to damaged image')
    parser.add_argument('--mask', default=DEFAULT_MASK, help='Path to damage mask')
    parser.add_argument('--style', default=DEFAULT_STYLE_REF, help='Path to style reference image')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output path for restored image')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH, help='Path to model checkpoint')
    parser.add_argument('--batch', action='store_true', help='Run batch inference on random samples')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples for batch inference')
    
    args = parser.parse_args()
    
    # Load models
    gen, style_enc = load_models(args.checkpoint)
    if gen is None or style_enc is None:
        return
    
    if args.batch:
        # Run batch inference
        run_batch_inference(gen, style_enc, args.num_samples)
    else:
        # Run single inference
        perform_inference(gen, style_enc, args.damaged, args.mask, args.style, args.output)

if __name__ == "__main__":
    main()
