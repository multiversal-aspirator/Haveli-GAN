def load_models(checkpoint_path):
    """Load Generator and StyleEncoder from unified checkpoint"""
    print(f"Loading models from {checkpoint_path}")
    
    gen = Generator().to(DEVICE)
    style_enc = StyleEncoder().to(DEVICE)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        gen.load_state_dict(checkpoint['generator'])
        style_enc.load_state_dict(checkpoint['style_encoder'])
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

def main():
    parser = argparse.ArgumentParser(description='Haveli-GAN Inference')
    parser.add_argument('--damaged', default=DEFAULT_DAMAGED_IMG, help='Path to damaged image')
    parser.add_argument('--mask', default=DEFAULT_MASK, help='Path to damage mask')
    parser.add_argument('--style', default=DEFAULT_STYLE_REF, help='Path to style reference image')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output path for restored image')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Load models
    gen, style_enc = load_models(args.checkpoint)
    if gen is None or style_enc is None:
        return
    
    # Perform inference
    perform_inference(gen, style_enc, args.damaged, args.mask, args.style, args.output)

if __name__ == "__main__":
    main()