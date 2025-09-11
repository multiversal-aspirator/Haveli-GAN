#!/usr/bin/env python3
"""
Debug script to understand tensor size mismatches
"""
import torch
from model import Generator

def debug_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator().to(device)
    
    # Create test inputs
    batch_size = 1
    img_size = 256
    damaged = torch.randn(batch_size, 3, img_size, img_size).to(device)
    mask = torch.randn(batch_size, 1, img_size, img_size).to(device)
    style_vector = torch.randn(batch_size, 512).to(device)
    
    print(f"Input damaged shape: {damaged.shape}")
    print(f"Input mask shape: {mask.shape}")
    print(f"Style vector shape: {style_vector.shape}")
    
    # Forward pass with debug
    x = torch.cat([damaged, mask], dim=1)
    print(f"Concatenated input shape: {x.shape}")
    
    # Encoder
    e1 = gen.enc1(x)
    print(f"e1 shape: {e1.shape}")
    e2 = gen.enc2(e1)
    print(f"e2 shape: {e2.shape}")
    e3 = gen.enc3(e2)
    print(f"e3 shape: {e3.shape}")
    e4 = gen.enc4(e3)
    print(f"e4 shape: {e4.shape}")
    
    # Bottleneck
    b = gen.bottleneck(e4)
    print(f"Bottleneck shape: {b.shape}")
    b = gen.attention(b)
    print(f"After attention shape: {b.shape}")
    
    # Style parameters
    style_params = gen.style_mlp(style_vector).view(style_vector.size(0), 6, 512)
    print(f"Style params shape: {style_params.shape}")
    
    # Decoder
    d4_in = torch.cat([b, e4], dim=1)
    print(f"d4_in shape: {d4_in.shape}")
    d4 = gen.dec4(d4_in)
    print(f"d4 shape: {d4.shape}")
    
    d3_in = torch.cat([d4, e3], dim=1)
    print(f"d3_in shape: {d3_in.shape}")

if __name__ == "__main__":
    debug_generator()
