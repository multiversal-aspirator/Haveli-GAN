"""
Mask-Aware Transformer (MAT) Implementation for Image Inpainting
Based on "Mask-Aware Transformer for Large Hole Image Inpainting" (Li et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention with mask awareness"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Add head and query dimensions
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return self.norm(output + query)

class FeedForward(nn.Module):
    """Feed forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm(x + residual)

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, mask=None):
        x = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x

class MaskAwareTransformer(nn.Module):
    """Mask-Aware Transformer for image inpainting"""
    
    def __init__(self, img_size=256, patch_size=8, in_channels=3, d_model=512, 
                 n_heads=8, n_layers=8, d_ff=2048, dropout=0.1):
        super(MaskAwareTransformer, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, d_model, patch_size, patch_size)
        self.mask_embed = nn.Conv2d(1, d_model, patch_size, patch_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, self.n_patches)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, patch_size * patch_size * in_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def patchify(self, x):
        """Convert image to patches"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, (H // self.patch_size) * (W // self.patch_size), C * self.patch_size * self.patch_size)
        return x
    
    def unpatchify(self, x, channels=3):
        """Convert patches back to image"""
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, channels, H * self.patch_size, W * self.patch_size)
        return x
    
    def forward(self, x, mask):
        B, C, H, W = x.shape
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # B, d_model, H/patch_size, W/patch_size
        mask_patches = self.mask_embed(mask)  # B, d_model, H/patch_size, W/patch_size
        
        # Flatten patches
        x_patches = x_patches.flatten(2).transpose(1, 2)  # B, N, d_model
        mask_patches = mask_patches.flatten(2).transpose(1, 2)  # B, N, d_model
        
        # Combine image and mask information
        combined_patches = x_patches + mask_patches
        
        # Add positional encoding
        combined_patches = self.pos_encoding(combined_patches.transpose(0, 1)).transpose(0, 1)
        combined_patches = self.dropout(combined_patches)
        
        # Create attention mask (0 for holes, 1 for valid regions)
        patch_mask = F.avg_pool2d(mask, self.patch_size, self.patch_size)
        patch_mask = patch_mask.flatten(2).transpose(1, 2)  # B, N, 1
        patch_mask = (patch_mask > 0.5).float().squeeze(-1)  # B, N
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            combined_patches = layer(combined_patches, patch_mask)
        
        # Output projection
        output = self.output_proj(combined_patches)  # B, N, patch_size^2 * C
        output = self.unpatchify(output, channels=C)
        
        return output

class MATGenerator(nn.Module):
    """Complete MAT Generator with CNN encoder/decoder and transformer"""
    
    def __init__(self, img_size=256, patch_size=8, base_channels=64):
        super(MATGenerator, self).__init__()
        
        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, base_channels, 7, 1, 3),  # Input + mask
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.ReLU(True),
        )
        
        # Transformer for the bottleneck
        self.transformer = MaskAwareTransformer(
            img_size=img_size // 8,  # After 3 downsampling layers
            patch_size=4,
            in_channels=base_channels * 8,
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048
        )
        
        # CNN Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(True),
            
            nn.Conv2d(base_channels, 3, 7, 1, 3),
            nn.Tanh()
        )
    
    def forward(self, x, mask):
        # Prepare input
        input_tensor = torch.cat([x, mask], dim=1)
        
        # Encode
        encoded = self.encoder(input_tensor)
        
        # Apply transformer
        # Downsample mask for transformer
        mask_downsampled = F.avg_pool2d(mask, 8, 8)
        transformed = self.transformer(encoded, mask_downsampled)
        
        # Decode
        output = self.decoder(transformed)
        
        return output

class MATDiscriminator(nn.Module):
    """Discriminator for MAT"""
    
    def __init__(self, in_channels=3, base_channels=64):
        super(MATDiscriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # No normalization in first layer
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class MATLoss(nn.Module):
    """Loss functions for MAT"""
    
    def __init__(self, hole_weight=6.0, valid_weight=1.0, 
                 perceptual_weight=0.05, style_weight=120.0, adversarial_weight=0.1):
        super(MATLoss, self).__init__()
        
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.adversarial_weight = adversarial_weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # VGG for perceptual loss (simplified)
        self.vgg_features = self._create_vgg_features()
    
    def _create_vgg_features(self):
        """Create simplified VGG features"""
        features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        for param in features.parameters():
            param.requires_grad = False
            
        return features
    
    def adversarial_loss(self, pred, target):
        """Adversarial loss"""
        return self.mse_loss(pred, target)
    
    def reconstruction_loss(self, pred, target, mask):
        """Reconstruction loss with hole and valid region weighting"""
        hole_loss = self.l1_loss(pred * (1 - mask), target * (1 - mask))
        valid_loss = self.l1_loss(pred * mask, target * mask)
        return self.hole_weight * hole_loss + self.valid_weight * valid_loss
    
    def perceptual_loss(self, pred, target):
        """Perceptual loss using VGG features"""
        pred_features = self.vgg_features(pred)
        target_features = self.vgg_features(target)
        return self.l1_loss(pred_features, target_features)
    
    def style_loss(self, pred, target):
        """Style loss using Gram matrices"""
        pred_features = self.vgg_features(pred)
        target_features = self.vgg_features(target)
        
        pred_gram = self._gram_matrix(pred_features)
        target_gram = self._gram_matrix(target_features)
        
        return self.l1_loss(pred_gram, target_gram)
    
    def _gram_matrix(self, features):
        """Compute Gram matrix"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

class MAT:
    """Mask-Aware Transformer Model Wrapper"""
    
    def __init__(self, device='cuda', lr=0.0001, beta1=0.0, beta2=0.9):
        self.device = device
        
        # Initialize networks
        self.generator = MATGenerator().to(device)
        self.discriminator = MATDiscriminator().to(device)
        
        # Loss function
        self.criterion = MATLoss().to(device)
        
        # Optimizers
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.dis_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr * 4, betas=(beta1, beta2)
        )
    
    def train_step(self, damaged_img, mask, target_img):
        """Single training step"""
        batch_size = damaged_img.size(0)
        
        # Train Discriminator
        self.discriminator.train()
        self.dis_optimizer.zero_grad()
        
        # Generate fake images
        with torch.no_grad():
            fake_img = self.generator(damaged_img, mask)
        
        # Discriminator loss
        real_pred = self.discriminator(target_img)
        fake_pred = self.discriminator(fake_img.detach())
        
        # Create labels dynamically based on discriminator output size
        real_label = torch.ones_like(real_pred).to(self.device)
        fake_label = torch.zeros_like(fake_pred).to(self.device)
        
        real_loss = self.criterion.adversarial_loss(real_pred, real_label)
        fake_loss = self.criterion.adversarial_loss(fake_pred, fake_label)
        dis_loss = (real_loss + fake_loss) * 0.5
        
        dis_loss.backward()
        self.dis_optimizer.step()
        
        # Train Generator
        self.generator.train()
        self.gen_optimizer.zero_grad()
        
        # Generate images
        pred_img = self.generator(damaged_img, mask)
        
        # Generator losses
        fake_pred = self.discriminator(pred_img)
        # Create label for generator training based on current discriminator output
        gen_real_label = torch.ones_like(fake_pred).to(self.device)
        adv_loss = self.criterion.adversarial_loss(fake_pred, gen_real_label)
        rec_loss = self.criterion.reconstruction_loss(pred_img, target_img, mask)
        perceptual_loss = self.criterion.perceptual_loss(pred_img, target_img)
        style_loss = self.criterion.style_loss(pred_img, target_img)
        
        gen_loss = (self.criterion.adversarial_weight * adv_loss +
                   rec_loss +
                   self.criterion.perceptual_weight * perceptual_loss +
                   self.criterion.style_weight * style_loss)
        
        gen_loss.backward()
        self.gen_optimizer.step()
        
        return {
            'dis_loss': dis_loss.item(),
            'gen_loss': gen_loss.item(),
            'adv_loss': adv_loss.item(),
            'rec_loss': rec_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'style_loss': style_loss.item()
        }
    
    def inference(self, damaged_img, mask):
        """Inference mode"""
        self.generator.eval()
        
        with torch.no_grad():
            pred_img = self.generator(damaged_img, mask)
            # Combine with original image in valid regions
            result = pred_img * (1 - mask) + damaged_img * mask
            
        return result
    
    def save_checkpoint(self, path, epoch=0):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'dis_optimizer': self.dis_optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        return checkpoint.get('epoch', 0)

def test_mat():
    """Test function for MAT"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    batch_size, channels, height, width = 2, 3, 256, 256
    damaged_img = torch.randn(batch_size, channels, height, width).to(device)
    mask = torch.ones(batch_size, 1, height, width).to(device)
    
    # Create random holes in mask
    mask[:, :, 50:150, 50:150] = 0
    mask[:, :, 180:220, 180:220] = 0
    
    # Create model
    model = MAT(device=device)
    
    # Test inference
    print("Testing MAT model...")
    output = model.inference(damaged_img, mask)
    print(f"Input shape: {damaged_img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ MAT model test passed!")

if __name__ == "__main__":
    test_mat()