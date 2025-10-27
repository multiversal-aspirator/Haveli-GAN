"""
Partial Convolutions Implementation for Image Inpainting
Based on "Image Inpainting for Irregular Holes Using Partial Convolutions" (Liu et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PartialConv2d(nn.Module):
    """Partial Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__()
        
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, groups, False)
        
        # Initialize mask conv weights to 1
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        
        # Make mask conv weights non-trainable
        for param in self.mask_conv.parameters():
            param.requires_grad = False
    
    def forward(self, input_x, mask):
        # input_x: input image
        # mask: binary mask (1 for valid pixels, 0 for holes)
        
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        
        output_pre = (self.input_conv(input_x * mask)) / mask_sum
        output = output_pre.masked_fill_(no_update_holes, 0.0)
        
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        
        return output, new_mask

class PartialConvUNet(nn.Module):
    """U-Net architecture with Partial Convolutions"""
    
    def __init__(self, input_channels=3, output_channels=3):
        super(PartialConvUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc_1 = PartialConv2d(input_channels, 64, 7, stride=2, padding=3)
        self.enc_2 = PartialConv2d(64, 128, 5, stride=2, padding=2)
        self.enc_3 = PartialConv2d(128, 256, 5, stride=2, padding=2)
        self.enc_4 = PartialConv2d(256, 512, 3, stride=2, padding=1)
        self.enc_5 = PartialConv2d(512, 512, 3, stride=2, padding=1)
        self.enc_6 = PartialConv2d(512, 512, 3, stride=2, padding=1)
        self.enc_7 = PartialConv2d(512, 512, 3, stride=2, padding=1)
        self.enc_8 = PartialConv2d(512, 512, 3, stride=2, padding=1)
        
        # Decoder (Upsampling)
        self.dec_8 = PartialConv2d(512 + 512, 512, 3, padding=1)
        self.dec_7 = PartialConv2d(512 + 512, 512, 3, padding=1)
        self.dec_6 = PartialConv2d(512 + 512, 512, 3, padding=1)
        self.dec_5 = PartialConv2d(512 + 512, 512, 3, padding=1)
        self.dec_4 = PartialConv2d(512 + 256, 256, 3, padding=1)
        self.dec_3 = PartialConv2d(256 + 128, 128, 3, padding=1)
        self.dec_2 = PartialConv2d(128 + 64, 64, 3, padding=1)
        self.dec_1 = PartialConv2d(64 + input_channels, output_channels, 3, padding=1)
        
        # Batch normalization layers
        self.bn_enc_1 = nn.BatchNorm2d(64)
        self.bn_enc_2 = nn.BatchNorm2d(128)
        self.bn_enc_3 = nn.BatchNorm2d(256)
        self.bn_enc_4 = nn.BatchNorm2d(512)
        self.bn_enc_5 = nn.BatchNorm2d(512)
        self.bn_enc_6 = nn.BatchNorm2d(512)
        self.bn_enc_7 = nn.BatchNorm2d(512)
        
        self.bn_dec_8 = nn.BatchNorm2d(512)
        self.bn_dec_7 = nn.BatchNorm2d(512)
        self.bn_dec_6 = nn.BatchNorm2d(512)
        self.bn_dec_5 = nn.BatchNorm2d(512)
        self.bn_dec_4 = nn.BatchNorm2d(256)
        self.bn_dec_3 = nn.BatchNorm2d(128)
        self.bn_dec_2 = nn.BatchNorm2d(64)
        
        self.activation = nn.ReLU(inplace=True)
        self.activation_last = nn.Tanh()
    
    def forward(self, input_img, mask):
        # Replicate mask to match input channels for first layer
        if mask.size(1) == 1 and input_img.size(1) > 1:
            mask = mask.repeat(1, input_img.size(1), 1, 1)
        
        # Encoder
        h_enc_1, m_enc_1 = self.enc_1(input_img, mask)
        h_enc_1 = self.bn_enc_1(h_enc_1)
        h_enc_1 = self.activation(h_enc_1)
        
        h_enc_2, m_enc_2 = self.enc_2(h_enc_1, m_enc_1)
        h_enc_2 = self.bn_enc_2(h_enc_2)
        h_enc_2 = self.activation(h_enc_2)
        
        h_enc_3, m_enc_3 = self.enc_3(h_enc_2, m_enc_2)
        h_enc_3 = self.bn_enc_3(h_enc_3)
        h_enc_3 = self.activation(h_enc_3)
        
        h_enc_4, m_enc_4 = self.enc_4(h_enc_3, m_enc_3)
        h_enc_4 = self.bn_enc_4(h_enc_4)
        h_enc_4 = self.activation(h_enc_4)
        
        h_enc_5, m_enc_5 = self.enc_5(h_enc_4, m_enc_4)
        h_enc_5 = self.bn_enc_5(h_enc_5)
        h_enc_5 = self.activation(h_enc_5)
        
        h_enc_6, m_enc_6 = self.enc_6(h_enc_5, m_enc_5)
        h_enc_6 = self.bn_enc_6(h_enc_6)
        h_enc_6 = self.activation(h_enc_6)
        
        h_enc_7, m_enc_7 = self.enc_7(h_enc_6, m_enc_6)
        h_enc_7 = self.bn_enc_7(h_enc_7)
        h_enc_7 = self.activation(h_enc_7)
        
        h_enc_8, m_enc_8 = self.enc_8(h_enc_7, m_enc_7)
        h_enc_8 = self.activation(h_enc_8)
        
        # Decoder
        h_dec_8 = F.interpolate(h_enc_8, scale_factor=2, mode='nearest')
        m_dec_8 = F.interpolate(m_enc_8, scale_factor=2, mode='nearest')
        h_dec_8 = torch.cat([h_dec_8, h_enc_7], dim=1)
        m_dec_8 = torch.cat([m_dec_8, m_enc_7], dim=1)
        h_dec_8, m_dec_8 = self.dec_8(h_dec_8, m_dec_8)
        h_dec_8 = self.bn_dec_8(h_dec_8)
        h_dec_8 = self.activation(h_dec_8)
        
        h_dec_7 = F.interpolate(h_dec_8, scale_factor=2, mode='nearest')
        m_dec_7 = F.interpolate(m_dec_8, scale_factor=2, mode='nearest')
        h_dec_7 = torch.cat([h_dec_7, h_enc_6], dim=1)
        m_dec_7 = torch.cat([m_dec_7, m_enc_6], dim=1)
        h_dec_7, m_dec_7 = self.dec_7(h_dec_7, m_dec_7)
        h_dec_7 = self.bn_dec_7(h_dec_7)
        h_dec_7 = self.activation(h_dec_7)
        
        h_dec_6 = F.interpolate(h_dec_7, scale_factor=2, mode='nearest')
        m_dec_6 = F.interpolate(m_dec_7, scale_factor=2, mode='nearest')
        h_dec_6 = torch.cat([h_dec_6, h_enc_5], dim=1)
        m_dec_6 = torch.cat([m_dec_6, m_enc_5], dim=1)
        h_dec_6, m_dec_6 = self.dec_6(h_dec_6, m_dec_6)
        h_dec_6 = self.bn_dec_6(h_dec_6)
        h_dec_6 = self.activation(h_dec_6)
        
        h_dec_5 = F.interpolate(h_dec_6, scale_factor=2, mode='nearest')
        m_dec_5 = F.interpolate(m_dec_6, scale_factor=2, mode='nearest')
        h_dec_5 = torch.cat([h_dec_5, h_enc_4], dim=1)
        m_dec_5 = torch.cat([m_dec_5, m_enc_4], dim=1)
        h_dec_5, m_dec_5 = self.dec_5(h_dec_5, m_dec_5)
        h_dec_5 = self.bn_dec_5(h_dec_5)
        h_dec_5 = self.activation(h_dec_5)
        
        h_dec_4 = F.interpolate(h_dec_5, scale_factor=2, mode='nearest')
        m_dec_4 = F.interpolate(m_dec_5, scale_factor=2, mode='nearest')
        h_dec_4 = torch.cat([h_dec_4, h_enc_3], dim=1)
        m_dec_4 = torch.cat([m_dec_4, m_enc_3], dim=1)
        h_dec_4, m_dec_4 = self.dec_4(h_dec_4, m_dec_4)
        h_dec_4 = self.bn_dec_4(h_dec_4)
        h_dec_4 = self.activation(h_dec_4)
        
        h_dec_3 = F.interpolate(h_dec_4, scale_factor=2, mode='nearest')
        m_dec_3 = F.interpolate(m_dec_4, scale_factor=2, mode='nearest')
        h_dec_3 = torch.cat([h_dec_3, h_enc_2], dim=1)
        m_dec_3 = torch.cat([m_dec_3, m_enc_2], dim=1)
        h_dec_3, m_dec_3 = self.dec_3(h_dec_3, m_dec_3)
        h_dec_3 = self.bn_dec_3(h_dec_3)
        h_dec_3 = self.activation(h_dec_3)
        
        h_dec_2 = F.interpolate(h_dec_3, scale_factor=2, mode='nearest')
        m_dec_2 = F.interpolate(m_dec_3, scale_factor=2, mode='nearest')
        h_dec_2 = torch.cat([h_dec_2, h_enc_1], dim=1)
        m_dec_2 = torch.cat([m_dec_2, m_enc_1], dim=1)
        h_dec_2, m_dec_2 = self.dec_2(h_dec_2, m_dec_2)
        h_dec_2 = self.bn_dec_2(h_dec_2)
        h_dec_2 = self.activation(h_dec_2)
        
        h_dec_1 = F.interpolate(h_dec_2, scale_factor=2, mode='nearest')
        m_dec_1 = F.interpolate(m_dec_2, scale_factor=2, mode='nearest')
        h_dec_1 = torch.cat([h_dec_1, input_img], dim=1)
        m_dec_1 = torch.cat([m_dec_1, mask], dim=1)
        h_dec_1, m_dec_1 = self.dec_1(h_dec_1, m_dec_1)
        output = self.activation_last(h_dec_1)
        
        return output

class PartialConvLoss(nn.Module):
    """Loss function for Partial Convolutions"""
    
    def __init__(self, hole_weight=6.0, valid_weight=1.0, 
                 perceptual_weight=0.05, style_weight=120.0, tv_weight=0.1):
        super(PartialConvLoss, self).__init__()
        
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        self.l1_loss = nn.L1Loss()
        
        # VGG for perceptual loss (simplified version)
        self.vgg_features = self._create_vgg_features()
    
    def _create_vgg_features(self):
        """Create simplified VGG features for perceptual loss"""
        # Simplified version - in practice you'd use pre-trained VGG16
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
        )
        
        # Freeze parameters
        for param in features.parameters():
            param.requires_grad = False
            
        return features
    
    def forward(self, output, target, mask):
        # L1 Loss for hole and valid regions
        hole_loss = self.l1_loss(output * (1 - mask), target * (1 - mask))
        valid_loss = self.l1_loss(output * mask, target * mask)
        
        # Total variation loss for smoothness
        tv_loss = self._total_variation_loss(output * (1 - mask))
        
        # Perceptual loss (simplified)
        perceptual_loss = self.l1_loss(
            self.vgg_features(output), 
            self.vgg_features(target)
        )
        
        total_loss = (self.hole_weight * hole_loss + 
                     self.valid_weight * valid_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.tv_weight * tv_loss)
        
        return total_loss, {
            'hole_loss': hole_loss.item(),
            'valid_loss': valid_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'tv_loss': tv_loss.item()
        }
    
    def _total_variation_loss(self, img):
        """Calculate total variation loss for smoothness"""
        tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w

class PartialConvModel:
    """Wrapper class for training and inference with Partial Convolutions"""
    
    def __init__(self, device='cuda', lr=0.0002):
        self.device = device
        self.model = PartialConvUNet().to(device)
        self.criterion = PartialConvLoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        
    def train_step(self, damaged_img, mask, target_img):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(damaged_img, mask)
        
        # Calculate loss
        loss, loss_dict = self.criterion(output, target_img, mask)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_dict
    
    def inference(self, damaged_img, mask):
        """Inference mode"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(damaged_img, mask)
            # Combine output with original image in valid regions
            result = output * (1 - mask) + damaged_img * mask
            return result
    
    def save_checkpoint(self, path, epoch=0):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)

def test_partial_conv():
    """Test function for Partial Convolutions"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    batch_size, channels, height, width = 2, 3, 256, 256
    input_img = torch.randn(batch_size, channels, height, width).to(device)
    mask = torch.ones(batch_size, 1, height, width).to(device)
    
    # Create random holes in mask
    mask[:, :, 50:150, 50:150] = 0  # Square hole
    mask[:, :, 180:220, 180:220] = 0  # Another hole
    
    # Create model
    model = PartialConvModel(device=device)
    
    # Test inference
    print("Testing Partial Convolutions model...")
    output = model.inference(input_img, mask)
    print(f"Input shape: {input_img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Partial Convolutions model test passed!")

if __name__ == "__main__":
    test_partial_conv()