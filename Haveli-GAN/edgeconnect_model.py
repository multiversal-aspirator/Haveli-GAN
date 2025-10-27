"""
EdgeConnect Implementation for Image Inpainting
Based on "EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning" (Nazeri et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EdgeGenerator(nn.Module):
    """Edge Generator Network"""
    
    def __init__(self, residual_blocks=8, use_spectral_norm=True):
        super(EdgeGenerator, self).__init__()
        
        self.encoder = nn.Sequential(
            # Initial convolution
            spectral_norm(nn.Conv2d(2, 64, 7, 1, 3), use_spectral_norm),  # grayscale + mask
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
            # Downsampling
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )
        
        # Residual blocks
        blocks = []
        for _ in range(residual_blocks):
            blocks.append(ResnetBlock(256, use_spectral_norm=use_spectral_norm))
        self.middle = nn.Sequential(*blocks)
        
        self.decoder = nn.Sequential(
            # Upsampling
            spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            
            spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
            # Output layer
            nn.Conv2d(64, 1, 7, 1, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask):
        # x: grayscale image, mask: binary mask
        input_tensor = torch.cat([x, mask], dim=1)
        
        x = self.encoder(input_tensor)
        x = self.middle(x)
        x = self.decoder(x)
        
        return x

class InpaintGenerator(nn.Module):
    """Inpainting Generator Network"""
    
    def __init__(self, residual_blocks=8, use_spectral_norm=True):
        super(InpaintGenerator, self).__init__()
        
        self.encoder = nn.Sequential(
            # Initial convolution - RGB image + edge + mask
            spectral_norm(nn.Conv2d(5, 64, 7, 1, 3), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
            # Downsampling
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )
        
        # Residual blocks
        blocks = []
        for _ in range(residual_blocks):
            blocks.append(ResnetBlock(256, use_spectral_norm=use_spectral_norm))
        self.middle = nn.Sequential(*blocks)
        
        self.decoder = nn.Sequential(
            # Upsampling
            spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            
            spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            
            # Output layer
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
    
    def forward(self, x, edge, mask):
        # x: RGB image, edge: edge map, mask: binary mask
        input_tensor = torch.cat([x, edge, mask], dim=1)
        
        x = self.encoder(input_tensor)
        x = self.middle(x)
        x = self.decoder(x)
        
        return x

class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    
    def __init__(self, in_channels=3, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1), use_spectral_norm)
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1), use_spectral_norm)
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, 2, 1), use_spectral_norm)
        self.conv4 = spectral_norm(nn.Conv2d(256, 512, 4, 1, 1), use_spectral_norm)
        self.conv5 = spectral_norm(nn.Conv2d(512, 1, 4, 1, 1), use_spectral_norm)
        
        self.norm2 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.norm3 = nn.InstanceNorm2d(256, track_running_stats=False)
        self.norm4 = nn.InstanceNorm2d(512, track_running_stats=False)
        
        self.activation = nn.LeakyReLU(0.2, True)
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.activation(self.norm3(self.conv3(x)))
        x = self.activation(self.norm4(self.conv4(x)))
        x = self.conv5(x)
        
        return x

class ResnetBlock(nn.Module):
    """Residual Block"""
    
    def __init__(self, dim, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(dim, dim, 3), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(dim, dim, 3), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

def spectral_norm(module, use_spectral_norm=True):
    """Apply spectral normalization"""
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module

class EdgeConnectLoss(nn.Module):
    """Loss functions for EdgeConnect"""
    
    def __init__(self):
        super(EdgeConnectLoss, self).__init__()
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Perceptual loss using VGG (simplified)
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
        )
        
        for param in features.parameters():
            param.requires_grad = False
            
        return features
    
    def adversarial_loss(self, pred, target):
        """Adversarial loss"""
        return self.mse_loss(pred, target)
    
    def edge_loss(self, pred_edge, gt_edge):
        """Edge reconstruction loss"""
        return self.l1_loss(pred_edge, gt_edge)
    
    def reconstruction_loss(self, pred_img, gt_img, mask):
        """Image reconstruction loss"""
        hole_loss = self.l1_loss(pred_img * (1 - mask), gt_img * (1 - mask))
        valid_loss = self.l1_loss(pred_img * mask, gt_img * mask)
        return hole_loss + valid_loss
    
    def perceptual_loss(self, pred_img, gt_img):
        """Perceptual loss using VGG features"""
        pred_features = self.vgg_features(pred_img)
        gt_features = self.vgg_features(gt_img)
        return self.l1_loss(pred_features, gt_features)
    
    def style_loss(self, pred_img, gt_img):
        """Style loss using Gram matrices"""
        pred_features = self.vgg_features(pred_img)
        gt_features = self.vgg_features(gt_img)
        
        pred_gram = self._gram_matrix(pred_features)
        gt_gram = self._gram_matrix(gt_features)
        
        return self.l1_loss(pred_gram, gt_gram)
    
    def _gram_matrix(self, features):
        """Compute Gram matrix"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

class EdgeConnect:
    """EdgeConnect Model Wrapper"""
    
    def __init__(self, device='cuda', lr=0.0001, beta1=0.0, beta2=0.9):
        self.device = device
        
        # Initialize networks
        self.edge_generator = EdgeGenerator().to(device)
        self.inpaint_generator = InpaintGenerator().to(device)
        self.edge_discriminator = Discriminator(in_channels=1).to(device)
        self.inpaint_discriminator = Discriminator(in_channels=3).to(device)
        
        # Loss function
        self.criterion = EdgeConnectLoss().to(device)
        
        # Optimizers
        self.edge_gen_optimizer = torch.optim.Adam(
            self.edge_generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.inpaint_gen_optimizer = torch.optim.Adam(
            self.inpaint_generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.edge_dis_optimizer = torch.optim.Adam(
            self.edge_discriminator.parameters(), lr=lr * 4, betas=(beta1, beta2)
        )
        self.inpaint_dis_optimizer = torch.optim.Adam(
            self.inpaint_discriminator.parameters(), lr=lr * 4, betas=(beta1, beta2)
        )
    
    def rgb_to_grayscale(self, rgb_img):
        """Convert RGB to grayscale"""
        return 0.299 * rgb_img[:, 0:1, :, :] + 0.587 * rgb_img[:, 1:2, :, :] + 0.114 * rgb_img[:, 2:3, :, :]
    
    def canny_edge_detection(self, img):
        """Simplified edge detection (placeholder)"""
        # In practice, you'd use cv2.Canny or a learned edge detector
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        
        if len(img.shape) == 4 and img.shape[1] == 3:
            img_gray = self.rgb_to_grayscale(img)
        else:
            img_gray = img
        
        edge_x = F.conv2d(img_gray, sobel_x, padding=1)
        edge_y = F.conv2d(img_gray, sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge = torch.clamp(edge, 0, 1)
        
        return edge
    
    def train_step(self, damaged_img, mask, target_img):
        """Single training step"""
        batch_size = damaged_img.size(0)
        real_label = torch.ones(batch_size, 1, 30, 30).to(self.device)  # Adjust size based on discriminator output
        fake_label = torch.zeros(batch_size, 1, 30, 30).to(self.device)
        
        # Convert to grayscale for edge detection
        damaged_gray = self.rgb_to_grayscale(damaged_img)
        target_gray = self.rgb_to_grayscale(target_img)
        target_edge = self.canny_edge_detection(target_img)
        
        # Phase 1: Train Edge Generator
        self.edge_generator.train()
        self.edge_discriminator.train()
        
        # Generate edges
        pred_edge = self.edge_generator(damaged_gray, mask)
        
        # Edge discriminator loss
        self.edge_dis_optimizer.zero_grad()
        
        real_edge_pred = self.edge_discriminator(target_edge)
        fake_edge_pred = self.edge_discriminator(pred_edge.detach())
        
        real_edge_loss = self.criterion.adversarial_loss(real_edge_pred, real_label)
        fake_edge_loss = self.criterion.adversarial_loss(fake_edge_pred, fake_label)
        edge_dis_loss = (real_edge_loss + fake_edge_loss) * 0.5
        
        edge_dis_loss.backward()
        self.edge_dis_optimizer.step()
        
        # Edge generator loss
        self.edge_gen_optimizer.zero_grad()
        
        fake_edge_pred = self.edge_discriminator(pred_edge)
        edge_gen_adv_loss = self.criterion.adversarial_loss(fake_edge_pred, real_label)
        edge_rec_loss = self.criterion.edge_loss(pred_edge, target_edge)
        edge_gen_loss = edge_gen_adv_loss + edge_rec_loss * 10
        
        edge_gen_loss.backward()
        self.edge_gen_optimizer.step()
        
        # Phase 2: Train Inpainting Generator
        self.inpaint_generator.train()
        self.inpaint_discriminator.train()
        
        # Generate inpainted image
        pred_img = self.inpaint_generator(damaged_img, pred_edge.detach(), mask)
        
        # Inpainting discriminator loss
        self.inpaint_dis_optimizer.zero_grad()
        
        real_img_pred = self.inpaint_discriminator(target_img)
        fake_img_pred = self.inpaint_discriminator(pred_img.detach())
        
        real_img_loss = self.criterion.adversarial_loss(real_img_pred, real_label)
        fake_img_loss = self.criterion.adversarial_loss(fake_img_pred, fake_label)
        inpaint_dis_loss = (real_img_loss + fake_img_loss) * 0.5
        
        inpaint_dis_loss.backward()
        self.inpaint_dis_optimizer.step()
        
        # Inpainting generator loss
        self.inpaint_gen_optimizer.zero_grad()
        
        fake_img_pred = self.inpaint_discriminator(pred_img)
        inpaint_gen_adv_loss = self.criterion.adversarial_loss(fake_img_pred, real_label)
        inpaint_rec_loss = self.criterion.reconstruction_loss(pred_img, target_img, mask)
        inpaint_perceptual_loss = self.criterion.perceptual_loss(pred_img, target_img)
        inpaint_style_loss = self.criterion.style_loss(pred_img, target_img)
        
        inpaint_gen_loss = (inpaint_gen_adv_loss + 
                           inpaint_rec_loss * 10 + 
                           inpaint_perceptual_loss * 0.1 + 
                           inpaint_style_loss * 250)
        
        inpaint_gen_loss.backward()
        self.inpaint_gen_optimizer.step()
        
        return {
            'edge_dis_loss': edge_dis_loss.item(),
            'edge_gen_loss': edge_gen_loss.item(),
            'inpaint_dis_loss': inpaint_dis_loss.item(),
            'inpaint_gen_loss': inpaint_gen_loss.item()
        }
    
    def inference(self, damaged_img, mask):
        """Inference mode"""
        self.edge_generator.eval()
        self.inpaint_generator.eval()
        
        with torch.no_grad():
            # Generate edges
            damaged_gray = self.rgb_to_grayscale(damaged_img)
            pred_edge = self.edge_generator(damaged_gray, mask)
            
            # Generate inpainted image
            pred_img = self.inpaint_generator(damaged_img, pred_edge, mask)
            
            # Combine with original image in valid regions
            result = pred_img * (1 - mask) + damaged_img * mask
            
        return result
    
    def save_checkpoint(self, path, epoch=0):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'edge_generator': self.edge_generator.state_dict(),
            'inpaint_generator': self.inpaint_generator.state_dict(),
            'edge_discriminator': self.edge_discriminator.state_dict(),
            'inpaint_discriminator': self.inpaint_discriminator.state_dict(),
            'edge_gen_optimizer': self.edge_gen_optimizer.state_dict(),
            'inpaint_gen_optimizer': self.inpaint_gen_optimizer.state_dict(),
            'edge_dis_optimizer': self.edge_dis_optimizer.state_dict(),
            'inpaint_dis_optimizer': self.inpaint_dis_optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.edge_generator.load_state_dict(checkpoint['edge_generator'])
        self.inpaint_generator.load_state_dict(checkpoint['inpaint_generator'])
        self.edge_discriminator.load_state_dict(checkpoint['edge_discriminator'])
        self.inpaint_discriminator.load_state_dict(checkpoint['inpaint_discriminator'])
        self.edge_gen_optimizer.load_state_dict(checkpoint['edge_gen_optimizer'])
        self.inpaint_gen_optimizer.load_state_dict(checkpoint['inpaint_gen_optimizer'])
        self.edge_dis_optimizer.load_state_dict(checkpoint['edge_dis_optimizer'])
        self.inpaint_dis_optimizer.load_state_dict(checkpoint['inpaint_dis_optimizer'])
        return checkpoint.get('epoch', 0)

def test_edgeconnect():
    """Test function for EdgeConnect"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    batch_size, channels, height, width = 2, 3, 256, 256
    damaged_img = torch.randn(batch_size, channels, height, width).to(device)
    mask = torch.ones(batch_size, 1, height, width).to(device)
    
    # Create random holes in mask
    mask[:, :, 50:150, 50:150] = 0
    mask[:, :, 180:220, 180:220] = 0
    
    # Create model
    model = EdgeConnect(device=device)
    
    # Test inference
    print("Testing EdgeConnect model...")
    output = model.inference(damaged_img, mask)
    print(f"Input shape: {damaged_img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ EdgeConnect model test passed!")

if __name__ == "__main__":
    test_edgeconnect()