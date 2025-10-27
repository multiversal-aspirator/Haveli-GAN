# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    """
    Gated Convolutional Layer for handling irregular masks.
    This layer learns to focus on valid pixels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.conv2d(x)
        mask = self.sigmoid(self.mask_conv2d(x))
        return feature * mask

class SelfAttention(nn.Module):
    """ Self-attention layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        # Avoid in-place operation
        result = self.gamma * out + x
        return result

def calc_mean_std(feat, eps=1e-5):
    """ Calculate mean and standard deviation for AdaIN """
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def AdaIN(content_feat, style_feat):
    """ Adaptive Instance Normalization """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / (content_std.expand(size) + 1e-5)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        features = self.net(x).view(x.size(0), -1)
        return self.fc(features)

class Generator(nn.Module):
    """ The main Generator network (Style-Semantic U-Net) """
    def __init__(self, input_channels=4): # 3 for image + 1 for mask
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(GatedConv2d(input_channels, 64, 3, 1, 1), nn.LeakyReLU(0.2))
        self.enc2 = nn.Sequential(GatedConv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(GatedConv2d(128, 256, 3, 2, 1), nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(GatedConv2d(256, 512, 3, 2, 1), nn.LeakyReLU(0.2))
        
        # Bottleneck with Self-Attention (no downsampling)
        self.bottleneck = nn.Sequential(GatedConv2d(512, 512, 3, 1, 1), nn.LeakyReLU(0.2))
        self.attention = SelfAttention(512)
        
        # Decoder with AdaIN style modulation
        self.style_mlp = nn.Linear(512, 512*6) # For 6 AdaIN layers
        
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(768, 256, 4, 2, 1), nn.ReLU())  # 512 + 256 = 768
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(384, 128, 4, 2, 1), nn.ReLU())  # 256 + 128 = 384
        self.dec1 = nn.Sequential(nn.Conv2d(192, 64, 3, 1, 1), nn.ReLU())           # 128 + 64 = 192, no upsampling
        self.out_conv = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, img, mask, style_vector):
        x = torch.cat([img, mask], dim=1)
        
        # Encoder - store outputs and clone them to avoid in-place issues
        e1 = self.enc1(x).clone()
        e2 = self.enc2(e1).clone()
        e3 = self.enc3(e2).clone()
        e4 = self.enc4(e3).clone()
        
        # Bottleneck
        b = self.bottleneck(e4)
        b = self.attention(b)
        
        # Get AdaIN parameters from style vector
        style_params = self.style_mlp(style_vector).view(style_vector.size(0), 6, 512)
        
        # Decoder - clone tensors before concatenation to prevent in-place operations
        d4_in = torch.cat([b.clone(), e4.clone()], dim=1)
        d4 = self.dec4(d4_in)
        d4 = self._apply_adain(d4, style_params[:, 0:2, :])
        
        d3_in = torch.cat([d4.clone(), e3.clone()], dim=1)
        d3 = self.dec3(d3_in)
        d3 = self._apply_adain(d3, style_params[:, 2:4, :])

        d2_in = torch.cat([d3.clone(), e2.clone()], dim=1)
        d2 = self.dec2(d2_in)
        d2 = self._apply_adain(d2, style_params[:, 4:6, :])
        
        d1_in = torch.cat([d2.clone(), e1.clone()], dim=1)
        d1 = self.dec1(d1_in)
        
        out = torch.tanh(self.out_conv(d1))
        return out

    def _apply_adain(self, features, params):
        gamma = params[:, 0, :features.size(1)].unsqueeze(-1).unsqueeze(-1)
        beta = params[:, 1, :features.size(1)].unsqueeze(-1).unsqueeze(-1)
        norm = F.instance_norm(features.clone())  # Ensure we're working with a copy
        return (1 + gamma) * norm + beta

class Discriminator(nn.Module):
    """ Conditional PatchGAN Discriminator """
    def __init__(self, input_channels=6): # 3 for image + 3 for conditional damaged img
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, norm=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, norm=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img, conditional_img):
        img_input = torch.cat((img, conditional_img), 1)
        return self.model(img_input)