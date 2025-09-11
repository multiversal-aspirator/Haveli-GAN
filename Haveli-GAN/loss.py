# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
        
        # Select layers for perceptual and style loss
        self.perceptual_layers = ['21'] # relu4_2
        self.style_layers = ['0', '5', '10', '19', '28'] # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        
        self.slices = nn.ModuleList()
        last_layer_idx = 0
        for i in range(29): # Up to relu5_1
            self.slices.add_module(str(i), vgg[i])
        
        for param in self.parameters():
            param.requires_grad = False
            
    def gram_matrix(self, y):
        (b, c, h, w) = y.size()
        # Clone the tensor to avoid in-place operation issues
        y_clone = y.clone()
        features = y_clone.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, generated, ground_truth):
        loss_p = 0.0
        loss_s = 0.0
        
        gen_features = []
        gt_features = []
        
        # Clone inputs to avoid in-place operation issues
        x_gen, x_gt = generated.clone(), ground_truth.clone()
        for name, layer in self.slices.named_children():
            x_gen = layer(x_gen)
            x_gt = layer(x_gt)
            
            if name in self.perceptual_layers:
                loss_p += F.l1_loss(x_gen, x_gt)
                
            if name in self.style_layers:
                loss_s += F.l1_loss(self.gram_matrix(x_gen), self.gram_matrix(x_gt))
                
        return loss_p, loss_s