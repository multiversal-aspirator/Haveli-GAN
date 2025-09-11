# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FrescoDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.damaged_dir = os.path.join(root_dir, 'train_damaged')
        self.gt_dir = os.path.join(root_dir, 'train_ground_truth')
        self.mask_dir = os.path.join(root_dir, 'train_masks')
        
        self.image_files = sorted(os.listdir(self.damaged_dir))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- IMPORTANT: Adapt this part to your file naming convention ---
        img_name = self.image_files[idx]
        
        damaged_path = os.path.join(self.damaged_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name) # Assuming same name
        mask_path = os.path.join(self.mask_dir, img_name) # Assuming same name
        
        damaged_img = Image.open(damaged_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Grayscale
        
        return {
            'damaged': self.transform(damaged_img),
            'gt': self.transform(gt_img),
            'mask': self.mask_transform(mask),
            'style_ref': self.transform(gt_img) # Using ground truth as style reference
        }