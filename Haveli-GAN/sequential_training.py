#!/usr/bin/env python3
"""
Sequential training of all models to avoid memory issues
"""

import torch
import torch.nn as nn
import os
import time
from torch.utils.data import DataLoader
from dataset import FrescoDataset
from tqdm import tqdm

# Import all models
from model import Generator, StyleEncoder
from loss import VGGPerceptualLoss
from partial_conv_model import PartialConvModel
from edgeconnect_model import EdgeConnect
from mat_model import MAT

class HaveliGANWrapper:
    """Wrapper for Haveli-GAN model to match interface"""
    def __init__(self, device, lr):
        self.device = device
        self.generator = Generator().to(device)
        self.style_encoder = StyleEncoder().to(device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        
        # Optimizer
        params = list(self.generator.parameters()) + list(self.style_encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        
    def train_step(self, damaged, mask, gt):
        """Single training step"""
        self.generator.train()
        self.style_encoder.train()
        
        self.optimizer.zero_grad()
        
        # Extract style from ground truth
        style_features = self.style_encoder(gt)
        
        # Generate image
        generated = self.generator(damaged, mask, style_features)
        
        # Calculate losses
        mse_loss = self.mse_loss(generated, gt)
        perceptual_loss = self.perceptual_loss(generated, gt)
        
        total_loss = mse_loss + 0.1 * perceptual_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        loss_dict = {
            'mse_loss': mse_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'generator': self.generator.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

class SequentialTrainer:
    """Train models one at a time to avoid memory issues"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize dataset
        self.dataset = FrescoDataset(root_dir=config['data_dir'])
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=2
        )
        
        # Results storage
        self.results = {}
        
    def train_single_model(self, model_name):
        """Train a single model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training {model_name}")
        print(f"{'='*60}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Initialize model
        if model_name == 'HaveliGAN':
            model = HaveliGANWrapper(device=self.device, lr=self.config['lr'])
        elif model_name == 'PartialConv':
            model = PartialConvModel(device=self.device, lr=self.config['lr'])
        elif model_name == 'EdgeConnect':
            model = EdgeConnect(device=self.device, lr=self.config['lr'])
        elif model_name == 'MAT':
            model = MAT(device=self.device, lr=self.config['lr'])
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Training statistics
        epoch_losses = []
        epoch_times = []
        
        print(f"📁 Dataset: {len(self.train_loader)} batches")
        print(f"🏃 Training for {self.config['epochs']} epochs")
        
        # Time estimation based on 5-epoch results
        time_estimates = {
            'PartialConv': 1.39 / 5,  # minutes per epoch
            'EdgeConnect': 7.15 / 5,
            'MAT': 3.21 / 5
        }
        
        if model_name in time_estimates:
            estimated_total_time = time_estimates[model_name] * self.config['epochs']
            print(f"⏱️ Estimated training time: {estimated_total_time:.1f} minutes ({estimated_total_time/60:.1f} hours)")
        
        training_start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\n🔄 Epoch {epoch+1}/{self.config['epochs']}")
            print("-" * 40)
            
            epoch_start = time.time()
            losses = []
            
            # Progress bar
            pbar = tqdm(self.train_loader, desc=f"{model_name} Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                # Get data
                damaged = batch['damaged'].to(self.device)
                mask = batch['mask'].to(self.device)
                gt = batch['gt'].to(self.device)
                
                try:
                    # Train step
                    if model_name == 'EdgeConnect':
                        loss_dict = model.train_step(damaged, mask, gt)
                        loss = loss_dict['edge_gen_loss'] + loss_dict['inpaint_gen_loss']
                    elif model_name == 'MAT':
                        loss_dict = model.train_step(damaged, mask, gt)
                        loss = loss_dict['gen_loss']
                    else:
                        loss, loss_dict = model.train_step(damaged, mask, gt)
                    
                    losses.append(loss.item() if hasattr(loss, 'item') else loss)
                    
                    # Update progress bar
                    pbar.set_postfix({'Loss': f'{losses[-1]:.4f}'})
                    
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    continue
            
            epoch_time = time.time() - epoch_start
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            
            # Calculate ETA
            elapsed_total = time.time() - training_start_time
            if epoch > 0:
                avg_epoch_time = elapsed_total / (epoch + 1)
                remaining_epochs = self.config['epochs'] - (epoch + 1)
                eta_minutes = (avg_epoch_time * remaining_epochs) / 60
                print(f"✅ Epoch {epoch+1} complete - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, ETA: {eta_minutes:.1f} min")
            else:
                print(f"✅ Epoch {epoch+1} complete - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Save intermediate checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                checkpoint_dir = self.config['checkpoint_dir']
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                try:
                    intermediate_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch+1}.pth')
                    model.save_checkpoint(intermediate_path)
                    print(f"💾 Intermediate checkpoint saved: {intermediate_path}")
                except Exception as e:
                    print(f"⚠️ Could not save intermediate checkpoint: {e}")
        
        # Save results
        self.results[model_name] = {
            'losses': epoch_losses,
            'times': epoch_times,
            'final_loss': epoch_losses[-1] if epoch_losses else 0
        }
        
        # Save model checkpoint
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_final.pth')
            model.save_checkpoint(checkpoint_path)
            print(f"💾 Final checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Could not save final checkpoint: {e}")
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
        
        total_training_time = time.time() - training_start_time
        print(f"✅ {model_name} training complete! Total time: {total_training_time/60:.1f} minutes")
    
    def train_all(self):
        """Train all models sequentially"""
        print("🎯 Starting sequential training of remaining models for 200 epochs")
        print("PartialConv already completed - continuing with EdgeConnect and MAT")
        print("This approach trains one model at a time to avoid memory issues")
        
        # Calculate total estimated time
        time_estimates = {
            'PartialConv': (1.39 / 5) * 200,  # minutes
            'EdgeConnect': (7.15 / 5) * 200,
            'MAT': (3.21 / 5) * 200
        }
        
        total_estimated_time = sum(time_estimates[model] for model in self.config['models_to_train'])
        print(f"\n⏱️ ESTIMATED TOTAL TRAINING TIME: {total_estimated_time:.1f} minutes ({total_estimated_time/60:.1f} hours)")
        print("Individual estimates:")
        for model in self.config['models_to_train']:
            est_time = time_estimates[model]
            print(f"  {model:12}: {est_time:.1f} minutes ({est_time/60:.1f} hours)")
        
        start_time = time.time()
        
        for model_name in self.config['models_to_train']:
            self.train_single_model(model_name)
        
        total_time = time.time() - start_time
        
        # Print final results
        print(f"\n{'='*60}")
        print("🏆 TRAINING COMPLETE - FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"Models trained: {len(self.results)}")
        
        print(f"\n📊 Final Loss Comparison:")
        for model_name, results in self.results.items():
            print(f"  {model_name:15}: {results['final_loss']:.4f}")
        
        print(f"\n⏱️ Training Time Comparison:")
        for model_name, results in self.results.items():
            total_model_time = sum(results['times'])
            print(f"  {model_name:15}: {total_model_time/60:.2f} minutes ({total_model_time/3600:.2f} hours)")
        
        return self.results

def main():
    config = {
        'models_to_train': ['PartialConv', 'EdgeConnect', 'MAT'],  # All models except HaveliGAN
        'epochs': 1,  # Single epoch training for testing
        'batch_size': 4,
        'lr': 0.0002,
        'data_dir': 'data',
        'checkpoint_dir': 'sequential_checkpoints_1epoch',
    }
    
    trainer = SequentialTrainer(config)
    results = trainer.train_all()
    
    return results

if __name__ == "__main__":
    results = main()