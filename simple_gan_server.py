#!/usr/bin/env python3
"""
Server Version: Simple GAN for Synthetic BUSI Image Generation
=============================================================
Optimized for full BUSI dataset (485 samples: 330 benign + 155 malignant)
This version uses larger architecture and proper training parameters.

Usage on server:
    python simple_gan_server.py --mode train --data_dir dataset/BioMedicalDataset/BUSI
    python simple_gan_server.py --mode generate --checkpoint simple_gan_epoch_100.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import argparse

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class ServerGenerator(nn.Module):
    """Server Generator for full dataset - larger architecture"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=256):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, 100)  # Larger embedding
        
        # Initial dense layer
        self.initial_size = img_size // 16  # 16x16 for 256x256 images
        self.initial_dim = 512  # Full size for server
        
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 100, self.initial_dim * self.initial_size * self.initial_size),
            nn.BatchNorm1d(self.initial_dim * self.initial_size * self.initial_size),
            nn.ReLU(True)
        )
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(self.initial_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        
        # Output heads for image and mask
        self.image_head = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )
        
        # IMPROVED: Initialize mask head to produce non-zero outputs
        # Initialize the final conv layer bias to encourage non-zero masks
        nn.init.constant_(self.mask_head[0].bias, 0.5)  # Bias towards 0.5 probability
        nn.init.normal_(self.mask_head[0].weight, 0.0, 0.02)
    
    def forward(self, noise, class_labels):
        # Embed class labels
        class_emb = self.class_emb(class_labels)
        
        # Concatenate noise and class embedding
        x = torch.cat([noise, class_emb], dim=1)
        
        # Pass through fully connected layer
        x = self.fc(x)
        x = x.view(-1, self.initial_dim, self.initial_size, self.initial_size)
        
        # Pass through convolutional blocks
        x = self.conv_blocks(x)
        
        # Generate image and mask
        image = self.image_head(x)
        mask = self.mask_head(x)
        
        return image, mask

class ServerDiscriminator(nn.Module):
    """Server Discriminator for full dataset - larger architecture"""
    
    def __init__(self, num_classes=2, img_size=256):
        super().__init__()
        self.num_classes = num_classes
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, img_size * img_size)
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(2, 64, 4, 2, 1, bias=False),  # 2 channels: image + class_emb
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, class_labels):
        batch_size = image.size(0)
        
        # Embed class labels and reshape to image dimensions
        class_emb = self.class_emb(class_labels)
        class_emb = class_emb.view(batch_size, 1, image.size(2), image.size(3))
        
        # Concatenate image and class embedding
        x = torch.cat([image, class_emb], dim=1)
        
        # Pass through convolutional blocks
        x = self.conv_blocks(x)
        
        # Flatten and classify
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x

class ServerBUSIDataset(Dataset):
    """Server BUSI dataset loader for full dataset"""
    
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
        
        print(f"Loading server dataset from: {data_dir}")
        
        # Check if the directory exists
        if not os.path.exists(data_dir):
            print(f"❌ Dataset directory does not exist: {data_dir}")
            print(f"   Please check the path and ensure the dataset is available")
            return
        
        # Load samples
        for class_name in ['benign', 'malignant']:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️  Class directory not found: {class_dir}")
                continue
                
            image_dir = os.path.join(class_dir, 'image')
            mask_dir = os.path.join(class_dir, 'mask')
            
            if not os.path.exists(image_dir):
                print(f"⚠️  Image directory not found: {image_dir}")
                continue
                
            if not os.path.exists(mask_dir):
                print(f"⚠️  Mask directory not found: {mask_dir}")
                continue
            
            class_count = 0
            for img_file in os.listdir(image_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_file)
                    mask_path = os.path.join(mask_dir, img_file.replace('.png', '_mask.png'))
                    
                    if os.path.exists(mask_path):
                        self.samples.append({
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'class': class_name,
                            'label': 0 if class_name == 'benign' else 1
                        })
                        class_count += 1
            
            print(f"  Loaded {class_count} {class_name} samples")
        
        print(f"Total samples loaded: {len(self.samples)}")
        
        # Only calculate distribution if we have samples
        if len(self.samples) > 0:
            benign_count = sum(1 for s in self.samples if s['label'] == 0)
            malignant_count = sum(1 for s in self.samples if s['label'] == 1)
            print(f"  Benign: {benign_count} ({benign_count/len(self.samples)*100:.1f}%)")
            print(f"  Malignant: {malignant_count} ({malignant_count/len(self.samples)*100:.1f}%)")
        else:
            print("❌ No valid samples found!")
            print("   Expected directory structure:")
            print("   dataset/BioMedicalDataset/BUSI/")
            print("   ├── benign/")
            print("   │   ├── image/")
            print("   │   └── mask/")
            print("   └── malignant/")
            print("       ├── image/")
            print("       └── mask/")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('L')
        if self.transform:
            image = self.transform(image)
        
        # Load mask - Handle BUSI white masks properly
        mask = Image.open(sample['mask_path']).convert('L')  # Convert to grayscale
        
        # Convert to binary mask (BUSI masks are white on black)
        mask_array = np.array(mask)
        binary_mask = mask_array > 128  # Threshold white pixels
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0)
        
        if self.mask_transform:
            # Apply resize if needed
            from torchvision.transforms.functional import resize
            mask_tensor = resize(mask_tensor, (256, 256))
        
        return image, mask_tensor, sample['label']

class ServerSimpleGAN:
    """Server Simple GAN for full dataset training"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=256, device='cuda'):
        self.device = device
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # Initialize networks
        self.generator = ServerGenerator(noise_dim, num_classes, img_size).to(device)
        self.discriminator = ServerDiscriminator(num_classes, img_size).to(device)
        
        # Loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        
        # Optimizers with different learning rates
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))  # Slower D
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=50, gamma=0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.optimizer_D, step_size=50, gamma=0.5)
        
        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
        
        print(f"Initialized Server GAN on {device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _weights_init(self, m):
        """Initialize network weights"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def train(self, dataloader, num_epochs=100, save_interval=10):
        """Train the GAN on server"""
        print(f"Training Server GAN for {num_epochs} epochs...")
        
        # Training history
        g_losses = []
        d_losses = []
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, (real_images, real_masks, class_labels) in enumerate(pbar):
                batch_size = real_images.size(0)
                
                real_images = real_images.to(self.device)
                real_masks = real_masks.to(self.device)
                class_labels = class_labels.to(self.device)
                
                # Labels for real and fake with label smoothing
                real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9  # Label smoothing
                fake_labels = torch.zeros(batch_size, 1, device=self.device) + 0.1  # Label smoothing
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                
                # Real images
                real_pred = self.discriminator(real_images, class_labels)
                d_real_loss = self.criterion_gan(real_pred, real_labels)
                
                # Fake images
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_images, fake_masks = self.generator(noise, class_labels)
                fake_pred = self.discriminator(fake_images.detach(), class_labels)
                d_fake_loss = self.criterion_gan(fake_pred, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()
                
                # -----------------
                # Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                
                # Generate fake images
                fake_pred = self.discriminator(fake_images, class_labels)
                g_gan_loss = self.criterion_gan(fake_pred, torch.ones_like(fake_pred))
                
                # IMPROVED mask loss with better weighting and BCE loss for masks
                mask_l1_loss = self.criterion_l1(fake_masks, real_masks) * 50  # Increased weight
                
                # Add BCE loss for better mask learning (treat as binary classification)
                mask_bce_loss = nn.BCELoss()(fake_masks, real_masks) * 20
                
                # Add regularization to encourage non-zero masks
                mask_mean = torch.mean(fake_masks)
                mask_reg_loss = torch.abs(mask_mean - 0.3) * 5  # Encourage masks to have ~30% white pixels
                
                # Total generator loss with improved mask supervision
                g_loss = g_gan_loss + mask_l1_loss + mask_bce_loss + mask_reg_loss
                g_loss.backward()
                self.optimizer_G.step()
                
                # Update progress
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'D_Loss': f'{d_loss.item():.4f}',
                    'G_Loss': f'{g_loss.item():.4f}',
                    'Mask_Mean': f'{mask_mean.item():.3f}',
                    'LR_G': f'{self.optimizer_G.param_groups[0]["lr"]:.6f}'
                })
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Calculate average losses
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs} - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}')
            
            # Save model checkpoints
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'simple_gan_epoch_{epoch+1}.pth', epoch, avg_g_loss, avg_d_loss)
                self.generate_test_samples(epoch + 1)
        
        # Save final model
        self.save_checkpoint('simple_gan_final.pth', num_epochs-1, avg_g_loss, avg_d_loss)
        
        # Plot training curves
        self.plot_training_curves(g_losses, d_losses)
    
    def save_checkpoint(self, filename, epoch, g_loss, d_loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
        }, filename)
        print(f'Saved checkpoint: {filename}')
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        return checkpoint['epoch'], checkpoint['g_loss'], checkpoint['d_loss']
    
    def generate_test_samples(self, epoch):
        """Generate test samples during training"""
        self.generator.eval()
        
        os.makedirs('server_test_output', exist_ok=True)
        
        mask_stats = {'benign': [], 'malignant': []}
        
        with torch.no_grad():
            # Generate 3 benign and 3 malignant samples
            for class_label in [0, 1]:
                class_name = 'benign' if class_label == 0 else 'malignant'
                
                for i in range(3):
                    # Generate noise
                    noise = torch.randn(1, self.noise_dim, device=self.device)
                    class_tensor = torch.tensor([class_label], device=self.device)
                    
                    # Generate image and mask
                    fake_image, fake_mask = self.generator(noise, class_tensor)
                    
                    # Convert to PIL images
                    fake_image = fake_image.squeeze().cpu()
                    fake_mask = fake_mask.squeeze().cpu()
                    
                    # DEBUG: Print mask statistics
                    mask_mean = torch.mean(fake_mask).item()
                    mask_max = torch.max(fake_mask).item()
                    mask_min = torch.min(fake_mask).item()
                    
                    mask_stats[class_name].append({
                        'mean': mask_mean,
                        'max': mask_max,
                        'min': mask_min,
                        'std': torch.std(fake_mask).item()
                    })
                    
                    # Denormalize image from [-1, 1] to [0, 255]
                    img_array = ((fake_image + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                    
                    # IMPROVED: Use adaptive threshold based on mask statistics (SAME AS LOCAL VERSION)
                    if mask_max > 0.1:  # If there's some signal
                        threshold = max(0.3, mask_mean)  # Use mean or 0.3, whichever is higher
                    else:
                        threshold = 0.1  # Very low threshold if mask is very dim
                    
                    mask_binary = (fake_mask > threshold).numpy().astype(np.uint8)
                    
                    # Create BUSI-style WHITE mask on black background (grayscale)
                    # Since we're generating synthetic data FROM BUSI, masks should match BUSI format
                    mask_array = mask_binary * 255  # White mask (255) on black background (0)
                    
                    # Save images
                    img_pil = Image.fromarray(img_array, mode='L')
                    mask_pil = Image.fromarray(mask_array, mode='L')  # Grayscale white mask
                    
                    img_filename = f'server_test_output/epoch_{epoch}_{class_name}_{i+1}_img.png'
                    mask_filename = f'server_test_output/epoch_{epoch}_{class_name}_{i+1}_mask.png'
                    
                    img_pil.save(img_filename)
                    mask_pil.save(mask_filename)
        
        # IMPROVED: Print mask statistics for debugging
        print(f"\n📊 Epoch {epoch} Mask Statistics:")
        for class_name in ['benign', 'malignant']:
            stats = mask_stats[class_name]
            if stats:  # Only if we have samples
                avg_mean = sum(s['mean'] for s in stats) / len(stats)
                avg_max = sum(s['max'] for s in stats) / len(stats)
                avg_std = sum(s['std'] for s in stats) / len(stats)
                print(f"  {class_name.capitalize()}: mean={avg_mean:.3f}, max={avg_max:.3f}, std={avg_std:.3f}")
        
        self.generator.train()
    
    def plot_training_curves(self, g_losses, d_losses):
        """Plot training loss curves"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(g_losses, label='Generator Loss')
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Training curves saved as 'training_curves.png'")
    
    def generate_synthetic_dataset(self, num_benign=175, num_malignant=89, output_dir='synthetic_gan_dataset'):
        """Generate full synthetic dataset"""
        self.generator.eval()
        
        print(f"🎨 Generating Synthetic Dataset")
        print(f"   Target: {num_benign} benign + {num_malignant} malignant")
        print(f"   Mask format: BUSI-style white masks on black background (grayscale)")
        
        # Create output directories
        for class_name in ['benign', 'malignant']:
            os.makedirs(os.path.join(output_dir, class_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, class_name, 'masks'), exist_ok=True)
        
        # Generate samples
        for class_label, class_name, num_samples in [(0, 'benign', num_benign), (1, 'malignant', num_malignant)]:
            print(f"\nGenerating {num_samples} {class_name} samples...")
            
            with torch.no_grad():
                for i in tqdm(range(num_samples)):
                    # Generate noise
                    noise = torch.randn(1, self.noise_dim, device=self.device)
                    class_tensor = torch.tensor([class_label], device=self.device)
                    
                    # Generate image and mask
                    fake_image, fake_mask = self.generator(noise, class_tensor)
                    
                    # Convert to PIL images
                    fake_image = fake_image.squeeze().cpu()
                    fake_mask = fake_mask.squeeze().cpu()
                    
                    # Denormalize image from [-1, 1] to [0, 255]
                    img_array = ((fake_image + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                    
                    # IMPROVED: Use adaptive threshold based on mask statistics (SAME AS LOCAL VERSION)
                    mask_mean = torch.mean(fake_mask).item()
                    mask_max = torch.max(fake_mask).item()
                    
                    if mask_max > 0.1:  # If there's some signal
                        threshold = max(0.3, mask_mean)  # Use mean or 0.3, whichever is higher
                    else:
                        threshold = 0.1  # Very low threshold if mask is very dim
                    
                    mask_binary = (fake_mask > threshold).numpy().astype(np.uint8)
                    
                    # Create BUSI-style WHITE mask on black background (grayscale)
                    # Since we're generating synthetic data FROM BUSI, masks should match BUSI format
                    mask_array = mask_binary * 255  # White mask (255) on black background (0)
                    
                    # Save images
                    img_pil = Image.fromarray(img_array, mode='L')
                    mask_pil = Image.fromarray(mask_array, mode='L')  # Grayscale white mask
                    
                    img_filename = f'synthetic_{class_name}_{i+1:03d}.png'
                    mask_filename = f'synthetic_{class_name}_{i+1:03d}_mask.png'
                    
                    img_pil.save(os.path.join(output_dir, class_name, 'images', img_filename))
                    mask_pil.save(os.path.join(output_dir, class_name, 'masks', mask_filename))
        
        print(f"\n🎉 Generation Complete!")
        print(f"   Output: {output_dir}")
        print(f"   Total: {num_benign + num_malignant} synthetic image+mask pairs")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Server Simple GAN for Synthetic Medical Images')
    parser.add_argument('--mode', choices=['train', 'generate'], required=True,
                       help='Mode: train or generate')
    parser.add_argument('--data_dir', default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset on server')
    parser.add_argument('--output_dir', default='synthetic_gan_dataset',
                       help='Output directory for generated images')
    parser.add_argument('--checkpoint', default='simple_gan_final.pth',
                       help='Checkpoint path for generation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_benign', type=int, default=175,
                       help='Number of benign samples to generate')
    parser.add_argument('--num_malignant', type=int, default=89,
                       help='Number of malignant samples to generate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        print("🚀 Training Server GAN...")
        
        # Transforms for full resolution
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Dataset and dataloader
        dataset = ServerBUSIDataset(args.data_dir, transform=transform, mask_transform=mask_transform)
        
        if len(dataset) == 0:
            print("❌ No samples found in dataset. Cannot proceed with training.")
            print("   Please check the dataset path and structure.")
            return
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        # Initialize and train GAN
        gan = ServerSimpleGAN(device=device)
        gan.train(dataloader, num_epochs=args.epochs, save_interval=10)
        
        print("✅ Training completed!")
        
    elif args.mode == 'generate':
        print("🎨 Generating synthetic dataset...")
        
        # Load trained GAN
        gan = ServerSimpleGAN(device=device)
        epoch, g_loss, d_loss = gan.load_checkpoint(args.checkpoint)
        
        print(f"✅ Loaded checkpoint from epoch {epoch}")
        print(f"   Generator loss: {g_loss:.4f}")
        print(f"   Discriminator loss: {d_loss:.4f}")
        
        # Generate synthetic dataset
        gan.generate_synthetic_dataset(
            args.num_benign, 
            args.num_malignant, 
            args.output_dir
        )
        
        print("✅ Generation completed!")

if __name__ == "__main__":
    main() 