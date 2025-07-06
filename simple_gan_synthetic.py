#!/usr/bin/env python3
"""
Simple GAN for Synthetic BUSI Image Generation
==============================================
A straightforward GAN approach for generating synthetic breast ultrasound images
with corresponding masks. This provides a fair comparison with CycleGAN/style transfer.

Key Features:
- Simple Generator/Discriminator architecture
- Class-conditional generation (benign/malignant)
- Joint image+mask generation
- Much simpler than diffusion models
- Direct comparison with style transfer approaches
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

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SimpleGenerator(nn.Module):
    """Simple Generator for medical images and masks"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=256):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, 50)
        
        # Initial dense layer
        self.initial_size = img_size // 16  # 16x16 for 256x256 images
        self.initial_dim = 512
        
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 50, self.initial_dim * self.initial_size * self.initial_size),
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

class SimpleDiscriminator(nn.Module):
    """Simple Discriminator for medical images"""
    
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
        )
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1),
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

class BUSIDataset(Dataset):
    """Simple BUSI dataset loader"""
    
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
        
        # Load samples
        for class_name in ['benign', 'malignant']:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            image_dir = os.path.join(class_dir, 'image')
            mask_dir = os.path.join(class_dir, 'mask')
            
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('L')
        if self.transform:
            image = self.transform(image)
        
        # Load mask
        mask = Image.open(sample['mask_path']).convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask, sample['label']

class SimpleGAN:
    """Simple GAN for medical image generation"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=256, device='cuda'):
        self.device = device
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # Initialize networks
        self.generator = SimpleGenerator(noise_dim, num_classes, img_size).to(device)
        self.discriminator = SimpleDiscriminator(num_classes, img_size).to(device)
        
        # Loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Initialize network weights"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def train(self, dataloader, num_epochs=100, save_interval=10):
        """Train the GAN"""
        print(f"Training Simple GAN for {num_epochs} epochs...")
        
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
                
                # Labels for real and fake
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
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
                g_gan_loss = self.criterion_gan(fake_pred, real_labels)
                
                # L1 loss for better image quality (optional)
                g_l1_loss = self.criterion_l1(fake_masks, real_masks) * 10  # Weight L1 loss
                
                # Total generator loss
                g_loss = g_gan_loss + g_l1_loss
                g_loss.backward()
                self.optimizer_G.step()
                
                # Update progress
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'D_Loss': f'{d_loss.item():.4f}',
                    'G_Loss': f'{g_loss.item():.4f}'
                })
            
            # Print epoch summary
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            print(f'Epoch {epoch+1}/{num_epochs} - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}')
            
            # Save model checkpoints
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'simple_gan_epoch_{epoch+1}.pth', epoch, avg_g_loss, avg_d_loss)
    
    def save_checkpoint(self, filename, epoch, g_loss, d_loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
        }, filename)
        print(f'Saved checkpoint: {filename}')
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        return checkpoint['epoch'], checkpoint['g_loss'], checkpoint['d_loss']
    
    def generate_samples(self, num_samples, class_label, save_dir):
        """Generate synthetic samples"""
        self.generator.eval()
        
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'masks'), exist_ok=True)
        
        class_name = 'benign' if class_label == 0 else 'malignant'
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc=f'Generating {class_name} samples'):
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
                
                # Denormalize mask from [0, 1] to [0, 255]
                mask_array = (fake_mask * 255).clamp(0, 255).numpy().astype(np.uint8)
                
                # Save images
                img_pil = Image.fromarray(img_array, mode='L')
                mask_pil = Image.fromarray(mask_array, mode='L')
                
                img_filename = f'synthetic_{class_name}_{i+1:03d}.png'
                mask_filename = f'synthetic_{class_name}_{i+1:03d}_mask.png'
                
                img_pil.save(os.path.join(save_dir, 'images', img_filename))
                mask_pil.save(os.path.join(save_dir, 'masks', mask_filename))
        
        print(f'Generated {num_samples} {class_name} samples in {save_dir}')

def train_simple_gan(data_dir, num_epochs=100, batch_size=4, device='cuda'):
    """Train Simple GAN on BUSI dataset"""
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Dataset and dataloader
    dataset = BUSIDataset(data_dir, transform=transform, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"Training dataset: {len(dataset)} samples")
    
    # Initialize and train GAN
    gan = SimpleGAN(device=device)
    gan.train(dataloader, num_epochs=num_epochs, save_interval=10)
    
    return gan

def generate_synthetic_dataset(checkpoint_path, output_dir, num_benign=175, num_malignant=89, device='cuda'):
    """Generate synthetic dataset using trained GAN"""
    
    print(f"🎨 Generating Synthetic Dataset using Simple GAN")
    print(f"   Target: {num_benign} benign + {num_malignant} malignant")
    
    # Load trained GAN
    gan = SimpleGAN(device=device)
    epoch, g_loss, d_loss = gan.load_checkpoint(checkpoint_path)
    
    print(f"✅ Loaded checkpoint from epoch {epoch}")
    print(f"   Generator loss: {g_loss:.4f}")
    print(f"   Discriminator loss: {d_loss:.4f}")
    
    # Create output directories
    benign_dir = os.path.join(output_dir, 'benign')
    malignant_dir = os.path.join(output_dir, 'malignant')
    
    # Generate samples
    gan.generate_samples(num_benign, class_label=0, save_dir=benign_dir)
    gan.generate_samples(num_malignant, class_label=1, save_dir=malignant_dir)
    
    print(f"🎉 Generation Complete!")
    print(f"   Output: {output_dir}")
    print(f"   Total: {num_benign + num_malignant} synthetic image+mask pairs")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple GAN for Synthetic Medical Images')
    parser.add_argument('--mode', choices=['train', 'generate'], required=True,
                       help='Mode: train or generate')
    parser.add_argument('--data_dir', default='dataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--output_dir', default='synthetic_gan_output',
                       help='Output directory for generated images')
    parser.add_argument('--checkpoint', default='simple_gan_epoch_100.pth',
                       help='Checkpoint path')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_benign', type=int, default=175,
                       help='Number of benign samples to generate')
    parser.add_argument('--num_malignant', type=int, default=89,
                       help='Number of malignant samples to generate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        print("🚀 Training Simple GAN...")
        train_simple_gan(args.data_dir, args.epochs, args.batch_size, device)
        print("✅ Training completed!")
        
    elif args.mode == 'generate':
        print("🎨 Generating synthetic dataset...")
        generate_synthetic_dataset(
            args.checkpoint, 
            args.output_dir, 
            args.num_benign, 
            args.num_malignant, 
            device
        )
        print("✅ Generation completed!")

if __name__ == "__main__":
    main() 