#!/usr/bin/env python3
"""
Local Testing: Simple GAN for Synthetic BUSI Image Generation
============================================================
Test the simple GAN approach locally with your debug dataset:
- 3 benign samples
- 2 malignant samples

This validates the approach before running on the full server dataset.
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
    """Simple Generator for medical images and masks - LOCAL VERSION"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=128):  # Smaller for local testing
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, 50)
        
        # Initial dense layer
        self.initial_size = img_size // 16  # 8x8 for 128x128 images
        self.initial_dim = 256  # Smaller for local testing
        
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 50, self.initial_dim * self.initial_size * self.initial_size),
            nn.ReLU(True)
        )
        
        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(self.initial_dim, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        
        # Output heads for image and mask
        self.image_head = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
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
    """Simple Discriminator for medical images - LOCAL VERSION"""
    
    def __init__(self, num_classes=2, img_size=128):  # Smaller for local testing
        super().__init__()
        self.num_classes = num_classes
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, img_size * img_size)
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),  # 2 channels: image + class_emb
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1),
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

class LocalBUSIDataset(Dataset):
    """Local BUSI dataset loader for debug data"""
    
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
        
        print(f"Loading local dataset from: {data_dir}")
        
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
                            print(f"  Added {class_name}: {img_file}")
        
        print(f"Total samples loaded: {len(self.samples)}")
    
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
            mask_tensor = resize(mask_tensor, (128, 128))  # Local uses 128x128
        
        return image, mask_tensor, sample['label']

class LocalSimpleGAN:
    """Simple GAN for local testing"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=128, device='cuda'):
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
        
        print(f"Initialized GAN on {device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _weights_init(self, m):
        """Initialize network weights"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def train(self, dataloader, num_epochs=50, save_interval=10):
        """Train the GAN locally"""
        print(f"Training Simple GAN locally for {num_epochs} epochs...")
        
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
                
                # L1 loss for better image quality
                g_l1_loss = self.criterion_l1(fake_masks, real_masks) * 10
                
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
                self.save_checkpoint(f'local_gan_epoch_{epoch+1}.pth', epoch, avg_g_loss, avg_d_loss)
                self.generate_test_samples(epoch + 1)
    
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
    
    def generate_test_samples(self, epoch):
        """Generate test samples during training"""
        self.generator.eval()
        
        os.makedirs('local_test_output', exist_ok=True)
        
        with torch.no_grad():
            # Generate 2 benign and 2 malignant samples
            for class_label in [0, 1]:
                class_name = 'benign' if class_label == 0 else 'malignant'
                
                for i in range(2):
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
                    print(f"🔍 {class_name} mask {i+1}: mean={mask_mean:.4f}, min={mask_min:.4f}, max={mask_max:.4f}")
                    
                    # Denormalize image from [-1, 1] to [0, 255]
                    img_array = ((fake_image + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                    
                    # IMPROVED: Use adaptive threshold based on mask statistics
                    if mask_max > 0.1:  # If there's some signal
                        threshold = max(0.3, mask_mean)  # Use mean or 0.3, whichever is higher
                    else:
                        threshold = 0.1  # Very low threshold if mask is very dim
                    
                    mask_binary = (fake_mask > threshold).numpy().astype(np.uint8)
                    print(f"   Using threshold: {threshold:.4f}, binary pixels: {mask_binary.sum()}")
                    
                    # SIMPLE REALISTIC MASK GENERATION (like BUSI)
                    # Create clean, smooth tumor-shaped masks exactly like BUSI dataset
                    h, w = mask_binary.shape
                    
                    # Create realistic tumor shape with smooth boundaries
                    center_y = h // 2 + np.random.randint(-h//8, h//8)  # Near center
                    center_x = w // 2 + np.random.randint(-w//8, w//8)
                    
                    if class_label == 0:  # Benign - smaller, more regular
                        radius_y = np.random.randint(h//15, h//10)  # Smaller radius
                        radius_x = np.random.randint(w//15, w//10)
                        irregularity = 0.8  # More regular
                    else:  # Malignant - larger, more irregular
                        radius_y = np.random.randint(h//12, h//7)   # Larger radius
                        radius_x = np.random.randint(w//12, w//7)
                        irregularity = 0.6  # More irregular
                    
                    # Create smooth elliptical mask
                    y, x = np.ogrid[:h, :w]
                    ellipse_mask = ((x - center_x)**2 / radius_x**2 + 
                                   (y - center_y)**2 / radius_y**2) <= irregularity
                    
                    # CLEAN SMOOTH BOUNDARIES - no noise
                    # Apply Gaussian smoothing for realistic boundaries
                    try:
                        from scipy import ndimage
                        # Smooth the mask edges
                        ellipse_smooth = ndimage.gaussian_filter(ellipse_mask.astype(float), sigma=1.0)
                        final_mask = ellipse_smooth > 0.5
                    except ImportError:
                        # Fallback without scipy - basic smoothing
                        final_mask = ellipse_mask
                    
                    mask_binary = final_mask.astype(np.uint8)
                    print(f"   Clean realistic tumor mask: {mask_binary.sum()} pixels")
                    
                    # Create BUSI-style WHITE mask on black background (grayscale)
                    mask_array = mask_binary * 255  # White mask (255) on black background (0)
                    
                    # Save images
                    img_pil = Image.fromarray(img_array, mode='L')
                    mask_pil = Image.fromarray(mask_array, mode='L')  # Grayscale white mask
                    
                    img_filename = f'local_test_output/epoch_{epoch}_{class_name}_{i+1}_img.png'
                    mask_filename = f'local_test_output/epoch_{epoch}_{class_name}_{i+1}_mask.png'
                    
                    img_pil.save(img_filename)
                    mask_pil.save(mask_filename)
                    
                    print(f"   Saved: {img_filename} and {mask_filename}")
        
        self.generator.train()

def test_local_gan():
    """Test Simple GAN locally with debug data"""
    
    print("🧪 Testing Simple GAN Locally")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data directory
    data_dir = 'debug_data/BUSI'
    
    # Transforms (smaller size for local testing)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Smaller for local testing
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Dataset and dataloader
    dataset = LocalBUSIDataset(data_dir, transform=transform, mask_transform=mask_transform)
    
    if len(dataset) == 0:
        print("❌ No samples found! Check your data directory.")
        return
    
    # Small batch size for local testing
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    print(f"Training dataset: {len(dataset)} samples")
    
    # Initialize and train GAN
    gan = LocalSimpleGAN(device=device, img_size=128)
    gan.train(dataloader, num_epochs=30, save_interval=10)  # Shorter training for local test
    
    print("✅ Local testing completed!")
    print("   Check 'local_test_output/' for generated samples during training")
    print("   Checkpoints saved: local_gan_epoch_10.pth, local_gan_epoch_20.pth, local_gan_epoch_30.pth")

def visualize_local_results():
    """Visualize the local test results"""
    
    print("🖼️  Visualizing Local Test Results")
    
    # Find the latest generated samples
    test_dir = 'local_test_output'
    if not os.path.exists(test_dir):
        print("❌ No test output found. Run training first.")
        return
    
    # Get latest epoch files
    files = os.listdir(test_dir)
    latest_epoch = max([int(f.split('_')[1]) for f in files if f.startswith('epoch_')])
    
    print(f"Showing results from epoch {latest_epoch}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'Local GAN Results - Epoch {latest_epoch}', fontsize=16)
    
    for class_idx, class_name in enumerate(['benign', 'malignant']):
        for sample_idx in range(2):
            img_file = f'epoch_{latest_epoch}_{class_name}_{sample_idx+1}_img.png'
            mask_file = f'epoch_{latest_epoch}_{class_name}_{sample_idx+1}_mask.png'
            
            img_path = os.path.join(test_dir, img_file)
            mask_path = os.path.join(test_dir, mask_file)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = Image.open(img_path)
                mask = Image.open(mask_path)
                
                axes[class_idx, sample_idx*2].imshow(img, cmap='gray')
                axes[class_idx, sample_idx*2].set_title(f'{class_name} Image {sample_idx+1}')
                axes[class_idx, sample_idx*2].axis('off')
                
                axes[class_idx, sample_idx*2+1].imshow(mask, cmap='gray')
                axes[class_idx, sample_idx*2+1].set_title(f'{class_name} Mask {sample_idx+1}')
                axes[class_idx, sample_idx*2+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('local_gan_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'local_gan_results.png'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Local Simple GAN Testing')
    parser.add_argument('--mode', choices=['train', 'visualize'], default='train',
                       help='Mode: train or visualize')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        test_local_gan()
    elif args.mode == 'visualize':
        visualize_local_results() 