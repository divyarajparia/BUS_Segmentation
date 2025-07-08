import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import csv

class ConditionalGenerator(nn.Module):
    """Conditional Generator that produces both ultrasound images and segmentation masks"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_channels=1, img_size=256):
        super(ConditionalGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Class embedding layer
        self.class_embedding = nn.Embedding(num_classes, noise_dim)
        
        # Input dimension: noise + class embedding
        input_dim = noise_dim * 2
        
        # Shared initial layers
        self.initial = nn.Sequential(
            nn.Linear(input_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4))
        )
        
        # Shared upsampling layers
        self.shared_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        
        # Image generation head (128x128 -> 256x256)
        self.image_head = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Mask generation head (128x128 -> 256x256)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output range [0, 1]
        )
        
    def forward(self, noise, labels):
        # Get class embeddings
        class_emb = self.class_embedding(labels)
        
        # Concatenate noise and class embedding
        x = torch.cat([noise, class_emb], dim=1)
        
        # Pass through shared layers
        x = self.initial(x)
        x = self.shared_layers(x)
        
        # Generate image and mask
        image = self.image_head(x)
        mask = self.mask_head(x)
        
        return image, mask

class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator that evaluates image-mask pairs with class information"""
    
    def __init__(self, num_classes=2, img_channels=1, img_size=256):
        super(ConditionalDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # Main discriminator network
        # Input: image + mask + class_map = 3 channels
        self.conv_layers = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(img_channels + 1 + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 1x1
            nn.Conv2d(1024, 1, kernel_size=8, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, image, mask, labels):
        batch_size = image.size(0)
        
        # Create class conditioning map
        class_emb = self.class_embedding(labels)
        class_map = class_emb.view(batch_size, 1, self.img_size, self.img_size)
        
        # Concatenate image, mask, and class map
        x = torch.cat([image, mask, class_map], dim=1)
        
        # Pass through discriminator
        output = self.conv_layers(x)
        # Ensure output shape is [batch_size]
        return output.view(batch_size)

class BUSIDataset(Dataset):
    """BUSI Dataset loader for GAN training"""
    
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
        
        # Load samples
        for class_name in ['benign', 'malignant']:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found, skipping...")
                continue
                
            image_dir = os.path.join(class_dir, 'image')
            mask_dir = os.path.join(class_dir, 'mask')
            
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                image_files = sorted([f for f in os.listdir(image_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                for img_file in image_files:
                    # Find corresponding mask
                    base_name = os.path.splitext(img_file)[0]
                    mask_file = f"{base_name}_mask.png"
                    
                    img_path = os.path.join(image_dir, img_file)
                    mask_path = os.path.join(mask_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        class_label = 0 if class_name == 'benign' else 1
                        self.samples.append({
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'class_label': class_label,
                            'class_name': class_name
                        })
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        benign_count = sum(1 for s in self.samples if s['class_label'] == 0)
        malignant_count = sum(1 for s in self.samples if s['class_label'] == 1)
        print(f"  Benign: {benign_count}, Malignant: {malignant_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and mask
        image = Image.open(sample['image_path']).convert('L')
        mask = Image.open(sample['mask_path']).convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return image, mask, sample['class_label']

class BUSIConditionalGAN:
    """Conditional GAN for BUSI dataset synthesis"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=256, device='cuda', lr=0.0002):
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.generator = ConditionalGenerator(noise_dim, num_classes, 1, img_size).to(self.device)
        self.discriminator = ConditionalDiscriminator(num_classes, 1, img_size).to(self.device)
        
        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        print(f"Initialized Conditional GAN on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _weights_init(self, m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def train(self, dataloader, num_epochs=200, save_interval=20, checkpoint_dir='checkpoints'):
        """Train the conditional GAN"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs('training_samples', exist_ok=True)
        
        print(f"Training Conditional GAN for {num_epochs} epochs...")
        
        g_losses = []
        d_losses = []
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for real_images, real_masks, real_labels in progress_bar:
                batch_size = real_images.size(0)
                
                # Move to device
                real_images = real_images.to(self.device)
                real_masks = real_masks.to(self.device)
                real_labels = real_labels.to(self.device)
                
                # Labels for real and fake
                real_label = torch.ones(batch_size, device=self.device)
                fake_label = torch.zeros(batch_size, device=self.device)
                
                # ===============================
                # Train Discriminator
                # ===============================
                self.d_optimizer.zero_grad()
                
                # Train with real data
                output_real = self.discriminator(real_images, real_masks, real_labels)
                d_loss_real = self.criterion(output_real, real_label)
                
                # Train with fake data
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                
                fake_images, fake_masks = self.generator(noise, fake_labels)
                output_fake = self.discriminator(fake_images.detach(), fake_masks.detach(), fake_labels)
                d_loss_fake = self.criterion(output_fake, fake_label)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # ===============================
                # Train Generator
                # ===============================
                self.g_optimizer.zero_grad()
                
                # Generator wants discriminator to classify fake as real
                output = self.discriminator(fake_images, fake_masks, fake_labels)
                g_loss = self.criterion(output, real_label)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # Update losses
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'D_Loss': f'{d_loss.item():.4f}',
                    'G_Loss': f'{g_loss.item():.4f}'
                })
            
            # Average losses for epoch
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs} - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}')
            
            # Save checkpoint and generate samples
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'busi_gan_epoch_{epoch+1}.pth'), 
                    epoch+1, avg_g_loss, avg_d_loss
                )
                self.generate_training_samples(epoch + 1)
        
        # Save final model
        self.save_checkpoint(
            os.path.join(checkpoint_dir, 'busi_gan_final.pth'), 
            num_epochs, avg_g_loss, avg_d_loss
        )
        
        return g_losses, d_losses
    
    def save_checkpoint(self, filename, epoch, g_loss, d_loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            'noise_dim': self.noise_dim,
            'num_classes': self.num_classes,
            'img_size': self.img_size
        }, filename)
        print(f'Saved checkpoint: {filename}')
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        print(f'Loaded checkpoint: {filename}')
        return checkpoint['epoch']
    
    def generate_training_samples(self, epoch, num_samples=4):
        """Generate sample images during training for monitoring"""
        self.generator.eval()
        
        with torch.no_grad():
            for class_label in [0, 1]:
                class_name = 'benign' if class_label == 0 else 'malignant'
                
                for i in range(num_samples):
                    noise = torch.randn(1, self.noise_dim, device=self.device)
                    label = torch.tensor([class_label], device=self.device)
                    
                    fake_image, fake_mask = self.generator(noise, label)
                    
                    # Convert to numpy and denormalize
                    image = fake_image.squeeze().cpu().numpy()
                    mask = fake_mask.squeeze().cpu().numpy()
                    
                    # Denormalize image from [-1, 1] to [0, 255]
                    image = ((image + 1) * 127.5).astype(np.uint8)
                    mask = (mask * 255).astype(np.uint8)
                    
                    # Save
                    Image.fromarray(image, mode='L').save(
                        f'training_samples/epoch_{epoch}_{class_name}_{i+1}_image.png'
                    )
                    Image.fromarray(mask, mode='L').save(
                        f'training_samples/epoch_{epoch}_{class_name}_{i+1}_mask.png'
                    )
        
        self.generator.train()
    
    def generate_synthetic_dataset(self, output_dir, num_benign=175, num_malignant=89):
        """Generate synthetic dataset with specified numbers of each class"""
        self.generator.eval()
        
        # Create output directories
        for class_name in ['benign', 'malignant']:
            os.makedirs(os.path.join(output_dir, class_name, 'image'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, class_name, 'mask'), exist_ok=True)
        
        # Generate samples
        samples_to_generate = [('benign', 0, num_benign), ('malignant', 1, num_malignant)]
        csv_data = []
        
        print("Generating synthetic dataset...")
        
        with torch.no_grad():
            for class_name, class_label, num_samples in samples_to_generate:
                print(f"Generating {num_samples} {class_name} samples...")
                
                for i in tqdm(range(num_samples), desc=f"Generating {class_name}"):
                    # Generate noise and label
                    noise = torch.randn(1, self.noise_dim, device=self.device)
                    label = torch.tensor([class_label], device=self.device)
                    
                    # Generate image and mask
                    fake_image, fake_mask = self.generator(noise, label)
                    
                    # Convert to numpy
                    image = fake_image.squeeze().cpu().numpy()
                    mask = fake_mask.squeeze().cpu().numpy()
                    
                    # Denormalize and convert to uint8
                    image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    mask = (mask * 255).astype(np.uint8)
                    
                    # Apply threshold to mask to ensure binary segmentation
                    mask = (mask > 127).astype(np.uint8) * 255
                    
                    # Create filenames
                    img_filename = f"synthetic_{class_name}_{i+1:03d}.png"
                    mask_filename = f"synthetic_{class_name}_{i+1:03d}_mask.png"
                    
                    # Save images
                    img_path = os.path.join(output_dir, class_name, 'image', img_filename)
                    mask_path = os.path.join(output_dir, class_name, 'mask', mask_filename)
                    
                    Image.fromarray(image, mode='L').save(img_path)
                    Image.fromarray(mask, mode='L').save(mask_path)
                    
                    # Add to CSV data
                    csv_data.append({
                        'image_path': os.path.join(class_name, 'image', img_filename),
                        'mask_path': os.path.join(class_name, 'mask', mask_filename),
                        'class': class_name,
                        'class_label': class_label
                    })
        
        # Save CSV file
        csv_path = os.path.join(output_dir, 'synthetic_dataset.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'mask_path', 'class', 'class_label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"✅ Synthetic dataset generated!")
        print(f"   Output directory: {output_dir}")
        print(f"   Total samples: {len(csv_data)}")
        print(f"   Benign: {num_benign}, Malignant: {num_malignant}")
        print(f"   Dataset CSV: {csv_path}")
        
        self.generator.train()

def main():
    parser = argparse.ArgumentParser(description='BUSI Conditional GAN for Synthetic Data Generation')
    parser.add_argument('--mode', choices=['train', 'generate'], default='train', 
                       help='Mode: train the GAN or generate synthetic data')
    parser.add_argument('--data_dir', default='dataset/BUSI', 
                       help='Path to BUSI dataset directory')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, 
                       help='Learning rate')
    parser.add_argument('--noise_dim', type=int, default=100, 
                       help='Noise vector dimension')
    parser.add_argument('--checkpoint', type=str, 
                       help='Checkpoint file to load (for generation or resuming training)')
    parser.add_argument('--checkpoint_dir', default='checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--output_dir', default='synthetic_busi_dataset', 
                       help='Output directory for generated synthetic data')
    parser.add_argument('--num_benign', type=int, default=175, 
                       help='Number of benign samples to generate')
    parser.add_argument('--num_malignant', type=int, default=89, 
                       help='Number of malignant samples to generate')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data transforms
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    if args.mode == 'train':
        # Load dataset
        print(f"Loading BUSI dataset from: {args.data_dir}")
        dataset = BUSIDataset(args.data_dir, transform=image_transform, mask_transform=mask_transform)
        
        if len(dataset) == 0:
            print("❌ No samples found! Please check your data directory structure.")
            print("Expected structure:")
            print("  data_dir/")
            print("    benign/")
            print("      image/")
            print("      mask/")
            print("    malignant/")
            print("      image/")
            print("      mask/")
            return
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        # Initialize GAN
        gan = BUSIConditionalGAN(
            noise_dim=args.noise_dim,
            num_classes=2,
            img_size=256,
            device=device,
            lr=args.lr
        )
        
        # Load checkpoint if provided
        start_epoch = 0
        if args.checkpoint:
            start_epoch = gan.load_checkpoint(args.checkpoint)
            print(f"Resuming training from epoch {start_epoch}")
        
        # Train
        g_losses, d_losses = gan.train(
            dataloader, 
            num_epochs=args.epochs, 
            save_interval=20,
            checkpoint_dir=args.checkpoint_dir
        )
        
        print("✅ Training completed!")
        
    elif args.mode == 'generate':
        if not args.checkpoint:
            print("❌ Checkpoint file required for generation mode!")
            print("Use --checkpoint to specify the trained model file")
            return
        
        # Initialize GAN
        gan = BUSIConditionalGAN(
            noise_dim=args.noise_dim,
            num_classes=2,
            img_size=256,
            device=device
        )
        
        # Load checkpoint
        gan.load_checkpoint(args.checkpoint)
        
        # Generate synthetic dataset
        gan.generate_synthetic_dataset(
            output_dir=args.output_dir,
            num_benign=args.num_benign,
            num_malignant=args.num_malignant
        )

if __name__ == "__main__":
    main() 