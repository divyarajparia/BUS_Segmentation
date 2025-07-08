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

class ServerGenerator(nn.Module):
    """Generator for Server Simple GAN"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=256):
        super(ServerGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, noise_dim)
        
        # Combined input dimension
        input_dim = noise_dim + noise_dim  # noise + class embedding
        
        # Image generation path
        self.img_layers = nn.Sequential(
            nn.Linear(input_dim, 256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 1, 4, 2, 1),     # 256x256
            nn.Tanh()
        )
        
        # Mask generation path
        self.mask_layers = nn.Sequential(
            nn.Linear(input_dim, 256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 1, 4, 2, 1),     # 256x256
            nn.Sigmoid()
        )
    
    def forward(self, noise, class_labels):
        # Embed class labels
        class_emb = self.class_embedding(class_labels)
        
        # Combine noise and class embedding
        combined_input = torch.cat([noise, class_emb], dim=1)
        
        # Generate image and mask
        fake_image = self.img_layers(combined_input)
        fake_mask = self.mask_layers(combined_input)
        
        return fake_image, fake_mask

class ServerDiscriminator(nn.Module):
    """Discriminator for Server Simple GAN"""
    
    def __init__(self, num_classes=2, img_size=256):
        super(ServerDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # Convolutional layers for image + mask
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),   # Input: image + mask
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification layers
        conv_output_size = 1024 * 8 * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size + img_size * img_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, class_labels):
        batch_size = image.size(0)
        
        # Get class embeddings and reshape
        class_emb = self.class_embedding(class_labels)
        class_emb = class_emb.view(batch_size, self.img_size, self.img_size)
        class_emb = class_emb.unsqueeze(1)  # Add channel dimension
        
        # The input 'image' is already combined (image + mask = 2 channels)
        # Process through conv layers directly
        conv_out = self.conv_layers(image)
        
        # Flatten and combine with class embedding
        conv_flat = conv_out.view(batch_size, -1)
        class_flat = class_emb.view(batch_size, -1)
        final_input = torch.cat([conv_flat, class_flat], dim=1)
        
        # Final classification
        output = self.classifier(final_input)
        return output.view(-1, 1).squeeze(1)

class ServerBUSIDataset(Dataset):
    """BUSI Dataset for Server Training"""
    
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
        
        # Load samples from all subdirectories
        for class_name in ['benign', 'malignant']:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            image_dir = os.path.join(class_dir, 'image')
            mask_dir = os.path.join(class_dir, 'mask')
            
            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
                
                for img_file in image_files:
                    # Find corresponding mask
                    base_name = img_file.replace('.png', '')
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
                        print(f"  Added {class_name}: {img_file}")
        
        print(f"Total samples loaded: {len(self.samples)}")
    
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
        
        # Combine image and mask for discriminator input
        combined = torch.cat([image, mask], dim=0)  # Stack along channel dimension
        
        return combined, sample['class_label']

class ServerSimpleGAN:
    """Simple GAN for Server Training"""
    
    def __init__(self, noise_dim=100, num_classes=2, img_size=256, device='cuda'):
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device
        
        # Initialize networks
        self.generator = ServerGenerator(noise_dim, num_classes, img_size).to(device)
        self.discriminator = ServerDiscriminator(num_classes, img_size).to(device)
        
        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        print(f"Initialized GAN on {device}")
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
    
    def train(self, dataloader, num_epochs=100, save_interval=10):
        """Train the GAN"""
        print(f"Training Simple GAN for {num_epochs} epochs...")
        
        g_losses = []
        d_losses = []
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for real_data, real_labels in progress_bar:
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                real_labels = real_labels.to(self.device)
                
                # Real and fake labels
                real_label = torch.ones(batch_size, device=self.device)
                fake_label = torch.zeros(batch_size, device=self.device)
                
                # ===============================
                # Train Discriminator
                # ===============================
                self.d_optimizer.zero_grad()
                
                # Train with real data
                output_real = self.discriminator(real_data, real_labels)
                d_loss_real = self.criterion(output_real, real_label)
                
                # Train with fake data
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
                
                fake_image, fake_mask = self.generator(noise, fake_labels)
                fake_data = torch.cat([fake_image, fake_mask], dim=1)
                
                output_fake = self.discriminator(fake_data.detach(), fake_labels)
                d_loss_fake = self.criterion(output_fake, fake_label)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # ===============================
                # Train Generator
                # ===============================
                self.g_optimizer.zero_grad()
                
                # Generator wants discriminator to think fake data is real
                output = self.discriminator(fake_data, fake_labels)
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
                self.save_checkpoint(f'server_gan_epoch_{epoch+1}.pth', epoch+1, avg_g_loss, avg_d_loss)
                self.generate_test_samples(epoch + 1)
        
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
        }, filename)
        print(f'Saved checkpoint: {filename}')
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print(f'Loaded checkpoint: {filename}')
    
    def generate_test_samples(self, epoch):
        """Generate test samples during training"""
        self.generator.eval()
        
        os.makedirs('server_test_output', exist_ok=True)
        
        # Initialize mask statistics tracking
        mask_stats = {'benign': [], 'malignant': []}
        
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
                    
                    mask_stats[class_name].append({
                        'mean': mask_mean,
                        'max': mask_max,
                        'min': mask_min,
                        'std': torch.std(fake_mask).item()
                    })
                    
                    # REALISTIC MASK GENERATION USING GENERATOR OUTPUT
                    # Use the generator's learned mask features and enhance them
                    h, w = fake_mask.shape
                    
                    # Convert generator mask to numpy for processing
                    generator_mask = fake_mask.numpy()
                    
                    # Enhance the generator's mask to create realistic tumor shapes
                    # Step 1: Normalize and threshold the generator output
                    mask_normalized = (generator_mask - generator_mask.min()) / (generator_mask.max() - generator_mask.min() + 1e-8)
                    
                    # Step 2: Use adaptive thresholding based on class
                    if class_label == 0:  # Benign - more defined, smaller regions
                        threshold = 0.6  # Higher threshold for cleaner, smaller masks
                        smooth_sigma = 1.5
                    else:  # Malignant - more complex, irregular shapes
                        threshold = 0.4  # Lower threshold for larger, more irregular masks
                        smooth_sigma = 0.8
                    
                    # Step 3: Create base mask from generator output
                    base_mask = mask_normalized > threshold
                    
                    # Step 4: Add realistic irregularities and smooth boundaries
                    try:
                        from scipy import ndimage
                        from scipy.ndimage import binary_erosion, binary_dilation
                        
                        # Apply morphological operations for realistic shapes
                        if class_label == 0:  # Benign - more regular
                            # Smooth and slightly regularize
                            processed_mask = ndimage.gaussian_filter(base_mask.astype(float), sigma=smooth_sigma)
                            final_mask = processed_mask > 0.5
                            
                            # Clean up small artifacts
                            final_mask = binary_erosion(final_mask, iterations=1)
                            final_mask = binary_dilation(final_mask, iterations=1)
                            
                        else:  # Malignant - more irregular and complex
                            # Create irregular boundaries
                            processed_mask = ndimage.gaussian_filter(base_mask.astype(float), sigma=smooth_sigma)
                            
                            # Add some controlled irregularity
                            noise_pattern = np.random.normal(0, 0.1, (h, w))
                            irregular_mask = processed_mask + noise_pattern
                            final_mask = irregular_mask > 0.4
                            
                            # Morphological operations for realistic malignant appearance
                            final_mask = binary_dilation(final_mask, iterations=1)
                            final_mask = binary_erosion(final_mask, iterations=1)
                        
                        # Ensure we have at least one connected component
                        from scipy.ndimage import label
                        labeled_mask, num_features = label(final_mask)
                        if num_features == 0:
                            # Fallback: create a small central region
                            center_y, center_x = h//2, w//2
                            radius = 15 if class_label == 0 else 25
                            y, x = np.ogrid[:h, :w]
                            fallback_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                            final_mask = fallback_mask
                        
                    except ImportError:
                        # Fallback without scipy - use generator output more directly
                        final_mask = mask_normalized > threshold
                        
                        # Simple morphological operations using numpy
                        if final_mask.sum() == 0:  # If empty, create minimal shape
                            center_y, center_x = h//2, w//2
                            radius = 15 if class_label == 0 else 25
                            y, x = np.ogrid[:h, :w]
                            final_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                    
                    # Convert to binary mask
                    mask_binary = final_mask.astype(np.uint8)
                    
                    # Create BUSI-style WHITE mask on black background (grayscale)
                    mask_array = mask_binary * 255  # White mask (255) on black background (0)
                    
                    # Denormalize image from [-1, 1] to [0, 255]
                    img_array = ((fake_image + 1) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
                    
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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Server Simple GAN for Synthetic Medical Images')
    parser.add_argument('--mode', choices=['train', 'generate'], default='train', help='Mode: train or generate')
    parser.add_argument('--data_dir', default='dataset/BioMedicalDataset/BUSI', help='Data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file to load')
    parser.add_argument('--output_dir', default='synthetic_gan_dataset', help='Output directory for generated data')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    if args.mode == 'train':
        # Dataset and dataloader
        print(f"Loading dataset from: {args.data_dir}")
        dataset = ServerBUSIDataset(args.data_dir, transform=transform, mask_transform=mask_transform)
        
        if len(dataset) == 0:
            print("❌ No samples found! Check your data directory.")
            return
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print(f"Training dataset: {len(dataset)} samples")
        
        # Initialize and train GAN
        gan = ServerSimpleGAN(device=device)
        
        if args.checkpoint:
            gan.load_checkpoint(args.checkpoint)
        
        g_losses, d_losses = gan.train(dataloader, num_epochs=args.epochs, save_interval=10)
        
        print("✅ Training completed!")
        print("   Check 'server_test_output/' for generated samples during training")
        
    elif args.mode == 'generate':
        if not args.checkpoint:
            print("❌ Checkpoint file required for generation mode")
            return
        
        # Initialize GAN and load checkpoint
        gan = ServerSimpleGAN(device=device)
        gan.load_checkpoint(args.checkpoint)
        
        # Generate synthetic dataset
        gan.generate_synthetic_dataset(output_dir=args.output_dir)
        
        print(f"✅ Synthetic dataset generated in: {args.output_dir}")

if __name__ == "__main__":
    main() 