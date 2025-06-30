"""
Simplified Diffusion Model for BUSI Synthetic Generation
======================================================

A starter implementation for generating synthetic BUSI images using diffusion models.
This is a simplified version to get you started quickly.

Usage:
    python simple_diffusion_busi.py --data_dir /path/to/busi --mode train
    python simple_diffusion_busi.py --checkpoint /path/to/model.pth --mode generate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse

class BUSIDataset(Dataset):
    """Simple BUSI dataset loader"""
    
    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        # Load samples
        for class_name in ['benign', 'malignant']:
            class_dir = os.path.join(data_dir, class_name, 'images')
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'path': os.path.join(class_dir, img_file),
                            'class': self.class_to_idx[class_name]
                        })
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('L')
        image = self.transform(image)
        return image, sample['class']

class SimpleUNet(nn.Module):
    """Simplified U-Net for diffusion denoising"""
    
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, num_classes=2):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, time_dim)
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        
        # Time conditioning layers
        self.time_proj = nn.ModuleList([
            nn.Linear(time_dim, 64),
            nn.Linear(time_dim, 128),
            nn.Linear(time_dim, 256),
            nn.Linear(time_dim, 512),
            nn.Linear(time_dim, 1024)
        ])
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU()
        )
    
    def forward(self, x, t, class_label=None):
        # Time embedding
        t_emb = self.get_time_embedding(t, 256)
        t_emb = self.time_mlp(t_emb)
        
        # Add class conditioning
        if class_label is not None:
            class_emb = self.class_emb(class_label)
            t_emb = t_emb + class_emb
        
        # Encoder
        e1 = self.enc1(x)
        e1 = e1 + self.time_proj[0](t_emb)[:, :, None, None]
        
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e2 = e2 + self.time_proj[1](t_emb)[:, :, None, None]
        
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e3 = e3 + self.time_proj[2](t_emb)[:, :, None, None]
        
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e4 = e4 + self.time_proj[3](t_emb)[:, :, None, None]
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(e4, 2))
        bottleneck = bottleneck + self.time_proj[4](t_emb)[:, :, None, None]
        
        # Decoder
        d4 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)
    
    def get_time_embedding(self, timesteps, embedding_dim):
        """Sinusoidal time embeddings"""
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :].to(timesteps.device)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class SimpleDiffusion:
    """Simplified diffusion process"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Add noise to images"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_start, t, class_labels=None):
        """Training loss"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, class_labels)
        return F.mse_loss(noise, predicted_noise)
    
    @torch.no_grad()
    def sample(self, model, shape, class_labels=None, device='cuda'):
        """Generate samples"""
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling'):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t, class_labels)
            
            # Remove noise
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            # Compute previous sample
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(beta_t) * noise
            else:
                x = pred_x0
        
        return x

def train_model(data_dir, num_epochs=50, batch_size=8, lr=1e-4, device='cuda'):
    """Train the diffusion model"""
    
    # Dataset and dataloader
    dataset = BUSIDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model and diffusion
    model = SimpleUNet().to(device)
    diffusion = SimpleDiffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Training on {len(dataset)} samples")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, class_labels) in enumerate(pbar):
            images = images.to(device)
            class_labels = class_labels.to(device)
            
            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            
            # Calculate loss
            loss = diffusion.p_losses(model, images, t, class_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }, f'diffusion_model_epoch_{epoch+1}.pth')

def generate_synthetic_images(checkpoint_path, output_dir, num_benign=100, num_malignant=100, device='cuda'):
    """Generate synthetic images with flexible class control"""
    
    # Load model
    model = SimpleUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = SimpleDiffusion()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each class with custom counts
    class_counts = {'benign': num_benign, 'malignant': num_malignant}
    
    for class_idx, class_name in enumerate(['benign', 'malignant']):
        num_samples = class_counts[class_name]
        
        if num_samples == 0:
            print(f'Skipping {class_name} (0 samples requested)')
            continue
            
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f'Generating {num_samples} {class_name} samples...')
        
        # Generate in batches
        batch_size = 8
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        sample_count = 0
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - sample_count)
            
            # Class labels
            class_labels = torch.full((current_batch_size,), class_idx, device=device)
            
            # Generate samples
            samples = diffusion.sample(
                model, 
                (current_batch_size, 1, 256, 256), 
                class_labels, 
                device
            )
            
            # Save images
            for i in range(current_batch_size):
                # Convert to PIL
                img_array = ((samples[i].cpu().numpy()[0] + 1) * 127.5).astype(np.uint8)
                img_pil = Image.fromarray(img_array, mode='L')
                
                # Save
                filename = f'synthetic_{class_name}_{sample_count:04d}.png'
                img_pil.save(os.path.join(class_dir, filename))
                sample_count += 1
        
        print(f'Saved {num_samples} {class_name} samples to {class_dir}')
    
    print(f'Total generated: {num_benign} benign + {num_malignant} malignant = {num_benign + num_malignant} samples')

def main():
    parser = argparse.ArgumentParser(description='Simple Diffusion for BUSI')
    parser.add_argument('--data_dir', type=str, help='Path to BUSI dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for generation')
    parser.add_argument('--output_dir', type=str, default='./synthetic_busi', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples per class (legacy)')
    parser.add_argument('--num_benign', type=int, default=None, help='Number of benign samples to generate')
    parser.add_argument('--num_malignant', type=int, default=None, help='Number of malignant samples to generate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.mode == 'train':
        if not args.data_dir:
            raise ValueError('data_dir required for training')
        train_model(args.data_dir, args.num_epochs, args.batch_size, device=device)
    
    elif args.mode == 'generate':
        if not args.checkpoint:
            raise ValueError('checkpoint required for generation')
        
        # Handle flexible generation counts
        num_benign = args.num_benign if args.num_benign is not None else args.num_samples
        num_malignant = args.num_malignant if args.num_malignant is not None else args.num_samples
        
        generate_synthetic_images(args.checkpoint, args.output_dir, num_benign, num_malignant, device)

if __name__ == '__main__':
    main() 