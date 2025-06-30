"""
FIXED Diffusion Model for BUSI Dataset
=====================================

This version fixes:
1. Image normalization issues (black images)
2. Missing mask generation
3. Proper output constraints
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class BUSIDataset(Dataset):
    """BUSI dataset loader that works with CSV files"""
    
    def __init__(self, data_dir, img_size=256, split='train'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        # Load samples from CSV
        import pandas as pd
        csv_path = os.path.join(data_dir, f'{split}_frame.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                image_path = str(row['image_path'])
                mask_path = str(row['mask_path'])
                
                # Determine class from filename
                if image_path.startswith('benign'):
                    class_name = 'benign'
                elif image_path.startswith('malignant'):
                    class_name = 'malignant'
                else:
                    continue
                
                # Check for image existence
                possible_image_paths = [
                    os.path.join(data_dir, image_path),
                    os.path.join(data_dir, 'image', image_path),
                    os.path.join(data_dir, class_name, 'image', image_path),
                    os.path.join(data_dir, class_name, image_path),
                ]
                
                # Check for mask existence
                possible_mask_paths = [
                    os.path.join(data_dir, mask_path),
                    os.path.join(data_dir, 'mask', mask_path),
                    os.path.join(data_dir, class_name, 'mask', mask_path),
                    os.path.join(data_dir, class_name, mask_path),
                ]
                
                actual_image_path = None
                actual_mask_path = None
                
                for path in possible_image_paths:
                    if os.path.exists(path):
                        actual_image_path = path
                        break
                        
                for path in possible_mask_paths:
                    if os.path.exists(path):
                        actual_mask_path = path
                        break
                
                if actual_image_path and actual_mask_path:
                    self.samples.append({
                        'image_path': actual_image_path,
                        'mask_path': actual_mask_path,
                        'class': self.class_to_idx[class_name]
                    })
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('L')
        image = self.transform(image)
        
        # Load mask
        mask = Image.open(sample['mask_path']).convert('L')
        mask = self.mask_transform(mask)
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask, sample['class']

class FixedUNet(nn.Module):
    """FIXED U-Net with proper output constraints"""
    
    def __init__(self, in_channels=2, out_channels=2, time_dim=256, num_classes=2):  # 2 channels: image + mask
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
        
        # Output layers
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
        
        output = self.out_conv(d1)
        
        # CRITICAL FIX: Constrain outputs to [-1, 1] for images and [0, 1] for masks
        image_output = torch.tanh(output[:, :1, :, :])  # Image channel: [-1, 1]
        mask_output = torch.sigmoid(output[:, 1:, :, :])  # Mask channel: [0, 1]
        
        return torch.cat([image_output, mask_output], dim=1)
    
    def get_time_embedding(self, timesteps, embedding_dim):
        """Sinusoidal time embeddings"""
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :].to(timesteps.device)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class FixedDiffusion:
    """Fixed diffusion process for image + mask generation"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
    
    def q_sample(self, x_start, t, noise=None):
        """Add noise to image+mask pairs"""
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
        """Generate image+mask samples"""
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

def train_fixed_model(data_dir, num_epochs=50, batch_size=8, lr=1e-4, device='cuda'):
    """Train the FIXED diffusion model"""
    
    # Dataset and dataloader
    dataset = BUSIDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model and diffusion
    model = FixedUNet().to(device)
    diffusion = FixedDiffusion(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Training FIXED model on {len(dataset)} samples")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks, class_labels) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            class_labels = class_labels.to(device)
            
            # Combine image and mask
            x_start = torch.cat([images, masks], dim=1)  # Shape: [B, 2, H, W]
            
            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            
            # Calculate loss
            loss = diffusion.p_losses(model, x_start, t, class_labels)
            
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
            }, f'fixed_diffusion_model_epoch_{epoch+1}.pth')

def generate_fixed_synthetic_data(checkpoint_path, output_dir, num_benign=100, num_malignant=100, device='cuda'):
    """Generate BOTH images AND masks with FIXED normalization"""
    
    # Load model
    model = FixedUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = FixedDiffusion(device=device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each class
    class_counts = {'benign': num_benign, 'malignant': num_malignant}
    
    for class_idx, class_name in enumerate(['benign', 'malignant']):
        num_samples = class_counts[class_name]
        
        if num_samples == 0:
            continue
            
        # Create directories
        image_dir = os.path.join(output_dir, class_name, 'image')
        mask_dir = os.path.join(output_dir, class_name, 'mask')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        print(f'Generating {num_samples} {class_name} samples (images + masks)...')
        
        # Generate in batches
        batch_size = 8
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        sample_count = 0
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - sample_count)
            
            # Class labels
            class_labels = torch.full((current_batch_size,), class_idx, device=device)
            
            # Generate samples (2 channels: image + mask)
            samples = diffusion.sample(
                model, 
                (current_batch_size, 2, 256, 256), 
                class_labels, 
                device
            )
            
            # Save images and masks
            for i in range(current_batch_size):
                # Extract image and mask
                image_tensor = samples[i, 0:1, :, :]  # First channel
                mask_tensor = samples[i, 1:2, :, :]   # Second channel
                
                # Convert image (from [-1, 1] to [0, 255])
                image_array = ((image_tensor.cpu().numpy()[0] + 1) * 127.5).astype(np.uint8)
                image_pil = Image.fromarray(image_array, mode='L')
                
                # Convert mask (from [0, 1] to [0, 255])
                mask_array = (mask_tensor.cpu().numpy()[0] * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_array, mode='L')
                
                # Save files
                image_filename = f'synthetic_{class_name}_{sample_count:04d}.png'
                mask_filename = f'synthetic_{class_name}_{sample_count:04d}_mask.png'
                
                image_pil.save(os.path.join(image_dir, image_filename))
                mask_pil.save(os.path.join(mask_dir, mask_filename))
                
                sample_count += 1
        
        print(f'Saved {num_samples} {class_name} samples (images + masks)')
    
    print(f'Total generated: {num_benign} benign + {num_malignant} malignant = {num_benign + num_malignant} image-mask pairs')

def main():
    parser = argparse.ArgumentParser(description='FIXED Diffusion for BUSI')
    parser.add_argument('--data_dir', type=str, help='Path to BUSI dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for generation')
    parser.add_argument('--output_dir', type=str, default='./fixed_synthetic_busi', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_benign', type=int, default=175, help='Number of benign samples to generate')
    parser.add_argument('--num_malignant', type=int, default=89, help='Number of malignant samples to generate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.mode == 'train':
        if not args.data_dir:
            raise ValueError('data_dir required for training')
        train_fixed_model(args.data_dir, args.num_epochs, args.batch_size, device=str(device))
    
    elif args.mode == 'generate':
        if not args.checkpoint:
            raise ValueError('checkpoint required for generation')
        generate_fixed_synthetic_data(args.checkpoint, args.output_dir, args.num_benign, args.num_malignant, str(device))

if __name__ == '__main__':
    main() 