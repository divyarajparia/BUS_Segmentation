#!/usr/bin/env python3
"""
FINAL: Joint Image+Mask Diffusion Training for Server
====================================================
Complete script for training joint image+mask diffusion model on full BUSI dataset.
Run this on your server where the full dataset is available.

Usage:
    python FINAL_train_joint_diffusion_server.py

Output:
    - joint_diffusion_epoch_50.pth (final trained model)
    - Supports generation of 264 high-quality image+mask pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

class BUSIJointDataset(Dataset):
    """BUSI dataset loader for joint image+mask training"""
    
    def __init__(self, csv_path, data_dir, img_size=64):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []
        
        # Load CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"📊 Loading from CSV: {len(df)} samples")
        else:
            print(f"❌ CSV not found: {csv_path}")
            return
        
        # Process each sample
        for _, row in df.iterrows():
            img_path = str(row['image_path'])
            mask_path = str(row['mask_path'])
            
            # Extract class from filename
            class_name = 'benign' if 'benign' in img_path.lower() else 'malignant'
            
            # Try different path combinations for server dataset
            possible_img_paths = [
                os.path.join(data_dir, img_path),
                os.path.join(data_dir, class_name, img_path),
                os.path.join(data_dir, class_name, 'image', img_path),
                os.path.join(data_dir, 'images', img_path),
                os.path.join(data_dir, 'image', img_path)
            ]
            
            possible_mask_paths = [
                os.path.join(data_dir, mask_path),
                os.path.join(data_dir, class_name, mask_path),
                os.path.join(data_dir, class_name, 'mask', mask_path),
                os.path.join(data_dir, 'masks', mask_path),
                os.path.join(data_dir, 'mask', mask_path)
            ]
            
            # Find valid paths
            valid_img_path = None
            valid_mask_path = None
            
            for img_path_candidate in possible_img_paths:
                if os.path.exists(img_path_candidate):
                    valid_img_path = img_path_candidate
                    break
            
            for mask_path_candidate in possible_mask_paths:
                if os.path.exists(mask_path_candidate):
                    valid_mask_path = mask_path_candidate
                    break
            
            if valid_img_path and valid_mask_path:
                self.samples.append({
                    'image_path': valid_img_path,
                    'mask_path': valid_mask_path,
                    'class': 0 if class_name == 'benign' else 1
                })
        
        print(f"✅ Successfully loaded {len(self.samples)} valid samples")
        
        if len(self.samples) > 0:
            # Check class distribution
            benign_count = sum(1 for s in self.samples if s['class'] == 0)
            malignant_count = len(self.samples) - benign_count
            print(f"   Benign: {benign_count} samples ({benign_count/len(self.samples)*100:.1f}%)")
            print(f"   Malignant: {malignant_count} samples ({malignant_count/len(self.samples)*100:.1f}%)")
        
        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('L')
            image = self.img_transform(image)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            image = torch.zeros(1, self.img_size, self.img_size)
        
        # Load mask
        try:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = self.mask_transform(mask)
            # Binarize mask
            mask = (mask > 0.5).float()
        except Exception as e:
            print(f"Error loading mask {sample['mask_path']}: {e}")
            mask = torch.zeros(1, self.img_size, self.img_size)
        
        return image, mask, sample['class']

class JointDiffusionUNet(nn.Module):
    """U-Net architecture for joint image+mask diffusion"""
    
    def __init__(self, in_channels=2, out_channels=2, time_dim=256, num_classes=2):
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
        self.enc1 = self._make_conv_block(in_channels, 64)
        self.enc2 = self._make_conv_block(64, 128)
        self.enc3 = self._make_conv_block(128, 256)
        self.enc4 = self._make_conv_block(256, 512)
        
        # Bottleneck with conditioning
        self.bottleneck = self._make_conv_block(512, 1024)
        self.cond_proj = nn.Linear(time_dim * 2, 1024)  # time + class
        
        # Decoder
        self.dec4 = self._make_conv_block(1024 + 512, 512)
        self.dec3 = self._make_conv_block(512 + 256, 256)
        self.dec2 = self._make_conv_block(256 + 128, 128)
        self.dec1 = self._make_conv_block(128 + 64, 64)
        
        # Output heads
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU()
        )
    
    def positional_encoding(self, timesteps, dim=256):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, timesteps, class_labels):
        # Time and class conditioning
        t_emb = self.positional_encoding(timesteps)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(class_labels)
        
        # Combine conditioning
        cond = torch.cat([t_emb, c_emb], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck with conditioning
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Apply conditioning
        cond_proj = self.cond_proj(cond)
        b = b + cond_proj.view(-1, 1024, 1, 1)
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([F.interpolate(b, scale_factor=2), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        
        # Final output
        output = self.final(d1)
        
        return output

class JointDiffusion:
    """Joint diffusion process for images and masks"""
    
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
        """Add noise to image+mask pairs"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_start, t, class_labels=None):
        """Training loss for joint generation"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, class_labels)
        
        # Separate losses for image and mask channels
        img_loss = F.mse_loss(noise[:, :1], predicted_noise[:, :1])
        mask_loss = F.mse_loss(noise[:, 1:], predicted_noise[:, 1:])
        
        # Combined loss with weighting
        return img_loss + 0.5 * mask_loss
    
    @torch.no_grad()
    def sample(self, model, shape, class_labels=None, device='cuda'):
        """Generate joint image+mask samples"""
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Generating'):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t, class_labels)
            
            # DDPM sampling step
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            # Compute x_{t-1}
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(beta_t) * noise
            else:
                x = pred_x0
        
        return x

def train_joint_diffusion():
    """Train the joint diffusion model"""
    
    print("🏥 FINAL Joint Image+Mask Diffusion Training")
    print("=" * 70)
    
    # Dataset paths - adjust these for your server
    data_dir = "dataset/BioMedicalDataset/BUSI"  # Adjust this path
    train_csv = os.path.join(data_dir, "train_frame.csv")
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found: {data_dir}")
        print(f"   Please adjust the data_dir path for your server")
        return
    
    if not os.path.exists(train_csv):
        print(f"❌ Training CSV not found: {train_csv}")
        return
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Dataset and dataloader
    dataset = BUSIJointDataset(train_csv, data_dir, img_size=64)
    
    if len(dataset) == 0:
        print(f"❌ No valid samples found in dataset")
        print(f"   Check that image and mask files exist in: {data_dir}")
        return
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    
    # Model setup
    model = JointDiffusionUNet(in_channels=2, out_channels=2).to(device)
    diffusion = JointDiffusion()
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    num_epochs = 50
    save_interval = 10
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Training batches per epoch: {len(dataloader)}")
    
    # Training loop
    print(f"\n🚀 Starting Training...")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks, class_labels) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            class_labels = class_labels.to(device)
            
            # Combine image and mask for joint training
            x_start = torch.cat([images, masks], dim=1)  # Shape: [B, 2, H, W]
            
            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)
            
            # Forward pass
            loss = diffusion.p_losses(model, x_start, t, class_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'num_samples': len(dataset),
                'model_config': {
                    'in_channels': 2,
                    'out_channels': 2,
                    'time_dim': 256,
                    'num_classes': 2
                }
            }
            checkpoint_path = f'joint_diffusion_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f'✅ Saved checkpoint: {checkpoint_path}')
            
            # Also save as 'latest' for easy access
            if epoch == num_epochs - 1:
                latest_path = 'joint_diffusion_final.pth'
                torch.save(checkpoint, latest_path)
                print(f'✅ Final model saved as: {latest_path}')
    
    print(f"\n🎉 Training Complete!")
    print(f"   Final model: joint_diffusion_epoch_{num_epochs}.pth")
    print(f"   Trained on {len(dataset)} samples from full BUSI dataset")
    print(f"   Model supports joint image+mask generation")
    print(f"   Ready for high-quality synthetic data generation!")
    print(f"\n📋 Next Steps:")
    print(f"   1. Copy the trained checkpoint back to local machine")
    print(f"   2. Use for generating 264 synthetic image+mask pairs")
    print(f"   3. Compare with style transfer results")

if __name__ == "__main__":
    train_joint_diffusion() 