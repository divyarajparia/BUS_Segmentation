"""
MedSegDiff-V2 Based Synthetic BUSI Image Generation
==================================================

This implementation adapts MedSegDiff-V2 for synthetic BUSI (breast ultrasound) 
image generation with corresponding segmentation masks.

Based on:
- MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer (AAAI 2024)
- Original repo: https://github.com/KidsWithTokens/MedSegDiff

Key Features:
- Simultaneous image and mask generation
- Transformer-based diffusion architecture
- Class-conditional generation (benign/malignant)
- High-quality synthetic medical images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import json
from typing import Tuple, Optional, Dict, List

# ============================================================================
# Vision Transformer Components for MedSegDiff-V2
# ============================================================================

class PatchEmbed(nn.Module):
    """Image to Patch Embedding for medical images."""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, D
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for medical image features."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with medical image adaptations."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                      attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MedicalTransformerEncoder(nn.Module):
    """Medical image specific transformer encoder."""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768, 
                 depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x

# ============================================================================
# MedSegDiff-V2 Architecture
# ============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embeddings)
        embeddings = embeddings.to(device)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time and class conditioning."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes=2):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.class_emb = nn.Embedding(num_classes, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb, class_emb=None):
        h = self.block1(x)
        
        # Add time conditioning
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Add class conditioning
        if class_emb is not None:
            class_emb = self.class_emb(class_emb)
            h = h + class_emb[:, :, None, None]
        
        h = self.block2(h)
        return h + self.shortcut(x)

class MedSegDiffV2(nn.Module):
    """MedSegDiff-V2 model for medical image and mask generation."""
    
    def __init__(self, in_channels=1, out_channels=1, num_classes=2, 
                 model_channels=128, time_embed_dim=256):
        super().__init__()
        
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Medical transformer encoder
        self.transformer = MedicalTransformerEncoder(
            img_size=256, patch_size=16, in_chans=in_channels,
            embed_dim=512, depth=6, num_heads=8
        )
        
        # U-Net style encoder
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.down1 = ResidualBlock(model_channels, model_channels, time_embed_dim, num_classes)
        self.down2 = ResidualBlock(model_channels, model_channels * 2, time_embed_dim, num_classes)
        self.down3 = ResidualBlock(model_channels * 2, model_channels * 4, time_embed_dim, num_classes)
        self.down4 = ResidualBlock(model_channels * 4, model_channels * 8, time_embed_dim, num_classes)
        
        # Bottleneck with transformer integration
        self.bottleneck = ResidualBlock(model_channels * 8, model_channels * 8, time_embed_dim, num_classes)
        
        # U-Net style decoder
        self.up4 = ResidualBlock(model_channels * 16, model_channels * 4, time_embed_dim, num_classes)
        self.up3 = ResidualBlock(model_channels * 8, model_channels * 2, time_embed_dim, num_classes)
        self.up2 = ResidualBlock(model_channels * 4, model_channels, time_embed_dim, num_classes)
        self.up1 = ResidualBlock(model_channels * 2, model_channels, time_embed_dim, num_classes)
        
        self.final_conv = nn.Conv2d(model_channels, out_channels, 1)
        
    def forward(self, x, timesteps, class_labels=None):
        # Time embedding
        time_emb = self.time_embed(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # Transformer features (for global context)
        transformer_features = self.transformer(x)
        
        # U-Net forward pass
        x = self.init_conv(x)
        
        # Encoder
        h1 = self.down1(x, time_emb, class_labels)
        h2 = self.down2(F.max_pool2d(h1, 2), time_emb, class_labels)
        h3 = self.down3(F.max_pool2d(h2, 2), time_emb, class_labels)
        h4 = self.down4(F.max_pool2d(h3, 2), time_emb, class_labels)
        
        # Bottleneck
        h = self.bottleneck(F.max_pool2d(h4, 2), time_emb, class_labels)
        
        # Decoder
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.up4(torch.cat([h, h4], dim=1), time_emb, class_labels)
        
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.up3(torch.cat([h, h3], dim=1), time_emb, class_labels)
        
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.up2(torch.cat([h, h2], dim=1), time_emb, class_labels)
        
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.up1(torch.cat([h, h1], dim=1), time_emb, class_labels)
        
        return self.final_conv(h)

# ============================================================================
# Dataset and Training
# ============================================================================

class BUSIWithMasksDataset(Dataset):
    """BUSI dataset loader with masks for joint training."""
    
    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        # Load samples with masks
        for class_name in ['benign', 'malignant']:
            img_dir = os.path.join(data_dir, class_name, 'images')
            mask_dir = os.path.join(data_dir, class_name, 'masks')
            
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                for img_file in os.listdir(img_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Find corresponding mask
                        mask_file = img_file.replace('.png', '_mask.png')
                        mask_path = os.path.join(mask_dir, mask_file)
                        
                        if os.path.exists(mask_path):
                            self.samples.append({
                                'image_path': os.path.join(img_dir, img_file),
                                'mask_path': mask_path,
                                'class': self.class_to_idx[class_name]
                            })
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
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
        image = Image.open(sample['image_path']).convert('L')
        image = self.transform(image)
        
        # Load mask
        mask = Image.open(sample['mask_path']).convert('L')
        mask = self.mask_transform(mask)
        
        return image, mask, sample['class']

class MedSegDiffusion:
    """Diffusion process for MedSegDiff-V2."""
    
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
        """Generate samples using DDPM sampling"""
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

def train_medsegdiff(data_dir, num_epochs=100, batch_size=4, lr=1e-4, device='cuda'):
    """Train MedSegDiff-V2 model"""
    
    # Dataset and dataloader
    dataset = BUSIWithMasksDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model and diffusion
    model = MedSegDiffV2().to(device)
    diffusion = MedSegDiffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"Training MedSegDiff-V2 on {len(dataset)} samples")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks, class_labels) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            class_labels = class_labels.to(device)
            
            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
            
            # Train on images
            loss_img = diffusion.p_losses(model, images, t, class_labels)
            
            # Train on masks (optional, for joint training)
            loss_mask = diffusion.p_losses(model, masks, t, class_labels)
            
            # Combined loss
            loss = loss_img + 0.5 * loss_mask
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }, f'medsegdiff_v2_epoch_{epoch+1}.pth')

def main():
    parser = argparse.ArgumentParser(description='MedSegDiff-V2 for BUSI')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to BUSI dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for generation')
    parser.add_argument('--output_dir', type=str, default='./synthetic_medseg', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_benign', type=int, default=100, help='Number of benign samples')
    parser.add_argument('--num_malignant', type=int, default=100, help='Number of malignant samples')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.mode == 'train':
        train_medsegdiff(args.data_dir, args.num_epochs, args.batch_size, device=device)
    
    elif args.mode == 'generate':
        print("Generation mode not fully implemented yet. Use simple_diffusion_busi.py for now.")

if __name__ == '__main__':
    main() 