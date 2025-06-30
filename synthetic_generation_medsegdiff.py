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

# ============================================================================
# Diffusion Model Components
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    """Build sinusoidal embeddings for timesteps."""
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class ResBlock(nn.Module):
    """Residual block with time and class conditioning."""
    
    def __init__(self, channels, time_emb_dim, num_classes=2, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, channels)
        self.class_emb = nn.Embedding(num_classes, channels)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb, class_label=None):
        h = self.conv1(F.relu(self.norm1(x)))
        
        # Add time embedding
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Add class embedding if provided
        if class_label is not None:
            class_emb = self.class_emb(class_label)
            h = h + class_emb[:, :, None, None]
        
        h = self.conv2(F.relu(self.norm2(self.dropout(h))))
        return h + x

class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""
    
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        attn = torch.bmm(q, k) * (c ** -0.5)  # b, hw, hw
        attn = F.softmax(attn, dim=-1)
        
        v = v.reshape(b, c, h * w)  # b, c, hw
        h = torch.bmm(v, attn.permute(0, 2, 1))  # b, c, hw
        h = h.reshape(b, c, h, w)
        
        return x + self.proj_out(h)

# ============================================================================
# MedSegDiff-V2 Main Architecture
# ============================================================================

class MedSegDiffV2(nn.Module):
    """
    MedSegDiff-V2: Diffusion-based Medical Image Segmentation with Transformer
    
    Generates both medical images and corresponding segmentation masks simultaneously.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = 2,  # benign, malignant
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.1,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        use_transformer: bool = True,
        transformer_depth: int = 6,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.model_channels = model_channels
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Class embedding for conditional generation
        self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Transformer components (if enabled)
        self.use_transformer = use_transformer
        if use_transformer:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=16, 
                                        in_chans=model_channels, embed_dim=model_channels)
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(model_channels, num_heads, dropout=dropout)
                for _ in range(transformer_depth)
            ])
            self.transformer_norm = nn.LayerNorm(model_channels)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, num_classes, dropout)]
                if ch in [model_channels * m for m in attention_resolutions]:
                    layers.append(AttentionBlock(ch))
                self.down_blocks.append(nn.ModuleList(layers))
            
            if level != len(channel_mult) - 1:  # No downsampling on last level
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]))
                ch *= 2
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResBlock(ch, time_embed_dim, num_classes, dropout),
            AttentionBlock(ch),
            ResBlock(ch, time_embed_dim, num_classes, dropout),
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch, time_embed_dim, num_classes, dropout)]
                if ch in [model_channels * m for m in attention_resolutions]:
                    layers.append(AttentionBlock(ch))
                self.up_blocks.append(nn.ModuleList(layers))
                
                if i == num_res_blocks and level != 0:
                    ch //= 2
            
            if level != 0:
                self.up_blocks.append(nn.ModuleList([nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)]))
        
        # Output layers
        self.out_norm = nn.GroupNorm(8, model_channels)
        self.out_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
        # Separate head for mask generation
        self.mask_head = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, class_labels=None, return_mask=False):
        """
        Forward pass for MedSegDiff-V2
        
        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            class_labels: Class labels for conditional generation [B]
            return_mask: Whether to return segmentation mask
        """
        # Time embedding
        time_emb = get_timestep_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(time_emb)
        
        # Add class conditioning if provided
        if class_labels is not None:
            class_emb = self.class_embed(class_labels)
            time_emb = time_emb + class_emb
        
        # Initial convolution
        h = self.input_conv(x)
        
        # Apply transformer if enabled
        if self.use_transformer:
            # Save spatial features for later
            spatial_h = h
            
            # Apply transformer on patches
            h_patches = self.patch_embed(h)
            for block in self.transformer_blocks:
                h_patches = block(h_patches)
            h_patches = self.transformer_norm(h_patches)
            
            # Reshape back to spatial
            B, N, D = h_patches.shape
            grid_size = int(N ** 0.5)
            h_transformer = h_patches.transpose(1, 2).reshape(B, D, grid_size, grid_size)
            h_transformer = F.interpolate(h_transformer, size=h.shape[2:], mode='bilinear', align_corners=False)
            
            # Combine spatial and transformer features
            h = h + h_transformer
        
        # Downsampling
        skip_connections = [h]
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb, class_labels)
                else:
                    h = layer(h)
            skip_connections.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb, class_labels)
            else:
                h = layer(h)
        
        # Upsampling
        for block in self.up_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    if skip_connections:
                        skip = skip_connections.pop()
                        h = torch.cat([h, skip], dim=1)
                    h = layer(h, time_emb, class_labels)
                else:
                    h = layer(h)
        
        # Output
        h = F.silu(self.out_norm(h))
        image_out = self.out_conv(h)
        
        if return_mask:
            mask_out = torch.sigmoid(self.mask_head(h))
            return image_out, mask_out
        
        return image_out

# ============================================================================
# Diffusion Training and Sampling
# ============================================================================

class DiffusionProcess:
    """Handles the diffusion process for training and sampling."""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, t, class_labels=None, noise=None):
        """Calculate training losses."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t, class_labels)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, class_labels=None):
        """Sample from p(x_{t-1} | x_t)."""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = model(x, t, class_labels)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, class_labels=None, return_all_timesteps=False):
        """Generate samples from the model."""
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        xs = [x] if return_all_timesteps else None
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, class_labels)
            if return_all_timesteps:
                xs.append(x)
        
        if return_all_timesteps:
            return torch.stack(xs, dim=1)
        return x

    def sample(self, model, batch_size, img_size, num_classes=2, class_labels=None):
        """Generate synthetic medical images."""
        if class_labels is None:
            class_labels = torch.randint(0, num_classes, (batch_size,))
        
        shape = (batch_size, 1, img_size, img_size)
        return self.p_sample_loop(model, shape, class_labels)

# ============================================================================
# BUSI Dataset Loader
# ============================================================================

class BUSIDataset(Dataset):
    """Dataset loader for BUSI breast ultrasound images."""
    
    def __init__(self, data_dir, split='train', img_size=256, augment=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Load image and mask paths
        self.samples = self._load_samples(split)
        
        # Class mapping
        self.class_to_idx = {'benign': 0, 'malignant': 1}

    def _load_samples(self, split):
        """Load image and mask file paths."""
        samples = []
        
        # Look for different folder structures
        for class_name in ['benign', 'malignant']:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            img_dir = os.path.join(class_path, 'images')
            mask_dir = os.path.join(class_path, 'masks')
            
            if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
                continue
            
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Find corresponding mask
                    mask_file = img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
                    if not mask_file.endswith('_mask.png'):
                        mask_file = img_file.replace('.png', '_mask.png')
                    
                    img_path = os.path.join(img_dir, img_file)
                    mask_path = os.path.join(mask_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        samples.append({
                            'image_path': img_path,
                            'mask_path': mask_path,
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name]
                        })
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and mask
        image = Image.open(sample['image_path']).convert('L')
        mask = Image.open(sample['mask_path']).convert('L')
        
        # Apply transforms
        image = self.transform(image)
        mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        mask = mask_transform(mask)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'class_idx': sample['class_idx'],
            'class_name': sample['class']
        }

# ============================================================================
# Training Functions
# ============================================================================

def train_medsegdiff(
    model,
    dataloader,
    diffusion,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
    save_dir='./checkpoints',
    save_interval=10
):
    """Train MedSegDiff-V2 model."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            optimizer.zero_grad()
            
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            class_labels = batch['class_idx'].to(device)
            
            # Random timesteps
            batch_size = images.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device).long()
            
            # Calculate loss for both image and mask generation
            img_loss = diffusion.p_losses(model, images, t, class_labels)
            
            # For mask generation, we can use a separate head or joint training
            # Here we'll focus on image generation first
            loss = img_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{total_loss/num_batches:.4f}'})
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'medsegdiff_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss / num_batches,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_busi(
    model,
    diffusion,
    num_samples_per_class=500,
    output_dir='./synthetic_busi',
    img_size=256,
    device='cuda'
):
    """Generate synthetic BUSI images and masks."""
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directories
    for class_name in ['benign', 'malignant']:
        os.makedirs(os.path.join(output_dir, class_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, class_name, 'masks'), exist_ok=True)
    
    class_names = ['benign', 'malignant']
    
    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            print(f'Generating {num_samples_per_class} synthetic {class_name} samples...')
            
            # Generate in batches
            batch_size = 8
            num_batches = (num_samples_per_class + batch_size - 1) // batch_size
            
            sample_count = 0
            for batch_idx in tqdm(range(num_batches), desc=f'Generating {class_name}'):
                current_batch_size = min(batch_size, num_samples_per_class - sample_count)
                
                # Generate class labels
                class_labels = torch.full((current_batch_size,), class_idx, device=device)
                
                # Generate synthetic images
                synthetic_images = diffusion.sample(
                    model, 
                    current_batch_size, 
                    img_size, 
                    num_classes=2,
                    class_labels=class_labels
                )
                
                # Generate corresponding masks using the mask head
                model_out, synthetic_masks = model(
                    synthetic_images, 
                    torch.zeros(current_batch_size, device=device).long(),
                    class_labels,
                    return_mask=True
                )
                
                # Save images and masks
                for i in range(current_batch_size):
                    # Convert to PIL images
                    img_array = ((synthetic_images[i].cpu().numpy()[0] + 1) * 127.5).astype(np.uint8)
                    mask_array = (synthetic_masks[i].cpu().numpy()[0] * 255).astype(np.uint8)
                    
                    img_pil = Image.fromarray(img_array, mode='L')
                    mask_pil = Image.fromarray(mask_array, mode='L')
                    
                    # Save files
                    img_filename = f'synthetic_{class_name}_{sample_count:04d}.png'
                    mask_filename = f'synthetic_{class_name}_{sample_count:04d}_mask.png'
                    
                    img_path = os.path.join(output_dir, class_name, 'images', img_filename)
                    mask_path = os.path.join(output_dir, class_name, 'masks', mask_filename)
                    
                    img_pil.save(img_path)
                    mask_pil.save(mask_path)
                    
                    sample_count += 1
    
    print(f'Generated {num_samples_per_class} synthetic samples per class in {output_dir}')

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MedSegDiff-V2 for BUSI synthetic generation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to BUSI dataset')
    parser.add_argument('--output_dir', type=str, default='./synthetic_busi', help='Output directory for synthetic data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_synthetic_per_class', type=int, default=500, help='Number of synthetic samples per class')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='train', help='Mode: train or generate')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for generation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model and diffusion process
    model = MedSegDiffV2(
        img_size=args.img_size,
        in_channels=1,
        out_channels=1,
        num_classes=2,
        model_channels=128,
        use_transformer=True
    ).to(device)
    
    diffusion = DiffusionProcess(num_timesteps=1000)
    
    if args.mode == 'train':
        # Load dataset
        dataset = BUSIDataset(args.data_dir, img_size=args.img_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        print(f'Training on {len(dataset)} samples')
        
        # Train model
        train_medsegdiff(
            model=model,
            dataloader=dataloader,
            diffusion=diffusion,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device,
            save_dir=args.checkpoint_dir
        )
        
    elif args.mode == 'generate':
        if not args.checkpoint_path:
            raise ValueError('Checkpoint path required for generation mode')
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {args.checkpoint_path}')
        
        # Generate synthetic data
        generate_synthetic_busi(
            model=model,
            diffusion=diffusion,
            num_samples_per_class=args.num_synthetic_per_class,
            output_dir=args.output_dir,
            img_size=args.img_size,
            device=device
        )

if __name__ == '__main__':
    main() 