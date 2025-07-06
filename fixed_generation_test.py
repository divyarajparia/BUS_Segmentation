#!/usr/bin/env python3
"""
Fixed Generation Test - With Numerical Stability
===============================================
Fix the extreme value explosion in DDPM sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# Same model class
class JointDiffusionUNet(nn.Module):
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
        self.cond_proj = nn.Linear(time_dim * 2, 1024)
        
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
    
    def positional_encoding(self, timesteps, dim):
        """Sinusoidal positional encoding for timesteps"""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # Zero pad if odd dimension
            emb = F.pad(emb, (0, 1))
        return emb
    
    def forward(self, x, timesteps, class_labels=None):
        # Time embedding
        t_emb = self.positional_encoding(timesteps, 256)
        t_emb = self.time_mlp(t_emb)
        
        # Class embedding
        if class_labels is not None:
            c_emb = self.class_emb(class_labels)
            cond = torch.cat([t_emb, c_emb], dim=1)
        else:
            cond = t_emb
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck with conditioning
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Add conditioning
        cond_proj = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        b = b + cond_proj
        
        # Decoder
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


class FixedJointDiffusion:
    """Fixed Joint diffusion with numerical stability"""
    
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
    
    @torch.no_grad()
    def sample(self, model, shape, class_labels=None, device='cuda'):
        """Generate with numerical stability fixes"""
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Generating'):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t, class_labels)
            
            # FIXED DDPM sampling step with stability
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            # Add numerical stability
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            
            # CRITICAL FIX: Add epsilon to prevent division by zero
            epsilon = 1e-8
            sqrt_alpha_cumprod_t = torch.clamp(sqrt_alpha_cumprod_t, min=epsilon)
            
            # Compute x_{t-1} with stability
            pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
            
            # CRITICAL FIX: Clamp pred_x0 to reasonable range
            pred_x0 = torch.clamp(pred_x0, min=-10, max=10)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(beta_t) * noise
            else:
                x = pred_x0
            
            # CRITICAL FIX: Clamp x to prevent explosion
            x = torch.clamp(x, min=-20, max=20)
        
        return x


def robust_denormalize_image(tensor):
    """Convert tensor to image array with robust normalization"""
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = tensor
    
    print(f"   Raw image stats: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")
    
    # Method 1: If roughly in [-1, 1], use standard normalization
    if img.min() >= -2.0 and img.max() <= 2.0:
        img_norm = np.clip((img + 1) * 127.5, 0, 255)
        if img_norm.max() > img_norm.min():
            return img_norm.astype(np.uint8)
    
    # Method 2: Min-max normalization
    if img.max() != img.min():
        img_norm = (img - img.min()) / (img.max() - img.min()) * 255
        return img_norm.astype(np.uint8)
    
    # Method 3: All same values
    return np.full_like(img, 128, dtype=np.uint8)


def robust_denormalize_mask(tensor):
    """Convert tensor to mask array"""
    if isinstance(tensor, torch.Tensor):
        mask = tensor.detach().cpu().numpy()
    else:
        mask = tensor
    
    print(f"   Raw mask stats: min={mask.min():.4f}, max={mask.max():.4f}")
    
    # Simple threshold approach
    if mask.max() != mask.min():
        # Normalize to [0, 1] then threshold
        mask_norm = (mask - mask.min()) / (mask.max() - mask.min())
        mask_binary = (mask_norm > 0.5) * 255
    else:
        mask_binary = np.zeros_like(mask)
    
    return mask_binary.astype(np.uint8)


def test_fixed_generation():
    """Test generation with numerical stability fixes"""
    
    print("🔧 TESTING FIXED GENERATION")
    print("=" * 50)
    
    device = torch.device('cpu')  # Use CPU for consistency
    
    # Load a healthy checkpoint
    checkpoint_path = "joint_diffusion_epoch_30.pth"
    print(f"📂 Loading: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    model = JointDiffusionUNet(in_channels=2, out_channels=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Use fixed diffusion process
    diffusion = FixedJointDiffusion(device=device)
    
    print(f"✅ Model and fixed diffusion loaded")
    
    # Create output directory
    output_dir = "fixed_generation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test generation
    print(f"\n🎨 GENERATING WITH FIXES...")
    
    num_samples = 3
    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        
        # Alternate between classes
        class_label = i % 2
        class_name = "benign" if class_label == 0 else "malignant"
        class_tensor = torch.tensor([class_label], dtype=torch.long, device=device)
        
        print(f"Class: {class_name}")
        
        # Generate
        generated = diffusion.sample(
            model, 
            (1, 2, 64, 64), 
            class_labels=class_tensor, 
            device=device
        )
        
        print(f"Generated tensor stats: min={generated.min():.4f}, max={generated.max():.4f}")
        
        # Extract channels
        img_tensor = generated[0, 0]  # First channel (image)
        mask_tensor = generated[0, 1]  # Second channel (mask)
        
        # Convert to images
        print("Converting image...")
        img_array = robust_denormalize_image(img_tensor)
        
        print("Converting mask...")
        mask_array = robust_denormalize_mask(mask_tensor)
        
        # Save files
        img_filename = f"fixed_{class_name}_{i+1:02d}_img.png"
        mask_filename = f"fixed_{class_name}_{i+1:02d}_mask.png"
        
        img_path = os.path.join(output_dir, img_filename)
        mask_path = os.path.join(output_dir, mask_filename)
        
        Image.fromarray(img_array, mode='L').save(img_path)
        Image.fromarray(mask_array, mode='L').save(mask_path)
        
        print(f"✅ Saved: {img_filename}")
        print(f"   Image range: [{img_array.min()}, {img_array.max()}], unique={len(np.unique(img_array))}")
        print(f"✅ Saved: {mask_filename}")
        print(f"   Mask range: [{mask_array.min()}, {mask_array.max()}], unique={len(np.unique(mask_array))}")
    
    print(f"\n🎉 FIXED GENERATION COMPLETE!")
    print(f"   Check images in: {output_dir}/")
    print(f"   If these look good, the issue was numerical instability in DDPM sampling!")


if __name__ == "__main__":
    test_fixed_generation() 