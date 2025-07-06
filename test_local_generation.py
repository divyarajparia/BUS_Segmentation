#!/usr/bin/env python3
"""
Test Local Generation - Debug Blank Images
==========================================
Use the joint_diffusion_fnal.pth checkpoint to test generation locally
and debug why images were blank on the server.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Model classes (copied from the working scripts)
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


class JointDiffusion:
    """Joint diffusion process for images and masks"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule - MOVE TO DEVICE
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        
        # For sampling - MOVE TO DEVICE
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
    
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


def robust_denormalize_image(tensor):
    """Convert tensor to image array with robust normalization"""
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = tensor
    
    print(f"   Raw tensor stats: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}, std={img.std():.4f}")
    
    # Method 1: Standard [-1, 1] to [0, 255] normalization
    if img.min() >= -2.0 and img.max() <= 2.0:
        img_norm = np.clip((img + 1) * 127.5, 0, 255)
        print(f"   Standard norm: min={img_norm.min()}, max={img_norm.max()}")
        if img_norm.max() > img_norm.min():
            return img_norm.astype(np.uint8)
    
    # Method 2: Min-max normalization
    if img.max() != img.min():
        img_norm = (img - img.min()) / (img.max() - img.min()) * 255
        print(f"   Min-max norm: min={img_norm.min()}, max={img_norm.max()}")
        return img_norm.astype(np.uint8)
    
    # Method 3: All values are same - return gray image
    print(f"   WARNING: Flat image detected, returning gray")
    return np.full_like(img, 128, dtype=np.uint8)


def robust_denormalize_mask(tensor):
    """Convert tensor to mask array"""
    if isinstance(tensor, torch.Tensor):
        mask = tensor.detach().cpu().numpy()
    else:
        mask = tensor
    
    print(f"   Raw mask stats: min={mask.min():.4f}, max={mask.max():.4f}")
    
    # Apply sigmoid to get probabilities, then threshold
    mask_prob = 1 / (1 + np.exp(-mask))  # Sigmoid
    mask_binary = (mask_prob > 0.5) * 255
    
    print(f"   Mask binary: min={mask_binary.min()}, max={mask_binary.max()}, nonzero={np.count_nonzero(mask_binary)}")
    
    return mask_binary.astype(np.uint8)


def test_local_generation():
    """Test generation locally with detailed debugging"""
    
    print("🧪 LOCAL GENERATION TEST")
    print("=" * 50)
    
    # Check for checkpoint
    checkpoint_path = "joint_diffusion_final.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "joint_diffusion_fnal.pth"
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✅ Checkpoint loaded successfully")
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"   Loss: {checkpoint['loss']:.4f}")
        if 'num_samples' in checkpoint:
            print(f"   Training samples: {checkpoint['num_samples']}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Load model
    try:
        model = JointDiffusionUNet(in_channels=2, out_channels=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Model loaded and set to eval mode")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Setup diffusion
    diffusion = JointDiffusion(device=device)
    print(f"✅ Diffusion process initialized")
    
    # Create output directory
    output_dir = "local_test_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Test generation with detailed logging
    print(f"\n🎨 GENERATING TEST SAMPLES...")
    
    num_samples = 3
    for i in range(num_samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        
        # Alternate between classes
        class_label = i % 2
        class_name = "benign" if class_label == 0 else "malignant"
        class_tensor = torch.tensor([class_label], dtype=torch.long, device=device)
        
        print(f"Class: {class_name} (label={class_label})")
        
        try:
            # Generate with detailed progress
            print("Generating...")
            generated = diffusion.sample(
                model, 
                (1, 2, 64, 64), 
                class_labels=class_tensor, 
                device=device
            )
            
            # Extract channels
            img_tensor = generated[0, 0]  # First channel (image)
            mask_tensor = generated[0, 1]  # Second channel (mask)
            
            print("Converting image...")
            img_array = robust_denormalize_image(img_tensor)
            
            print("Converting mask...")
            mask_array = robust_denormalize_mask(mask_tensor)
            
            # Save files
            img_filename = f"test_{class_name}_{i+1:02d}_img.png"
            mask_filename = f"test_{class_name}_{i+1:02d}_mask.png"
            
            img_path = os.path.join(output_dir, img_filename)
            mask_path = os.path.join(output_dir, mask_filename)
            
            Image.fromarray(img_array, mode='L').save(img_path)
            Image.fromarray(mask_array, mode='L').save(mask_path)
            
            print(f"✅ Saved: {img_filename}")
            print(f"   Image: [{img_array.min()}, {img_array.max()}], unique={len(np.unique(img_array))}")
            print(f"✅ Saved: {mask_filename}")
            print(f"   Mask: [{mask_array.min()}, {mask_array.max()}], unique={len(np.unique(mask_array))}")
            
        except Exception as e:
            print(f"❌ Error generating sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 LOCAL TEST COMPLETE!")
    print(f"   Check images in: {output_dir}/")
    print(f"   If images are blank here too, the issue is with the model.")
    print(f"   If images look good here, the issue was server-specific.")


if __name__ == "__main__":
    test_local_generation() 