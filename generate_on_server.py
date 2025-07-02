#!/usr/bin/env python3
"""
Generate Synthetic Dataset ON SERVER
====================================
Run this script on the server where you trained the model.
This avoids the need to copy large checkpoint files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Model classes (same as training)
class JointDiffusionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, time_dim=256, num_classes=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)
        self.enc1 = self._make_conv_block(in_channels, 64)
        self.enc2 = self._make_conv_block(64, 128)
        self.enc3 = self._make_conv_block(128, 256)
        self.enc4 = self._make_conv_block(256, 512)
        self.bottleneck = self._make_conv_block(512, 1024)
        self.cond_proj = nn.Linear(time_dim * 2, 1024)
        self.dec4 = self._make_conv_block(1024 + 512, 512)
        self.dec3 = self._make_conv_block(512 + 256, 256)
        self.dec2 = self._make_conv_block(256 + 128, 128)
        self.dec1 = self._make_conv_block(128 + 64, 64)
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
        t_emb = self.positional_encoding(timesteps)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(class_labels)
        cond = torch.cat([t_emb, c_emb], dim=1)
        
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        cond_proj = self.cond_proj(cond)
        b = b + cond_proj.view(-1, 1024, 1, 1)
        
        d4 = self.dec4(torch.cat([F.interpolate(b, scale_factor=2), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        
        output = self.final(d1)
        return output

class JointDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
    
    @torch.no_grad()
    def sample(self, model, shape, class_labels=None, device='cuda'):
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Generating'):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            predicted_noise = model(x, t, class_labels)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(beta_t) * noise
            else:
                x = pred_x0
        
        return x

def robust_denormalize_image(tensor):
    """Robust denormalization - handles any tensor range"""
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    # Min-max normalization to [0,1]
    if max_val > min_val:
        tensor = (tensor - min_val) / (max_val - min_val)
    else:
        tensor = torch.zeros_like(tensor)
    
    # Convert to [0,255]
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return array

def robust_denormalize_mask(tensor):
    """Robust mask denormalization"""
    # Use sigmoid + threshold
    tensor = torch.sigmoid(tensor)
    tensor = (tensor > 0.5).float()
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return array

def generate_on_server():
    print("🎨 GENERATING ON SERVER")
    print("=" * 40)
    
    # Find checkpoint
    checkpoint_path = "joint_diffusion_epoch_50.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "joint_diffusion_final.pth"
        if not os.path.exists(checkpoint_path):
            print("❌ No checkpoint found!")
            return
    
    print(f"📂 Loading: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = JointDiffusionUNet(in_channels=2, out_channels=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = JointDiffusion(device=device)
    
    print(f"✅ Model loaded! Loss: {checkpoint['loss']:.4f}")
    
    # TEST GENERATION - Just 3 samples first
    print(f"\n🧪 TEST: Generating 3 samples...")
    
    output_dir = "server_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(3):
        print(f"   Sample {i+1}/3...")
        
        # Alternate between benign (0) and malignant (1)
        class_label = torch.tensor([i % 2], dtype=torch.long, device=device)
        
        generated = diffusion.sample(
            model, 
            (1, 2, 64, 64), 
            class_labels=class_label, 
            device=device
        )
        
        # Extract channels
        img_tensor = generated[0, 0:1]
        mask_tensor = generated[0, 1:2]
        
        # Denormalize
        img_array = robust_denormalize_image(img_tensor)[0]
        mask_array = robust_denormalize_mask(mask_tensor)[0]
        
        # Save
        class_name = "benign" if i % 2 == 0 else "malignant"
        img_path = os.path.join(output_dir, f"test_{class_name}_{i+1}_img.png")
        mask_path = os.path.join(output_dir, f"test_{class_name}_{i+1}_mask.png")
        
        Image.fromarray(img_array, mode='L').save(img_path)
        Image.fromarray(mask_array, mode='L').save(mask_path)
        
        print(f"   ✅ Saved: {img_path}")
        print(f"      Image range: [{img_array.min()}, {img_array.max()}]")
        print(f"      Mask range: [{mask_array.min()}, {mask_array.max()}]")
    
    print(f"\n🎉 TEST COMPLETE!")
    print(f"   Check images in: {output_dir}/")
    print(f"   If they look good, run full generation!")

if __name__ == "__main__":
    generate_on_server() 