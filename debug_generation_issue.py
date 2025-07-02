#!/usr/bin/env python3
"""
Debug Generation Issue - Why Are Images Black?
==============================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Copy the model classes
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

def debug_generation():
    print("🔍 DEBUGGING BLACK IMAGE GENERATION")
    print("=" * 50)

    # Check checkpoint
    checkpoint_path = "joint_diffusion_epoch_50.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "joint_diffusion_final.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Found checkpoint: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Loss: {checkpoint['loss']:.6f}")
        
        # Check if loss is suspiciously low (model collapsed)
        if checkpoint['loss'] < 0.001:
            print(f"⚠️  VERY LOW LOSS - Model may have collapsed!")
        
        # Check model weights
        state_dict = checkpoint['model_state_dict']
        
        # Check a few key layers
        for key, tensor in state_dict.items():
            if 'final.weight' in key:  # Final output layer
                mean_val = tensor.mean().item()
                std_val = tensor.std().item()
                print(f"   Final layer: mean={mean_val:.6f}, std={std_val:.6f}")
                if abs(mean_val) < 1e-6 and std_val < 1e-6:
                    print(f"   ❌ FINAL LAYER IS ZERO! This will cause black images!")
                break
    else:
        print("❌ No checkpoint found!")

if __name__ == "__main__":
    debug_generation() 