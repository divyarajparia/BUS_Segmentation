"""
CORRECTED Diffusion Model Generation Script
==========================================
Based on successful local debugging, this uses the working generation pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Class embedding (benign=0, malignant=1)
        self.class_emb = nn.Embedding(2, 256)
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1)
        )
        
        # Bottleneck with conditioning
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1)
        )
        
        # Conditioning projection
        self.cond_proj = nn.Linear(512, 512)  # time + class -> 512
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1)
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1)
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
    
    def positional_encoding(self, timesteps, dim=256):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, timesteps, class_labels):
        # Time embedding
        t_emb = self.positional_encoding(timesteps)
        t_emb = self.time_mlp(t_emb)
        
        # Class embedding
        c_emb = self.class_emb(class_labels)
        
        # Combine conditioning
        cond = t_emb + c_emb
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck with conditioning
        b = self.bottleneck(e3)
        
        # Apply conditioning
        cond_proj = self.cond_proj(cond)
        b_shaped = b + cond_proj.view(-1, 512, 1, 1)
        
        # Decoder
        d3 = self.dec3(b_shaped) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        
        return d1

def robust_denormalize(tensor):
    """WORKING robust denormalization from our local testing"""
    
    # Convert to numpy
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = tensor
    
    # Method 1: If data is roughly in [-1, 1], use standard denorm
    if img.min() >= -1.5 and img.max() <= 1.5:
        img = (img + 1) * 127.5
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    
    # Method 2: Min-max normalization as fallback
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:  # Avoid division by zero
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = np.full_like(img, 128)  # Gray if constant
    
    return np.clip(img, 0, 255).astype(np.uint8)

def generate_synthetic_images():
    """Generate synthetic images using the WORKING method"""
    
    print("🏥 CORRECTED Diffusion Generation Pipeline")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load model and checkpoint
    model = SimpleUNet().to(device)
    checkpoint_path = "diffusion_model_epoch_50.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Make sure the trained checkpoint is in the current directory")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Training loss: {checkpoint.get('loss', 'unknown')}")
    
    # Generation targets (matching style transfer numbers)
    target_counts = {'benign': 175, 'malignant': 89}
    
    print(f"\n🎨 Generating synthetic images...")
    print(f"   Target: {target_counts['benign']} benign + {target_counts['malignant']} malignant")
    
    # Create output directories
    output_dirs = {
        'benign': 'synthetic_images/benign',
        'malignant': 'synthetic_images/malignant'
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate samples
    with torch.no_grad():
        for class_name, target_count in target_counts.items():
            class_label = 0 if class_name == 'benign' else 1
            
            print(f"\n   Generating {target_count} {class_name} samples...")
            
            for i in range(target_count):
                if (i + 1) % 25 == 0:
                    print(f"      Progress: {i+1}/{target_count}")
                
                # Create initial noise
                noise = torch.randn(1, 1, 64, 64, device=device)
                class_tensor = torch.tensor([class_label], device=device)
                
                # CORRECTED denoising process
                x = noise
                num_steps = 20  # More steps for better quality
                
                for step in range(num_steps):
                    t = torch.tensor([num_steps - step - 1], device=device)
                    
                    # Predict noise
                    predicted_noise = model(x, t, class_tensor)
                    
                    # CORRECTED denoising step (from working local version)
                    x = x - 0.05 * predicted_noise
                
                # Convert to image using WORKING robust normalization
                generated_img = x[0, 0]  # Remove batch and channel dims
                img_array = robust_denormalize(generated_img)
                
                # Save image
                img = Image.fromarray(img_array)
                filename = f"{output_dirs[class_name]}/synthetic_{class_name}_{i+1:03d}.png"
                img.save(filename)
            
            print(f"   ✅ Generated {target_count} {class_name} images")
    
    print(f"\n🎉 Generation Complete!")
    print(f"   Output: synthetic_images/benign/ (175 images)")
    print(f"   Output: synthetic_images/malignant/ (89 images)")
    print(f"   Total: 264 synthetic images")

if __name__ == "__main__":
    generate_synthetic_images() 