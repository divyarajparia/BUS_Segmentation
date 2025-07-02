#!/usr/bin/env python3
"""
Generate Synthetic Dataset using Trained Joint Diffusion Model
==============================================================
This script loads the trained joint diffusion model and generates
264 synthetic image+mask pairs (175 benign + 89 malignant).

Usage:
    python generate_synthetic_dataset.py

Requirements:
    - joint_diffusion_epoch_50.pth (trained model checkpoint)
    
Output:
    - synthetic_dataset/ folder with generated images and masks
    - CSV files for easy dataset integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Copy the model classes from training script
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

def denormalize_image(tensor):
    """Convert normalized tensor [-1,1] to PIL Image [0,255]"""
    # From [-1,1] to [0,1]
    tensor = (tensor + 1.0) / 2.0
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # To numpy and scale to [0,255]
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return array

def denormalize_mask(tensor):
    """Convert mask tensor to binary PIL Image"""
    # Apply sigmoid to get probabilities
    tensor = torch.sigmoid(tensor)
    # Threshold at 0.5
    tensor = (tensor > 0.5).float()
    # To numpy and scale to [0,255]
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return array

def generate_synthetic_dataset():
    """Generate 264 synthetic image+mask pairs"""
    
    print("🎨 Generating Synthetic Dataset using Trained Joint Diffusion")
    print("=" * 70)
    
    # Check for trained model
    checkpoint_path = "joint_diffusion_epoch_50.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "joint_diffusion_final.pth"
        if not os.path.exists(checkpoint_path):
            print(f"❌ No trained model found!")
            print(f"   Expected: joint_diffusion_epoch_50.pth or joint_diffusion_final.pth")
            return
    
    print(f"📂 Loading trained model: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = JointDiffusionUNet(in_channels=2, out_channels=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup diffusion
    diffusion = JointDiffusion(device=device)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Trained epochs: {checkpoint['epoch']}")
    print(f"   Final loss: {checkpoint['loss']:.4f}")
    print(f"   Training samples: {checkpoint['num_samples']}")
    
    # Generation parameters
    target_benign = 175
    target_malignant = 89
    batch_size = 8  # Generate in batches
    img_size = 64
    
    print(f"\n🎯 Generation Plan:")
    print(f"   Target benign: {target_benign}")
    print(f"   Target malignant: {target_malignant}")
    print(f"   Total samples: {target_benign + target_malignant}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}x{img_size}")
    
    # Create output directories
    output_dir = "synthetic_dataset"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "benign", "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "benign", "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "malignant", "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "malignant", "mask"), exist_ok=True)
    
    # Track generated samples for CSV
    generated_samples = []
    
    # Generate benign samples
    print(f"\n🟢 Generating {target_benign} benign samples...")
    
    benign_count = 0
    while benign_count < target_benign:
        current_batch = min(batch_size, target_benign - benign_count)
        
        # Generate batch
        class_labels = torch.zeros(current_batch, dtype=torch.long, device=device)  # 0 = benign
        generated = diffusion.sample(
            model, 
            (current_batch, 2, img_size, img_size), 
            class_labels=class_labels, 
            device=device
        )
        
        # Save each sample in batch
        for i in range(current_batch):
            sample_idx = benign_count + i + 1
            
            # Extract image and mask
            img_tensor = generated[i, 0:1]  # First channel
            mask_tensor = generated[i, 1:2]  # Second channel
            
            # Convert to numpy
            img_array = denormalize_image(img_tensor)[0]  # Remove channel dim
            mask_array = denormalize_mask(mask_tensor)[0]  # Remove channel dim
            
            # Save as PIL Images
            img_path = os.path.join(output_dir, "benign", "image", f"synthetic_benign_{sample_idx:03d}.png")
            mask_path = os.path.join(output_dir, "benign", "mask", f"synthetic_benign_{sample_idx:03d}_mask.png")
            
            Image.fromarray(img_array, mode='L').save(img_path)
            Image.fromarray(mask_array, mode='L').save(mask_path)
            
            # Add to CSV tracking
            generated_samples.append({
                'image_path': f"benign/image/synthetic_benign_{sample_idx:03d}.png",
                'mask_path': f"benign/mask/synthetic_benign_{sample_idx:03d}_mask.png",
                'class': 'benign',
                'label': 0
            })
        
        benign_count += current_batch
        print(f"   Generated {benign_count}/{target_benign} benign samples")
    
    # Generate malignant samples
    print(f"\n🔴 Generating {target_malignant} malignant samples...")
    
    malignant_count = 0
    while malignant_count < target_malignant:
        current_batch = min(batch_size, target_malignant - malignant_count)
        
        # Generate batch
        class_labels = torch.ones(current_batch, dtype=torch.long, device=device)  # 1 = malignant
        generated = diffusion.sample(
            model, 
            (current_batch, 2, img_size, img_size), 
            class_labels=class_labels, 
            device=device
        )
        
        # Save each sample in batch
        for i in range(current_batch):
            sample_idx = malignant_count + i + 1
            
            # Extract image and mask
            img_tensor = generated[i, 0:1]  # First channel
            mask_tensor = generated[i, 1:2]  # Second channel
            
            # Convert to numpy
            img_array = denormalize_image(img_tensor)[0]  # Remove channel dim
            mask_array = denormalize_mask(mask_tensor)[0]  # Remove channel dim
            
            # Save as PIL Images
            img_path = os.path.join(output_dir, "malignant", "image", f"synthetic_malignant_{sample_idx:03d}.png")
            mask_path = os.path.join(output_dir, "malignant", "mask", f"synthetic_malignant_{sample_idx:03d}_mask.png")
            
            Image.fromarray(img_array, mode='L').save(img_path)
            Image.fromarray(mask_array, mode='L').save(mask_path)
            
            # Add to CSV tracking
            generated_samples.append({
                'image_path': f"malignant/image/synthetic_malignant_{sample_idx:03d}.png",
                'mask_path': f"malignant/mask/synthetic_malignant_{sample_idx:03d}_mask.png",
                'class': 'malignant',
                'label': 1
            })
        
        malignant_count += current_batch
        print(f"   Generated {malignant_count}/{target_malignant} malignant samples")
    
    # Create CSV file for easy dataset integration
    df = pd.DataFrame(generated_samples)
    csv_path = os.path.join(output_dir, "synthetic_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    # Create preview collage
    print(f"\n🖼️ Creating preview collage...")
    create_preview_collage(output_dir)
    
    print(f"\n🎉 Generation Complete!")
    print(f"=" * 70)
    print(f"   📁 Output directory: {output_dir}/")
    print(f"   🟢 Benign samples: {target_benign}")
    print(f"   🔴 Malignant samples: {target_malignant}")
    print(f"   📋 CSV file: {csv_path}")
    print(f"   🖼️ Preview: {output_dir}/preview_collage.png")
    print(f"\n📊 Dataset Summary:")
    print(f"   Total synthetic samples: {len(generated_samples)}")
    print(f"   Benign: {len([s for s in generated_samples if s['class'] == 'benign'])} ({len([s for s in generated_samples if s['class'] == 'benign'])/len(generated_samples)*100:.1f}%)")
    print(f"   Malignant: {len([s for s in generated_samples if s['class'] == 'malignant'])} ({len([s for s in generated_samples if s['class'] == 'malignant'])/len(generated_samples)*100:.1f}%)")
    print(f"\n✅ Ready for training with BUSI + Synthetic dataset!")

def create_preview_collage(output_dir):
    """Create a preview collage showing generated samples"""
    
    # Select random samples for preview
    benign_samples = []
    malignant_samples = []
    
    # Get 4 random benign samples
    for i in [1, 25, 50, 75]:  # Sample different indices
        img_path = os.path.join(output_dir, "benign", "image", f"synthetic_benign_{i:03d}.png")
        mask_path = os.path.join(output_dir, "benign", "mask", f"synthetic_benign_{i:03d}_mask.png")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            benign_samples.append((img_path, mask_path))
    
    # Get 4 random malignant samples
    for i in [1, 15, 30, 45]:  # Sample different indices
        img_path = os.path.join(output_dir, "malignant", "image", f"synthetic_malignant_{i:03d}.png")
        mask_path = os.path.join(output_dir, "malignant", "mask", f"synthetic_malignant_{i:03d}_mask.png")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            malignant_samples.append((img_path, mask_path))
    
    # Create collage
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Synthetic Dataset Preview (Images + Masks)', fontsize=16)
    
    # Plot benign samples
    for i, (img_path, mask_path) in enumerate(benign_samples[:4]):
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Benign Image {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Benign Mask {i+1}')
        axes[i, 1].axis('off')
    
    # Plot malignant samples
    for i, (img_path, mask_path) in enumerate(malignant_samples[:4]):
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        axes[i, 2].imshow(img, cmap='gray')
        axes[i, 2].set_title(f'Malignant Image {i+1}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(mask, cmap='gray')
        axes[i, 3].set_title(f'Malignant Mask {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    preview_path = os.path.join(output_dir, "preview_collage.png")
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_synthetic_dataset() 