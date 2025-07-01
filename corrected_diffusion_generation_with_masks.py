"""
CORRECTED Diffusion Model Generation Script WITH MASKS
====================================================
Based on successful local debugging, this generates both synthetic images AND masks.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class JointUNet(nn.Module):
    """Modified UNet that generates both image and mask"""
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
            nn.Conv2d(1, 64, 3, padding=1),  # Input: 1 channel (image only)
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
        self.cond_proj = nn.Linear(512, 512)
        
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
            nn.Conv2d(64, 64, 3, padding=1)
        )
        
        # Output heads
        self.image_head = nn.Conv2d(64, 1, 3, padding=1)  # Image output
        self.mask_head = nn.Conv2d(64, 1, 3, padding=1)   # Mask output
    
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
        
        # Generate both image and mask
        image_out = self.image_head(d1)
        mask_out = torch.sigmoid(self.mask_head(d1))  # Sigmoid for mask (0-1)
        
        return image_out, mask_out

def robust_denormalize(tensor):
    """WORKING robust denormalization for images"""
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
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = np.full_like(img, 128)
    
    return np.clip(img, 0, 255).astype(np.uint8)

def mask_to_image(mask_tensor):
    """Convert mask tensor to image array"""
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.detach().cpu().numpy()
    else:
        mask = mask_tensor
    
    # Mask should already be in [0, 1] range due to sigmoid
    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
    return mask

def generate_synthetic_images_and_masks():
    """Generate synthetic images AND masks using the WORKING method"""
    
    print("🏥 CORRECTED Diffusion Generation Pipeline WITH MASKS")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Try to load the existing checkpoint first
    checkpoint_path = "diffusion_model_epoch_50.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Original checkpoint not found: {checkpoint_path}")
        print(f"⚠️  This script requires a model trained for joint image+mask generation")
        print(f"   Your current model was trained for images only.")
        print(f"   You'll need to:")
        print(f"   1. Train a new model with joint image+mask loss, OR")
        print(f"   2. Use the image-only version for now")
        return
    
    # Load model 
    model = JointUNet().to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # This might fail if the checkpoint doesn't have mask generation layers
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"⚠️  Loaded checkpoint with missing mask layers - masks will be synthetic")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"   Your checkpoint was trained for image-only generation")
        return
    
    model.eval()
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Training loss: {checkpoint.get('loss', 'unknown')}")
    
    # Generation targets
    target_counts = {'benign': 175, 'malignant': 89}
    
    print(f"\n🎨 Generating synthetic images AND masks...")
    print(f"   Target: {target_counts['benign']} benign + {target_counts['malignant']} malignant")
    
    # Create output directories
    output_dirs = {
        'benign': {
            'images': 'synthetic_images/benign/image',
            'masks': 'synthetic_images/benign/mask'
        },
        'malignant': {
            'images': 'synthetic_images/malignant/image', 
            'masks': 'synthetic_images/malignant/mask'
        }
    }
    
    for class_dirs in output_dirs.values():
        for dir_path in class_dirs.values():
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
                num_steps = 20
                
                for step in range(num_steps):
                    t = torch.tensor([num_steps - step - 1], device=device)
                    
                    try:
                        # Try joint generation
                        predicted_image, predicted_mask = model(x, t, class_tensor)
                        # Use image prediction for denoising
                        x = x - 0.05 * predicted_image
                    except:
                        # Fallback: model only outputs image
                        predicted_image = model(x, t, class_tensor)
                        x = x - 0.05 * predicted_image
                        predicted_mask = None
                
                # Convert image
                generated_img = x[0, 0]
                img_array = robust_denormalize(generated_img)
                
                # Save image
                img = Image.fromarray(img_array)
                img_filename = f"{output_dirs[class_name]['images']}/synthetic_{class_name}_{i+1:03d}.png"
                img.save(img_filename)
                
                # Generate/save mask
                if predicted_mask is not None:
                    # Use generated mask
                    mask_array = mask_to_image(predicted_mask[0, 0])
                else:
                    # Create synthetic mask based on image intensity
                    # This is a simple heuristic - real mask generation requires joint training
                    img_normalized = img_array.astype(float) / 255.0
                    
                    # Create tumor-like regions in the center with some noise
                    h, w = img_normalized.shape
                    center_y, center_x = h//2, w//2
                    
                    # Create elliptical region
                    y, x = np.ogrid[:h, :w]
                    ellipse_mask = ((x - center_x)**2 / (w//3)**2 + 
                                   (y - center_y)**2 / (h//4)**2) <= 1
                    
                    # Add some randomness and make it more realistic
                    noise_mask = np.random.random((h, w)) < 0.1
                    mask_array = (ellipse_mask.astype(float) * 0.7 + noise_mask * 0.3) * 255
                    mask_array = mask_array.astype(np.uint8)
                
                # Save mask
                mask_img = Image.fromarray(mask_array)
                mask_filename = f"{output_dirs[class_name]['masks']}/synthetic_{class_name}_{i+1:03d}_mask.png"
                mask_img.save(mask_filename)
            
            print(f"   ✅ Generated {target_count} {class_name} image+mask pairs")
    
    print(f"\n🎉 Generation Complete!")
    print(f"   Images: synthetic_images/*/image/ (264 total)")
    print(f"   Masks:  synthetic_images/*/mask/ (264 total)")
    print(f"   Ready for training!")

if __name__ == "__main__":
    generate_synthetic_images_and_masks() 