"""
Quick Test: Generate Synthetic Images + Masks on Debug Data
==========================================================
Test the generation pipeline on our small debug dataset first.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from simple_diffusion_busi import SimpleUNet

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

def create_synthetic_mask(img_array, class_name):
    """Create synthetic mask based on image and class"""
    
    h, w = img_array.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Different mask patterns for different classes
    if class_name == 'benign':
        # Benign: smaller, more circular regions
        center_y, center_x = h//2 + np.random.randint(-5, 5), w//2 + np.random.randint(-5, 5)
        radius_x, radius_y = w//6 + np.random.randint(-3, 3), h//6 + np.random.randint(-3, 3)
        
        y, x = np.ogrid[:h, :w]
        ellipse_mask = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2) <= 1
        
        # Add some texture based on image intensity
        img_normalized = img_array.astype(float) / 255.0
        intensity_mask = img_normalized > np.percentile(img_normalized, 60)
        
        # Combine ellipse with intensity
        final_mask = ellipse_mask & intensity_mask
        
    else:  # malignant
        # Malignant: larger, more irregular regions
        center_y, center_x = h//2 + np.random.randint(-8, 8), w//2 + np.random.randint(-8, 8)
        radius_x, radius_y = w//4 + np.random.randint(-5, 5), h//4 + np.random.randint(-5, 5)
        
        y, x = np.ogrid[:h, :w]
        ellipse_mask = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2) <= 1
        
        # Add irregular boundaries for malignant
        noise = np.random.random((h, w)) > 0.3
        img_normalized = img_array.astype(float) / 255.0
        intensity_mask = img_normalized > np.percentile(img_normalized, 50)
        
        # More complex combination for malignant
        final_mask = (ellipse_mask & intensity_mask) | (ellipse_mask & noise)
    
    # Add some random noise and smooth
    random_noise = np.random.random((h, w)) < 0.05
    final_mask = final_mask | random_noise
    
    # Convert to 0-255
    mask = (final_mask * 255).astype(np.uint8)
    
    return mask

def test_generation_on_debug_data():
    """Test synthetic generation on debug data"""
    
    print("🧪 Testing Quick Generation on Debug Data")
    print("=" * 50)
    
    # Check checkpoint
    checkpoint_path = "debug_data/checkpoints/diffusion_model_epoch_50.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load model
    model = SimpleUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create output directories
    output_dir = "debug_synthetic_output"
    for class_name in ['benign', 'malignant']:
        os.makedirs(f"{output_dir}/{class_name}/image", exist_ok=True)
        os.makedirs(f"{output_dir}/{class_name}/mask", exist_ok=True)
    
    # Generate samples for each class
    samples_per_class = 3
    
    with torch.no_grad():
        for class_name in ['benign', 'malignant']:
            class_label = 0 if class_name == 'benign' else 1
            
            print(f"\n🎨 Generating {samples_per_class} {class_name} samples...")
            
            for i in range(samples_per_class):
                print(f"   Sample {i+1}/{samples_per_class}")
                
                # Create initial noise
                noise = torch.randn(1, 1, 64, 64, device=device)
                class_tensor = torch.tensor([class_label], device=device)
                
                # Denoising process
                x = noise
                num_steps = 20
                
                for step in range(num_steps):
                    t = torch.tensor([num_steps - step - 1], device=device)
                    predicted_noise = model(x, t, class_tensor)
                    x = x - 0.05 * predicted_noise
                
                # Convert to image
                generated_img = x[0, 0]
                img_array = robust_denormalize(generated_img)
                
                # Create synthetic mask
                mask_array = create_synthetic_mask(img_array, class_name)
                
                # Save image
                img = Image.fromarray(img_array)
                img_filename = f"{output_dir}/{class_name}/image/synthetic_{class_name}_{i+1:02d}.png"
                img.save(img_filename)
                
                # Save mask
                mask_img = Image.fromarray(mask_array)
                mask_filename = f"{output_dir}/{class_name}/mask/synthetic_{class_name}_{i+1:02d}_mask.png"
                mask_img.save(mask_filename)
                
                print(f"      ✅ Saved: {img_filename}")
                print(f"      ✅ Saved: {mask_filename}")
    
    print(f"\n🎉 Debug Generation Complete!")
    print(f"   Check: {output_dir}/")
    print(f"   Total: {samples_per_class * 2} image+mask pairs")
    
    # Show file structure
    print(f"\n📁 Generated Files:")
    for class_name in ['benign', 'malignant']:
        img_dir = f"{output_dir}/{class_name}/image"
        mask_dir = f"{output_dir}/{class_name}/mask"
        if os.path.exists(img_dir):
            img_files = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
            mask_files = len([f for f in os.listdir(mask_dir) if f.endswith('.png')])
            print(f"   {class_name}: {img_files} images, {mask_files} masks")

if __name__ == "__main__":
    test_generation_on_debug_data() 