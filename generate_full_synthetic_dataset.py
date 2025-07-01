"""
Full-Scale Synthetic Image + Mask Generation
===========================================
Generate 264 synthetic BUSI images with masks (175 benign + 89 malignant)
Based on successful debug testing.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import pandas as pd
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

def create_synthetic_mask(img_array, class_name, sample_idx):
    """Create synthetic mask based on image and class with variation"""
    
    h, w = img_array.shape
    
    # Set random seed for reproducibility while maintaining variety
    np.random.seed(sample_idx * 42 + (0 if class_name == 'benign' else 1000))
    
    # Different mask patterns for different classes
    if class_name == 'benign':
        # Benign: smaller, more circular regions
        center_offset = np.random.randint(-8, 8, 2)
        center_y, center_x = h//2 + center_offset[0], w//2 + center_offset[1]
        
        # Smaller, more regular shapes for benign
        base_radius_x = w // 7 + np.random.randint(-2, 3)
        base_radius_y = h // 7 + np.random.randint(-2, 3)
        
        # Create elliptical region
        y, x = np.ogrid[:h, :w]
        ellipse_mask = ((x - center_x)**2 / base_radius_x**2 + 
                       (y - center_y)**2 / base_radius_y**2) <= 1
        
        # Add subtle texture based on image intensity
        img_normalized = img_array.astype(float) / 255.0
        intensity_threshold = np.percentile(img_normalized, 65 + np.random.randint(-5, 5))
        intensity_mask = img_normalized > intensity_threshold
        
        # Combine with some erosion for smoother boundaries
        final_mask = ellipse_mask & intensity_mask
        
    else:  # malignant
        # Malignant: larger, more irregular regions
        center_offset = np.random.randint(-12, 12, 2)
        center_y, center_x = h//2 + center_offset[0], w//2 + center_offset[1]
        
        # Larger, more irregular shapes for malignant
        base_radius_x = w // 4 + np.random.randint(-4, 6)
        base_radius_y = h // 4 + np.random.randint(-4, 6)
        
        # Create irregular elliptical region
        y, x = np.ogrid[:h, :w]
        ellipse_mask = ((x - center_x)**2 / base_radius_x**2 + 
                       (y - center_y)**2 / base_radius_y**2) <= 1
        
        # Add irregular boundaries and extensions
        noise_density = 0.2 + np.random.random() * 0.3
        boundary_noise = np.random.random((h, w)) > noise_density
        
        img_normalized = img_array.astype(float) / 255.0
        intensity_threshold = np.percentile(img_normalized, 55 + np.random.randint(-10, 10))
        intensity_mask = img_normalized > intensity_threshold
        
        # More complex combination for irregular malignant appearance
        final_mask = (ellipse_mask & intensity_mask) | (ellipse_mask & boundary_noise)
        
        # Add some spiculated edges (characteristic of malignant)
        if np.random.random() > 0.3:
            spicule_mask = np.random.random((h, w)) < 0.03
            final_mask = final_mask | (spicule_mask & ellipse_mask)
    
    # Add small amount of random noise for realism
    random_noise = np.random.random((h, w)) < 0.02
    final_mask = final_mask | random_noise
    
    # Clean up mask - remove very small isolated regions
    from scipy import ndimage
    try:
        # Try to use scipy for cleaning if available
        labeled_mask, num_features = ndimage.label(final_mask)
        if num_features > 1:
            # Keep only the largest connected component
            sizes = ndimage.sum(final_mask, labeled_mask, range(num_features + 1))
            max_label = np.argmax(sizes[1:]) + 1
            final_mask = labeled_mask == max_label
    except ImportError:
        # Fallback if scipy not available
        pass
    
    # Convert to 0-255
    mask = (final_mask * 255).astype(np.uint8)
    
    return mask

def generate_full_synthetic_dataset():
    """Generate the complete synthetic dataset"""
    
    print("🏥 FULL-SCALE Synthetic Image + Mask Generation")
    print("=" * 60)
    
    # Check checkpoint
    checkpoint_path = "debug_data/checkpoints/diffusion_model_epoch_50.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Make sure you have the trained diffusion model checkpoint")
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
    print(f"   Training loss: {checkpoint.get('loss', 'unknown')}")
    
    # Generation targets (matching style transfer numbers)
    target_counts = {'benign': 175, 'malignant': 89}
    total_samples = sum(target_counts.values())
    
    print(f"\n🎯 Generation Targets:")
    print(f"   Benign: {target_counts['benign']} samples")
    print(f"   Malignant: {target_counts['malignant']} samples")
    print(f"   Total: {total_samples} image+mask pairs")
    
    # Create output directories
    output_base = "synthetic_diffusion_dataset"
    output_dirs = {
        'benign': {
            'images': f"{output_base}/benign/image",
            'masks': f"{output_base}/benign/mask"
        },
        'malignant': {
            'images': f"{output_base}/malignant/image", 
            'masks': f"{output_base}/malignant/mask"
        }
    }
    
    for class_dirs in output_dirs.values():
        for dir_path in class_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    print(f"\n📁 Output Structure:")
    print(f"   {output_base}/")
    print(f"   ├── benign/")
    print(f"   │   ├── image/ (175 synthetic images)")
    print(f"   │   └── mask/  (175 synthetic masks)")
    print(f"   └── malignant/")
    print(f"       ├── image/ (89 synthetic images)")
    print(f"       └── mask/  (89 synthetic masks)")
    
    # Generate samples
    print(f"\n🎨 Starting Generation...")
    
    # Prepare CSV data for tracking
    csv_data = []
    
    with torch.no_grad():
        for class_name, target_count in target_counts.items():
            class_label = 0 if class_name == 'benign' else 1
            
            print(f"\n   Generating {target_count} {class_name} samples...")
            progress_interval = max(1, target_count // 10)  # Show progress every 10%
            
            for i in range(target_count):
                if (i + 1) % progress_interval == 0 or i == 0:
                    print(f"      Progress: {i+1}/{target_count} ({(i+1)/target_count*100:.1f}%)")
                
                # Create initial noise
                noise = torch.randn(1, 1, 64, 64, device=device)
                class_tensor = torch.tensor([class_label], device=device)
                
                # Denoising process (same as successful debug version)
                x = noise
                num_steps = 20
                
                for step in range(num_steps):
                    t = torch.tensor([num_steps - step - 1], device=device)
                    predicted_noise = model(x, t, class_tensor)
                    x = x - 0.05 * predicted_noise
                
                # Convert to image
                generated_img = x[0, 0]
                img_array = robust_denormalize(generated_img)
                
                # Create synthetic mask with variation
                mask_array = create_synthetic_mask(img_array, class_name, i)
                
                # Save image
                img = Image.fromarray(img_array)
                img_filename = f"synthetic_{class_name}_{i+1:03d}.png"
                img_path = f"{output_dirs[class_name]['images']}/{img_filename}"
                img.save(img_path)
                
                # Save mask
                mask_img = Image.fromarray(mask_array)
                mask_filename = f"synthetic_{class_name}_{i+1:03d}_mask.png"
                mask_path = f"{output_dirs[class_name]['masks']}/{mask_filename}"
                mask_img.save(mask_path)
                
                # Add to CSV data
                csv_data.append({
                    'image_path': f"{class_name}/image/{img_filename}",
                    'mask_path': f"{class_name}/mask/{mask_filename}",
                    'class': class_name,
                    'label': class_label
                })
            
            print(f"   ✅ Completed {target_count} {class_name} samples")
    
    # Create CSV file for easy loading
    csv_df = pd.DataFrame(csv_data)
    csv_path = f"{output_base}/synthetic_dataset.csv"
    csv_df.to_csv(csv_path, index=False)
    
    print(f"\n🎉 Full-Scale Generation Complete!")
    print(f"   📊 Generated {len(csv_data)} image+mask pairs")
    print(f"   📁 Output: {output_base}/")
    print(f"   📋 CSV: {csv_path}")
    print(f"   🎯 Ready for segmentation training!")
    
    # Show final statistics
    print(f"\n📈 Final Statistics:")
    for class_name in ['benign', 'malignant']:
        class_count = len([d for d in csv_data if d['class'] == class_name])
        print(f"   {class_name.capitalize()}: {class_count} samples")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Check sample images in: {output_base}/")
    print(f"   2. Use CSV file for training: {csv_path}")
    print(f"   3. Compare with style transfer results")

if __name__ == "__main__":
    generate_full_synthetic_dataset() 