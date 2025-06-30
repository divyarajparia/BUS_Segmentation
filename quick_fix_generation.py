"""
Quick Fix for Synthetic Generation
=================================

This script fixes the immediate issues:
1. Black images due to normalization
2. Missing masks (generates simple masks)
3. Uses existing trained model
"""

import os
import torch
import numpy as np
from PIL import Image
from simple_diffusion_busi import SimpleUNet, SimpleDiffusion
from tqdm import tqdm

def generate_with_smart_normalization(checkpoint_path, output_dir, num_benign=175, num_malignant=89, device='cuda'):
    """Generate images with SMART normalization that adapts to model output"""
    
    print("🔧 Loading existing model with SMART normalization...")
    
    # Load model
    model = SimpleUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = SimpleDiffusion(device=device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_counts = {'benign': num_benign, 'malignant': num_malignant}
    
    for class_idx, class_name in enumerate(['benign', 'malignant']):
        num_samples = class_counts[class_name]
        
        if num_samples == 0:
            continue
        
        # Create directories for images and masks
        image_dir = os.path.join(output_dir, class_name, 'image')
        mask_dir = os.path.join(output_dir, class_name, 'mask')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        print(f'🎨 Generating {num_samples} {class_name} samples with smart normalization...')
        
        batch_size = 4  # Smaller batch for debugging
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        sample_count = 0
        successful_samples = 0
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - sample_count)
            
            # Class labels
            class_labels = torch.full((current_batch_size,), class_idx, device=device)
            
            try:
                # Generate samples
                with torch.no_grad():
                    samples = diffusion.sample(
                        model, 
                        (current_batch_size, 1, 256, 256), 
                        class_labels, 
                        device
                    )
                
                # Process each sample
                for i in range(current_batch_size):
                    sample = samples[i].cpu().numpy()[0]
                    
                    print(f"  Sample {sample_count}: range=[{sample.min():.3f}, {sample.max():.3f}], mean={sample.mean():.3f}")
                    
                    # SMART NORMALIZATION: Try multiple strategies
                    image_array = None
                    method_used = "none"
                    
                    # Strategy 1: If values are roughly in [-1, 1], use standard normalization
                    if sample.min() >= -2 and sample.max() <= 2:
                        sample_clamped = np.clip(sample, -1, 1)
                        image_array = ((sample_clamped + 1) * 127.5).astype(np.uint8)
                        method_used = "clamped"
                    
                    # Strategy 2: If values are very different, use min-max normalization
                    elif sample.max() != sample.min():
                        image_array = ((sample - sample.min()) / (sample.max() - sample.min()) * 255).astype(np.uint8)
                        method_used = "min-max"
                    
                    # Strategy 3: If completely flat, skip
                    else:
                        print(f"    ⚠️ Skipping flat sample {sample_count}")
                        sample_count += 1
                        continue
                    
                    # Check if normalization worked
                    if image_array.max() - image_array.min() < 10:
                        print(f"    ⚠️ Normalization failed, trying robust method...")
                        # Robust normalization using percentiles
                        p1, p99 = np.percentile(sample, [1, 99])
                        if p99 > p1:
                            sample_robust = np.clip(sample, p1, p99)
                            image_array = ((sample_robust - p1) / (p99 - p1) * 255).astype(np.uint8)
                            method_used = "robust"
                        else:
                            print(f"    ❌ Cannot normalize sample {sample_count}, skipping")
                            sample_count += 1
                            continue
                    
                    # Create image
                    image_pil = Image.fromarray(image_array, mode='L')
                    
                    # Generate a simple mask (threshold-based)
                    # This is a placeholder - ideally you'd train a model that generates both
                    mask_array = generate_simple_mask(image_array)
                    mask_pil = Image.fromarray(mask_array, mode='L')
                    
                    # Save files
                    image_filename = f'synthetic_{class_name}_{sample_count:04d}.png'
                    mask_filename = f'synthetic_{class_name}_{sample_count:04d}_mask.png'
                    
                    image_pil.save(os.path.join(image_dir, image_filename))
                    mask_pil.save(os.path.join(mask_dir, mask_filename))
                    
                    print(f"    ✅ Saved sample {sample_count} using {method_used} normalization")
                    sample_count += 1
                    successful_samples += 1
                    
            except Exception as e:
                print(f"    ❌ Error in batch {batch_idx}: {e}")
                sample_count += current_batch_size
                continue
        
        print(f'✅ Successfully generated {successful_samples}/{num_samples} {class_name} samples')
    
    total_generated = successful_samples * 2  # Multiply by 2 for both classes
    print(f'\n🎯 Total successfully generated: {total_generated} image-mask pairs')

def generate_simple_mask(image_array):
    """Generate a simple mask based on image intensity"""
    
    # Method 1: Otsu's thresholding
    try:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(image_array)
        mask = (image_array > thresh).astype(np.uint8) * 255
    except ImportError:
        # Fallback: Use mean + std as threshold
        thresh = image_array.mean() + 0.5 * image_array.std()
        mask = (image_array > thresh).astype(np.uint8) * 255
    
    # Simple morphological operations to clean up mask
    try:
        from scipy import ndimage
        # Fill holes and smooth
        mask = ndimage.binary_fill_holes(mask > 127).astype(np.uint8) * 255
        mask = ndimage.binary_opening(mask > 127, iterations=2).astype(np.uint8) * 255
    except ImportError:
        pass  # Skip morphological operations if scipy not available
    
    return mask

def create_csv_for_synthetic_data(output_dir):
    """Create CSV files for the generated synthetic data"""
    
    import pandas as pd
    
    for split in ['train']:  # Only create training split for synthetic data
        rows = []
        
        for class_name in ['benign', 'malignant']:
            image_dir = os.path.join(output_dir, class_name, 'image')
            mask_dir = os.path.join(output_dir, class_name, 'mask')
            
            if os.path.exists(image_dir):
                for filename in sorted(os.listdir(image_dir)):
                    if filename.endswith('.png'):
                        mask_filename = filename.replace('.png', '_mask.png')
                        
                        rows.append({
                            'image_path': f'{class_name}/image/{filename}',
                            'mask_path': f'{class_name}/mask/{mask_filename}',
                            'class': class_name
                        })
        
        # Save CSV
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f'{split}_frame.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"📋 Created {csv_path} with {len(rows)} samples")

def main():
    checkpoint_path = "diffusion_model_epoch_50.pth"
    output_dir = "./dataset/BioMedicalDataset/BUSI-Synthetic"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint {checkpoint_path} not found!")
        print("Run this script on the server where the model was trained.")
        return
    
    # Generate synthetic data
    generate_with_smart_normalization(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        num_benign=175,
        num_malignant=89,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create CSV files
    create_csv_for_synthetic_data(output_dir)
    
    print("\n🎉 Quick fix generation complete!")
    print("Check the generated images to see if they look reasonable now.")

if __name__ == "__main__":
    main() 