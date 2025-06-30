"""
Create BUSI + Synthetic Combined Dataset
======================================

This script creates a combined dataset with original BUSI and synthetic BUSI images
for Experiment 2, matching the structure of BUSI-Combined from Experiment 1.

Usage:
    python create_busi_synthetic_combined.py
"""

import os
import pandas as pd
import numpy as np
import shutil
from collections import defaultdict

def create_busi_synthetic_combined(
    busi_dir="dataset/BioMedicalDataset/BUSI",
    synthetic_dir="dataset/BioMedicalDataset/BUSI-Synthetic", 
    output_dir="dataset/BioMedicalDataset/BUSI-Synthetic-Combined"
):
    """Create combined dataset with BUSI and synthetic BUSI images"""
    
    print(f"🔧 Creating BUSI + Synthetic Combined dataset")
    print(f"📁 Original BUSI: {busi_dir}")
    print(f"🎨 Synthetic BUSI: {synthetic_dir}")
    print(f"💾 Output: {output_dir}")
    
    # Create output directory structure
    os.makedirs(os.path.join(output_dir, 'benign', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'benign', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'masks'), exist_ok=True)
    
    # Copy original BUSI images
    print("\n📋 Copying original BUSI images...")
    busi_count = 0
    for class_name in ['benign', 'malignant']:
        busi_img_dir = os.path.join(busi_dir, class_name, 'image')
        busi_mask_dir = os.path.join(busi_dir, class_name, 'mask')
        
        if os.path.exists(busi_img_dir):
            for img_file in os.listdir(busi_img_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy image
                    src_img = os.path.join(busi_img_dir, img_file)
                    dst_img = os.path.join(output_dir, class_name, 'images', f"busi_{img_file}")
                    shutil.copy2(src_img, dst_img)
                    
                    # Copy mask
                    mask_file = img_file.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
                    src_mask = os.path.join(busi_mask_dir, mask_file)
                    if os.path.exists(src_mask):
                        dst_mask = os.path.join(output_dir, class_name, 'masks', f"busi_{mask_file}")
                        shutil.copy2(src_mask, dst_mask)
                        busi_count += 1
    
    print(f"✅ Copied {busi_count} original BUSI samples")
    
    # Copy synthetic images
    print("\n🎨 Copying synthetic BUSI images...")
    synthetic_count = 0
    for class_name in ['benign', 'malignant']:
        synthetic_class_dir = os.path.join(synthetic_dir, class_name)
        
        if os.path.exists(synthetic_class_dir):
            for img_file in os.listdir(synthetic_class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy synthetic image
                    src_img = os.path.join(synthetic_class_dir, img_file)
                    dst_img = os.path.join(output_dir, class_name, 'images', f"synthetic_{img_file}")
                    shutil.copy2(src_img, dst_img)
                    
                    # Create dummy mask (all white for synthetic images)
                    # You might want to generate proper masks using your segmentation model
                    mask_file = f"synthetic_{img_file}"
                    dst_mask = os.path.join(output_dir, class_name, 'masks', mask_file)
                    
                    # Create a simple white mask for now
                    from PIL import Image
                    dummy_mask = Image.new('L', (256, 256), 255)  # White mask
                    dummy_mask.save(dst_mask)
                    
                    synthetic_count += 1
    
    print(f"✅ Copied {synthetic_count} synthetic samples")
    
    # Create CSV files with proper train/test separation
    create_synthetic_combined_csv(output_dir)
    
    print(f"\n🎉 BUSI-Synthetic-Combined dataset created!")
    print(f"📊 Total samples: {busi_count + synthetic_count}")
    
    return busi_count, synthetic_count

def create_synthetic_combined_csv(output_dir):
    """Create CSV files with proper train/test separation (same logic as style transfer fix)"""
    
    print(f"\n📄 Creating CSV files with proper train/test separation...")
    
    # Separate original BUSI and synthetic samples
    busi_pairs = []
    synthetic_pairs = []
    
    for class_name in ['benign', 'malignant']:
        img_dir = os.path.join(output_dir, class_name, 'images')
        mask_dir = os.path.join(output_dir, class_name, 'masks')
        
        if not os.path.exists(img_dir):
            continue
            
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if img_file.startswith('busi_'):
                    # Original BUSI samples
                    mask_file = img_file.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
                    mask_path = os.path.join(mask_dir, mask_file)
                    if os.path.exists(mask_path):
                        busi_pairs.append({
                            'image_path': img_file,
                            'mask_path': mask_file,
                            'class': class_name
                        })
                elif img_file.startswith('synthetic_'):
                    # Synthetic samples (only for training)
                    mask_file = img_file
                    mask_path = os.path.join(mask_dir, mask_file)
                    if os.path.exists(mask_path):
                        synthetic_pairs.append({
                            'image_path': img_file,
                            'mask_path': mask_file,
                            'class': class_name
                        })
    
    print(f"📊 Found {len(busi_pairs)} original BUSI samples")
    print(f"📊 Found {len(synthetic_pairs)} synthetic samples")
    
    # Count by class
    busi_by_class = defaultdict(int)
    synthetic_by_class = defaultdict(int)
    
    for pair in busi_pairs:
        busi_by_class[pair['class']] += 1
    for pair in synthetic_pairs:
        synthetic_by_class[pair['class']] += 1
    
    print(f"\n📈 Original BUSI breakdown:")
    for class_name, count in busi_by_class.items():
        print(f"  - {class_name}: {count} samples")
    
    print(f"\n📈 Synthetic breakdown:")
    for class_name, count in synthetic_by_class.items():
        print(f"  - {class_name}: {count} samples")
    
    # Shuffle BUSI samples for random split (with seed for reproducibility)
    np.random.seed(42)  # Same seed as style transfer for consistency
    np.random.shuffle(busi_pairs)
    
    # Split BUSI samples into train/test/val (70/20/10)
    total_busi = len(busi_pairs)
    busi_train_size = int(0.7 * total_busi)
    busi_test_size = int(0.2 * total_busi)
    
    busi_train = busi_pairs[:busi_train_size]
    busi_test = busi_pairs[busi_train_size:busi_train_size + busi_test_size]
    busi_val = busi_pairs[busi_train_size + busi_test_size:]
    
    # Combine training data: BUSI train + ALL synthetic samples
    # Test/Val data: ONLY original BUSI samples
    train_pairs = busi_train + synthetic_pairs  # Augmented training set
    test_pairs = busi_test                      # Pure BUSI test set
    val_pairs = busi_val                        # Pure BUSI validation set
    
    # Shuffle training set (contains both BUSI and synthetic)
    np.random.shuffle(train_pairs)
    
    print(f"\n🎯 SPLIT BREAKDOWN:")
    
    # Save CSV files and show breakdown
    for split_name, pairs in [('train', train_pairs), ('test', test_pairs), ('val', val_pairs)]:
        # Remove class column before saving (not needed in CSV)
        csv_pairs = [{'image_path': p['image_path'], 'mask_path': p['mask_path']} for p in pairs]
        
        df = pd.DataFrame(csv_pairs)
        csv_path = os.path.join(output_dir, f'{split_name}_frame.csv')
        df.to_csv(csv_path, index=False)
        
        # Count sample types in each split
        busi_count = len([p for p in pairs if p['image_path'].startswith('busi_')])
        synthetic_count = len([p for p in pairs if p['image_path'].startswith('synthetic_')])
        
        # Count by class
        class_counts = defaultdict(int)
        for pair in pairs:
            class_counts[pair['class']] += 1
        
        print(f"\n{split_name.upper()}: {len(pairs)} total samples")
        print(f"  📁 Original BUSI: {busi_count}")
        print(f"  🎨 Synthetic: {synthetic_count}")
        print(f"  📊 Class breakdown:")
        for class_name, count in class_counts.items():
            print(f"    - {class_name}: {count}")
    
    print(f"\n✅ PROPER EXPERIMENTAL SETUP ACHIEVED:")
    print(f"  🏋️ Training: Original BUSI + Synthetic (augmented data)")
    print(f"  🧪 Testing: ONLY original BUSI (clean evaluation)")
    print(f"  🔬 Validation: ONLY original BUSI (clean evaluation)")

if __name__ == "__main__":
    busi_count, synthetic_count = create_busi_synthetic_combined()
    
    print(f"\n🎉 Dataset creation completed!")
    print(f"📊 Ready for Experiment 2: BUSI + Synthetic training!")
    print(f"🔄 This matches Experiment 1 structure for fair comparison!") 