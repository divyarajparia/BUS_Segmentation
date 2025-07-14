#!/usr/bin/env python3
"""
Fix the CSV file for BUS-UCLM-Hybrid-Medium dataset.
Changes mask filenames from adding _mask.png suffix to same name as image.
"""

import os
import pandas as pd

def fix_csv_file(styled_dir):
    """Fix the train_frame.csv to use correct mask filenames."""
    
    print(f"🔧 Fixing CSV file for {styled_dir}")
    
    samples = []
    
    # Check benign images
    benign_image_dir = os.path.join(styled_dir, 'benign', 'image')  # Use actual folder name
    
    if os.path.exists(benign_image_dir):
        for filename in sorted(os.listdir(benign_image_dir)):
            if filename.endswith('.png'):
                mask_filename = filename  # Mask has same name as image (fixed!)
                samples.append({
                    'image_path': f'benign {filename}',
                    'mask_path': f'benign {mask_filename}'
                })
    
    # Check malignant images  
    malignant_image_dir = os.path.join(styled_dir, 'malignant', 'image')  # Use actual folder name
    
    if os.path.exists(malignant_image_dir):
        for filename in sorted(os.listdir(malignant_image_dir)):
            if filename.endswith('.png'):
                mask_filename = filename  # Mask has same name as image (fixed!)
                samples.append({
                    'image_path': f'malignant {filename}',
                    'mask_path': f'malignant {mask_filename}'
                })
    
    # Save corrected CSV
    csv_path = os.path.join(styled_dir, 'train_frame.csv')
    df = pd.DataFrame(samples)
    df.to_csv(csv_path, index=False)
    
    benign_count = len([s for s in samples if s['image_path'].startswith('benign')])
    malignant_count = len([s for s in samples if s['image_path'].startswith('malignant')])
    
    print(f"   ✅ Fixed CSV created: {len(samples)} samples")
    print(f"   📊 Breakdown:")
    print(f"      Benign: {benign_count}")
    print(f"      Malignant: {malignant_count}")
    print(f"   📄 Saved to: {csv_path}")
    
    # Show first few entries
    print(f"\n   📋 Sample entries:")
    for i, sample in enumerate(samples[:3]):
        print(f"      {i+1}. Image: {sample['image_path']}")
        print(f"         Mask:  {sample['mask_path']}")
    
    return csv_path

if __name__ == "__main__":
    styled_dir = 'dataset/BioMedicalDataset/BUS-UCLM-Hybrid-Medium'
    
    if os.path.exists(styled_dir):
        fix_csv_file(styled_dir)
        print(f"\n✅ CSV file fixed! Now you can train:")
        print(f"python IS2D_main.py --train_data_type BUSI-CCST --test_data_type BUSI --styled_dataset_path {styled_dir} --train --final_epoch 100")
    else:
        print(f"❌ Dataset not found at {styled_dir}")
        print(f"💡 Generate it first: python run_hybrid_pipeline.py medium") 