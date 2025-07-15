#!/usr/bin/env python3

import os
import pandas as pd
import cv2

styled_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Styled"
styled_csv = f"{styled_path}/styled_dataset.csv"

print(f"🔍 Debugging styled BUSI loading...")
print(f"   Path: {styled_path}")
print(f"   CSV: {styled_csv}")

if os.path.exists(styled_csv):
    styled_df = pd.read_csv(styled_csv)
    print(f"   ✅ CSV loaded: {len(styled_df)} rows")
    print(f"   📋 Columns: {list(styled_df.columns)}")
    
    # Try first few samples
    for i in range(min(3, len(styled_df))):
        row = styled_df.iloc[i]
        print(f"\n   📄 Row {i}: {row.to_dict()}")
        
        class_name = row['class']
        print(f"   🏷️  Class: {class_name}")
        
        # Strip class prefix from filename (CSV has "benign styled_benign.png" but file is "styled_benign.png")
        # Be careful to only remove the prefix, not all instances
        actual_img_filename = row['image_path']
        actual_mask_filename = row['mask_path']
        
        if actual_img_filename.startswith('benign '):
            actual_img_filename = actual_img_filename[7:]  # Remove "benign "
        elif actual_img_filename.startswith('malignant '):
            actual_img_filename = actual_img_filename[10:]  # Remove "malignant "
        
        if actual_mask_filename.startswith('benign '):
            actual_mask_filename = actual_mask_filename[7:]  # Remove "benign "
        elif actual_mask_filename.startswith('malignant '):
            actual_mask_filename = actual_mask_filename[10:]  # Remove "malignant "
        
        print(f"   📁 Processed img filename: {actual_img_filename}")
        print(f"   📁 Processed mask filename: {actual_mask_filename}")
        
        # Construct paths
        img_path = os.path.join(styled_path, class_name, 'image', actual_img_filename)
        mask_path = os.path.join(styled_path, class_name, 'mask', actual_mask_filename)
        
        print(f"   🖼️  Full img path: {img_path}")
        print(f"   🎭 Full mask path: {mask_path}")
        
        img_exists = os.path.exists(img_path)
        mask_exists = os.path.exists(mask_path)
        
        print(f"   🖼️  Image exists: {img_exists}")
        print(f"   🎭 Mask exists: {mask_exists}")
        
        if img_exists and mask_exists:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None and mask is not None:
                print(f"   ✅ Successfully loaded: {img.shape}, {mask.shape}")
            else:
                print(f"   ❌ Failed to load images (None returned)")
        else:
            print(f"   ❌ Files don't exist")
            
            # Let's check what's actually in the directories
            img_dir = os.path.join(styled_path, class_name, 'image')
            mask_dir = os.path.join(styled_path, class_name, 'mask')
            
            if os.path.exists(img_dir):
                files = os.listdir(img_dir)[:5]  # First 5 files
                print(f"   📂 Files in img dir: {files}")
            
            if os.path.exists(mask_dir):
                files = os.listdir(mask_dir)[:5]  # First 5 files
                print(f"   📂 Files in mask dir: {files}")

else:
    print(f"   ❌ CSV doesn't exist") 