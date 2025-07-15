#!/usr/bin/env python3
"""
Debug sample loading to understand why the analysis script is failing.
"""

import os
import cv2
import pandas as pd

def check_dataset(name, path, csv_path):
    print(f"\n🔍 Checking {name}...")
    print(f"   Path: {path}")
    print(f"   CSV: {csv_path}")
    
    if not os.path.exists(path):
        print(f"   ❌ Directory doesn't exist")
        return 0
    
    if not os.path.exists(csv_path):
        print(f"   ❌ CSV file doesn't exist")
        return 0
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ CSV loaded: {len(df)} rows")
        print(f"   📋 Columns: {list(df.columns)}")
        
        if len(df) > 0:
            # Try to load first sample
            row = df.iloc[0]
            print(f"   📄 First row: {row.to_dict()}")
            
            img_path = os.path.join(path, row['image_path'])
            mask_path = os.path.join(path, row['mask_path'])
            
            print(f"   🖼️  Image path: {img_path}")
            print(f"   🎭 Mask path: {mask_path}")
            
            img_exists = os.path.exists(img_path)
            mask_exists = os.path.exists(mask_path)
            
            print(f"   🖼️  Image exists: {img_exists}")
            print(f"   🎭 Mask exists: {mask_exists}")
            
            if img_exists and mask_exists:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    print(f"   ✅ Successfully loaded images: {img.shape}, {mask.shape}")
                    return len(df)
                else:
                    print(f"   ❌ Failed to load images (None returned)")
            else:
                print(f"   ❌ Image files don't exist")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return 0

def main():
    print("🔍 Debugging dataset loading...")
    
    datasets = [
        ("BUSI", "dataset/BioMedicalDataset/BUSI", "dataset/BioMedicalDataset/BUSI/train_frame.csv"),
        ("BUS-UCLM", "dataset/BioMedicalDataset/BUS-UCLM", "dataset/BioMedicalDataset/BUS-UCLM/train_frame.csv"),
        ("Styled BUSI", "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Styled", "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Styled/styled_dataset.csv")
    ]
    
    total_samples = 0
    for name, path, csv_path in datasets:
        count = check_dataset(name, path, csv_path)
        total_samples += count
    
    print(f"\n📊 Total available samples: {total_samples}")
    print("="*50)

if __name__ == "__main__":
    main() 