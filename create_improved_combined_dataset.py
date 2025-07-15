#!/usr/bin/env python3
"""
Create Improved Combined Dataset
===============================
Combines original BUS-UCLM with improved styled BUSI images for training.
"""

import os
import pandas as pd
import shutil
from pathlib import Path

def create_improved_combined_dataset():
    """Create combined dataset with improved styled images."""
    print("🔄 Creating improved combined dataset...")
    
    # Paths
    bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
    improved_styled_path = "dataset/BioMedicalDataset/BUSI-BUS-UCLM-Improved"
    output_path = "dataset/BioMedicalDataset/BUS-UCLM-Combined-Improved"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load datasets
    print("📊 Loading dataset information...")
    
    # Original BUS-UCLM
    bus_uclm_train = pd.read_csv(f"{bus_uclm_path}/train_frame.csv")
    bus_uclm_val = pd.read_csv(f"{bus_uclm_path}/val_frame.csv")
    
    # Improved styled BUSI
    improved_styled = pd.read_csv(f"{improved_styled_path}/improved_styled_dataset.csv")
    
    print(f"📈 Original BUS-UCLM train: {len(bus_uclm_train)} samples")
    print(f"📈 Original BUS-UCLM val: {len(bus_uclm_val)} samples")
    print(f"🎨 Improved styled BUSI: {len(improved_styled)} samples")
    
    # Combine training data
    combined_train_records = []
    
    # Add original BUS-UCLM train data
    for _, row in bus_uclm_train.iterrows():
        combined_train_records.append({
            'image_path': row['image_path'],
            'mask_path': row['mask_path'],
            'source': 'BUS-UCLM-Original'
        })
    
    # Add original BUS-UCLM val data (for more training data)
    for _, row in bus_uclm_val.iterrows():
        combined_train_records.append({
            'image_path': row['image_path'],
            'mask_path': row['mask_path'],
            'source': 'BUS-UCLM-Original'
        })
    
    # Add improved styled BUSI data
    for _, row in improved_styled.iterrows():
        combined_train_records.append({
            'image_path': row['image_path'],
            'mask_path': row['mask_path'],
            'source': 'BUSI-Improved-Styled'
        })
    
    # Create combined training CSV
    combined_df = pd.DataFrame(combined_train_records)
    csv_path = f"{output_path}/improved_combined_train_frame.csv"
    combined_df.to_csv(csv_path, index=False)
    
    # Copy style statistics for reference
    style_stats_src = f"{improved_styled_path}/enhanced_bus_uclm_style_stats.json"
    style_stats_dst = f"{output_path}/enhanced_bus_uclm_style_stats.json"
    if os.path.exists(style_stats_src):
        shutil.copy2(style_stats_src, style_stats_dst)
    
    # Print summary
    print("\n✅ Improved combined dataset created!")
    print(f"📁 Output path: {output_path}")
    print(f"📊 Total training samples: {len(combined_train_records)}")
    
    # Count by source
    source_counts = combined_df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"   📈 {source}: {count} samples")
    
    # Calculate data increase
    original_count = len(bus_uclm_train) + len(bus_uclm_val)
    increase_percentage = ((len(combined_train_records) - original_count) / original_count) * 100
    print(f"📊 Data increase: {increase_percentage:.1f}% ({original_count} → {len(combined_train_records)})")
    
    return len(combined_train_records)

if __name__ == "__main__":
    create_improved_combined_dataset() 