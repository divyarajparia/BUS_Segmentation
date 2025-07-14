#!/usr/bin/env python3
"""
Fix CSV format for CCSTDataset compatibility.
Ensures all required columns are present for IS2D training.
"""

import pandas as pd
import os
import sys

def fix_ccst_csv_format():
    """
    Fix the CSV format to ensure CCSTDataset compatibility.
    This addresses the KeyError: 'source_client' issue.
    """
    print("🔧 Fixing CCST CSV format for CCSTDataset compatibility...")
    
    # Paths
    busi_train_path = 'dataset/BioMedicalDataset/BUSI/train_frame.csv'
    styled_data_path = 'dataset/BioMedicalDataset/BUS-UCLM-Privacy-Styled/styled_dataset.csv'
    output_path = 'dataset/BioMedicalDataset/BUS-UCLM-Privacy-Styled/combined_ccst_train.csv'
    
    # Check if files exist
    if not os.path.exists(busi_train_path):
        print(f"❌ BUSI training CSV not found: {busi_train_path}")
        return False
    
    if not os.path.exists(styled_data_path):
        print(f"❌ Styled dataset CSV not found: {styled_data_path}")
        return False
    
    try:
        # Load datasets
        print("📊 Loading datasets...")
        busi_df = pd.read_csv(busi_train_path)
        styled_df = pd.read_csv(styled_data_path)
        
        # Add required columns for CCSTDataset compatibility
        # For original BUSI data - derive class from image_path
        print("🔄 Processing BUSI data...")
        busi_df['class'] = busi_df['image_path'].apply(lambda x: 'benign' if 'benign' in x else 'malignant')
        busi_df['source_client'] = 'BUSI'
        busi_df['style_client'] = 'BUSI'
        busi_df['augmentation_type'] = 'original'
        
        # Ensure styled data has required columns
        print("🎨 Processing styled data...")
        if 'source_client' not in styled_df.columns:
            styled_df['source_client'] = 'BUS-UCLM'
        if 'style_client' not in styled_df.columns:
            styled_df['style_client'] = 'BUSI'
        if 'augmentation_type' not in styled_df.columns:
            styled_df['augmentation_type'] = 'styled'
        
        # Combine datasets
        print("🔗 Combining datasets...")
        combined_df = pd.concat([busi_df, styled_df], ignore_index=True)
        
        # Ensure columns are in the correct order for CCSTDataset
        column_order = ['image_path', 'mask_path', 'class', 'source_client', 'style_client', 'augmentation_type']
        if 'original_image' in combined_df.columns:
            column_order.append('original_image')
        
        # Save combined dataset
        print("💾 Saving combined dataset...")
        combined_df[column_order].to_csv(output_path, index=False)
        
        print(f"✅ Fixed CSV saved: {output_path}")
        print(f"📊 Total samples: {len(combined_df)}")
        print(f"📊 Original BUSI: {len(busi_df)}")
        print(f"📊 Privacy-styled BUS-UCLM: {len(styled_df)}")
        
        # Verify the fix
        print("\n🔍 Verifying fix...")
        test_df = pd.read_csv(output_path)
        required_columns = ['image_path', 'mask_path', 'class', 'source_client', 'style_client', 'augmentation_type']
        
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            print(f"❌ Still missing columns: {missing_columns}")
            return False
        
        print("✅ All required columns present!")
        print("✅ CCSTDataset compatibility verified!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing CSV format: {e}")
        return False

if __name__ == "__main__":
    print("🔧 CCST CSV Format Fixer")
    print("=" * 50)
    
    success = fix_ccst_csv_format()
    
    if success:
        print("\n🎉 CSV format fixed successfully!")
        print("✅ Ready to train with IS2D_main.py")
        print("\nNext steps:")
        print("python IS2D_main.py --train_data_type BUSI-CCST --test_data_type BUSI --ccst_augmented_path dataset/BioMedicalDataset/BUS-UCLM-Privacy-Styled --train --final_epoch 100")
    else:
        print("\n❌ Failed to fix CSV format")
        sys.exit(1) 