#!/usr/bin/env python3
"""
Test script to verify the new simple BUSI-CCST concatenation approach.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from dataset.BioMedicalDataset.StyledSegmentationDataset import StyledSegmentationDataset
from torch.utils.data import ConcatDataset

def test_simple_ccst_loading():
    """Test the simple concatenation approach for BUSI-CCST."""
    
    print("🧪 Testing Simple BUSI-CCST Concatenation Approach")
    print("=" * 60)
    
    # Test paths
    busi_path = 'dataset/BioMedicalDataset/BUSI'
    styled_path = 'dataset/BioMedicalDataset/BUS-UCLM-Hybrid-Medium'  # Default from IS2D_main.py
    
    print(f"📁 BUSI path: {busi_path}")
    print(f"📁 Styled path: {styled_path}")
    
    # Check if datasets exist
    if not os.path.exists(busi_path):
        print(f"❌ BUSI dataset not found at {busi_path}")
        return False
        
    if not os.path.exists(styled_path):
        print(f"❌ Styled dataset not found at {styled_path}")
        print(f"💡 Run the hybrid pipeline first:")
        print(f"   python run_hybrid_pipeline.py medium")
        return False
    
    try:
        # Load datasets
        print("\n🔄 Loading BUSI dataset...")
        busi_dataset = BUSISegmentationDataset(busi_path, mode='train')
        print(f"   ✅ BUSI dataset loaded: {len(busi_dataset)} samples")
        
        print("\n🔄 Loading styled dataset...")
        styled_dataset = StyledSegmentationDataset(styled_path, mode='train')
        print(f"   ✅ Styled dataset loaded: {len(styled_dataset)} samples")
        
        # Create concatenated dataset
        print("\n🔄 Creating concatenated dataset...")
        combined_dataset = ConcatDataset([busi_dataset, styled_dataset])
        print(f"   ✅ Combined dataset created: {len(combined_dataset)} samples")
        
        # Test data loading
        print("\n🔄 Testing data loading...")
        dataloader = DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Load a few batches
        batch_count = 0
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx >= 3:  # Test first 3 batches
                break
                
            batch_count += 1
            print(f"   Batch {batch_idx + 1}: Images {images.shape}, Masks {masks.shape}")
            
            # Verify image and mask properties
            assert images.dim() == 4, f"Expected 4D images, got {images.dim()}D"
            assert masks.dim() == 4, f"Expected 4D masks, got {masks.dim()}D"
            assert images.shape[0] == masks.shape[0], "Batch size mismatch between images and masks"
            
        print(f"   ✅ Successfully loaded {batch_count} batches")
        
        # Test individual samples
        print("\n🔄 Testing individual samples...")
        
        # Test BUSI sample
        busi_image, busi_mask = busi_dataset[0]
        print(f"   BUSI sample: Image {busi_image.shape}, Mask {busi_mask.shape}")
        
        # Test styled sample
        styled_image, styled_mask = styled_dataset[0]
        print(f"   Styled sample: Image {styled_image.shape}, Mask {styled_mask.shape}")
        
        # Test combined sample
        combined_image, combined_mask = combined_dataset[0]
        print(f"   Combined sample: Image {combined_image.shape}, Mask {combined_mask.shape}")
        
        print("\n✅ All tests passed!")
        print(f"📊 Final statistics:")
        print(f"   Original BUSI: {len(busi_dataset)} samples")
        print(f"   Styled dataset: {len(styled_dataset)} samples")
        print(f"   Total combined: {len(combined_dataset)} samples")
        print(f"   Expected improvement: +{len(styled_dataset)/(len(busi_dataset)+len(styled_dataset))*100:.1f}% more data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_structure(styled_path):
    """Check if the styled dataset has the correct file structure."""
    
    print(f"\n🔍 Checking file structure of {styled_path}")
    
    required_dirs = [
        'benign/images',
        'benign/masks', 
        'malignant/images',
        'malignant/masks'
    ]
    
    required_files = [
        'train_frame.csv'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(styled_path, dir_path)
        if os.path.exists(full_path):
            file_count = len([f for f in os.listdir(full_path) if f.endswith('.png')])
            print(f"   ✅ {dir_path}: {file_count} files")
        else:
            print(f"   ❌ {dir_path}: Missing")
            all_good = False
    
    # Check files
    for file_path in required_files:
        full_path = os.path.join(styled_path, file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {file_path}: Found")
        else:
            print(f"   ❌ {file_path}: Missing")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🧪 Simple BUSI-CCST Loading Test")
    print("=" * 50)
    
    # Check if styled dataset exists and has correct structure
    styled_path = 'dataset/BioMedicalDataset/BUS-UCLM-Hybrid-Medium'
    
    if os.path.exists(styled_path):
        if check_file_structure(styled_path):
            # Run the main test
            success = test_simple_ccst_loading()
            
            if success:
                print("\n🎉 Ready to train with:")
                print("python IS2D_main.py --train_data_type BUSI-CCST --test_data_type BUSI --styled_dataset_path dataset/BioMedicalDataset/BUS-UCLM-Hybrid-Medium --train --final_epoch 100")
            else:
                print("\n❌ Fix the issues above before training")
        else:
            print("\n❌ File structure issues found. Re-run the hybrid pipeline.")
    else:
        print(f"\n❌ Styled dataset not found at {styled_path}")
        print("💡 Run the hybrid pipeline first:")
        print("   python run_hybrid_pipeline.py medium") 