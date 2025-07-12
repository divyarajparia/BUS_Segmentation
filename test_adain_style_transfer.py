#!/usr/bin/env python3
"""
Test script for AdaIN-based style transfer
Generates BUS-UCLM images in BUSI style following CCST methodology
"""

import os
import sys
import torch
from torchvision import transforms
from adain_style_transfer import AdaINStyleTransfer, DomainStyleExtractor, AdaINDatasetGenerator
from dataset.BioMedicalDataset.AdaINStyleTransferDataset import AdaINStyleTransferDataset

def test_adain_style_transfer():
    """Test the AdaIN style transfer pipeline"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Testing AdaIN-based style transfer on {device}")
    
    # Paths
    busi_path = "dataset/BioMedicalDataset/BUSI"
    bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
    output_path = "dataset/BioMedicalDataset/BUS-UCLM-AdaIN-styled"
    
    # Check if datasets exist
    if not os.path.exists(busi_path):
        print(f"❌ BUSI dataset not found at {busi_path}")
        return False
    
    if not os.path.exists(bus_uclm_path):
        print(f"❌ BUS-UCLM dataset not found at {bus_uclm_path}")
        return False
    
    try:
        # Step 1: Initialize AdaIN model
        print("\n📊 Step 1: Initializing AdaIN model...")
        adain_model = AdaINStyleTransfer(device=device)
        print("   ✅ AdaIN model initialized successfully")
        
        # Step 2: Extract BUSI domain style
        print("\n🎨 Step 2: Extracting BUSI domain style...")
        style_extractor = DomainStyleExtractor(adain_model.encoder, device=device)
        busi_domain_style = style_extractor.extract_domain_style(
            busi_path, 
            'train_frame.csv',
            image_folder='image'
        )
        print("   ✅ BUSI domain style extracted successfully")
        
        # Step 3: Generate styled BUS-UCLM dataset (small test first)
        print("\n🔄 Step 3: Generating styled BUS-UCLM dataset...")
        dataset_generator = AdaINDatasetGenerator(adain_model, device=device)
        
        styled_samples = dataset_generator.generate_styled_dataset(
            bus_uclm_path,
            busi_domain_style,
            output_path,
            csv_file='train_frame.csv'
        )
        
        print(f"   ✅ Generated {len(styled_samples)} styled images")
        
        # Step 4: Test dataset loading
        print("\n📁 Step 4: Testing dataset loading...")
        
        # Create basic transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        target_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Test loading the styled dataset
        test_dataset = AdaINStyleTransferDataset(
            busi_dir=busi_path,
            adain_styled_dir=output_path,
            mode='train',
            transform=transform,
            target_transform=target_transform
        )
        
        print(f"   ✅ Dataset loaded successfully: {len(test_dataset)} total samples")
        
        # Test a few samples
        print("\n🧪 Step 5: Testing sample loading...")
        for i in range(min(3, len(test_dataset))):
            image, mask = test_dataset[i]
            sample_info = test_dataset.get_sample_info(i)
            print(f"   Sample {i}: {sample_info['source']} - {sample_info['class']}")
            print(f"      Image shape: {image.shape}, Mask shape: {mask.shape}")
        
        print("\n🎉 AdaIN style transfer test completed successfully!")
        print(f"📁 Styled images saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during AdaIN style transfer: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def quick_test():
    """Quick test with minimal samples"""
    print("🧪 Running quick AdaIN test...")
    return test_adain_style_transfer()

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 