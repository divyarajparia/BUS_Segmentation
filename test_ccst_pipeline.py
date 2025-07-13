#!/usr/bin/env python3
"""
Test CCST Pipeline with Debug Data
=================================

This script tests the complete CCST pipeline using the debug data
to ensure everything works correctly before running on the full dataset.
"""

import os
import sys
import torch
import shutil
from datetime import datetime


def setup_test_environment():
    """Setup test environment and directories"""
    print("🔧 Setting up test environment...")
    
    # Create test directories
    test_dirs = [
        'test_results',
        'test_output',
        'test_logs',
        'test_models'
    ]
    
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"   ✅ Created {dir_name}")
    
    # Check if debug data exists
    debug_busi_dir = 'debug_data/BUSI'
    if not os.path.exists(debug_busi_dir):
        print(f"   ❌ Debug BUSI data not found at {debug_busi_dir}")
        return False
    
    print(f"   ✅ Debug BUSI data found at {debug_busi_dir}")
    
    # Create mock BUS-UCLM debug data if it doesn't exist
    debug_bus_uclm_dir = 'debug_data/BUS-UCLM'
    if not os.path.exists(debug_bus_uclm_dir):
        print(f"   📝 Creating mock BUS-UCLM debug data...")
        create_mock_bus_uclm_data(debug_bus_uclm_dir)
    
    print(f"   ✅ BUS-UCLM debug data ready at {debug_bus_uclm_dir}")
    
    return True


def create_mock_bus_uclm_data(output_dir):
    """Create mock BUS-UCLM data for testing"""
    # Create directory structure
    os.makedirs(os.path.join(output_dir, 'benign', 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'benign', 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'malignant', 'mask'), exist_ok=True)
    
    # Copy some BUSI images as mock BUS-UCLM data
    busi_debug_dir = 'debug_data/BUSI'
    
    # Copy benign images
    for i in range(1, 3):  # Copy first 2 benign images
        src_img = os.path.join(busi_debug_dir, 'benign', 'image', f'benign ({i}).png')
        src_mask = os.path.join(busi_debug_dir, 'benign', 'mask', f'benign ({i})_mask.png')
        
        if os.path.exists(src_img) and os.path.exists(src_mask):
            dst_img = os.path.join(output_dir, 'benign', 'image', f'bus_uclm_benign_{i:03d}.png')
            dst_mask = os.path.join(output_dir, 'benign', 'mask', f'bus_uclm_benign_{i:03d}_mask.png')
            
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
    
    # Copy malignant images
    for i in range(1, 3):  # Copy first 2 malignant images
        src_img = os.path.join(busi_debug_dir, 'malignant', 'image', f'malignant ({i}).png')
        src_mask = os.path.join(busi_debug_dir, 'malignant', 'mask', f'malignant ({i})_mask.png')
        
        if os.path.exists(src_img) and os.path.exists(src_mask):
            dst_img = os.path.join(output_dir, 'malignant', 'image', f'bus_uclm_malignant_{i:03d}.png')
            dst_mask = os.path.join(output_dir, 'malignant', 'mask', f'bus_uclm_malignant_{i:03d}_mask.png')
            
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
    
    # Create train_frame.csv for BUS-UCLM
    import pandas as pd
    
    samples = []
    for i in range(1, 3):
        samples.append({
            'image_path': f'benign/image/bus_uclm_benign_{i:03d}.png',
            'mask_path': f'benign/mask/bus_uclm_benign_{i:03d}_mask.png',
            'class': 'benign'
        })
        samples.append({
            'image_path': f'malignant/image/bus_uclm_malignant_{i:03d}.png',
            'mask_path': f'malignant/mask/bus_uclm_malignant_{i:03d}_mask.png',
            'class': 'malignant'
        })
    
    df = pd.DataFrame(samples)
    df.to_csv(os.path.join(output_dir, 'train_frame.csv'), index=False)
    
    print(f"   ✅ Created mock BUS-UCLM data with {len(samples)} samples")


def test_dataset_loading():
    """Test dataset loading functionality"""
    print("\n📊 Testing dataset loading...")
    
    try:
        # Test BUSICCSTCombinedDataset loading
        from dataset.BioMedicalDataset.BUSICCSTCombinedDataset import BUSICCSTCombinedDataset
        from torchvision import transforms
        
        # Create simple transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # This should work for validation/test mode even without combined dataset
        dataset = BUSICCSTCombinedDataset(
            combined_dir='debug_data/BUSI',  # Won't be used for test mode
            original_busi_dir='debug_data/BUSI',
            mode='test',
            transform=transform,
            target_transform=transform
        )
        
        print(f"   ✅ Dataset loaded successfully")
        print(f"   📊 Dataset size: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"   ✅ Sample loaded: image shape {image.shape}, mask shape {mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        return False


def test_style_extractor():
    """Test the privacy-preserving style extractor"""
    print("\n🎨 Testing style extractor...")
    
    try:
        from ccst_privacy_preserving_adain import PrivacyPreservingStyleExtractor
        
        # Initialize extractor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = PrivacyPreservingStyleExtractor(device=device)
        
        # Test style extraction on debug data
        busi_style = extractor.extract_domain_style(
            'debug_data/BUSI',
            'train_frame.csv',
            save_path='test_output/busi_style_stats.json'
        )
        
        print(f"   ✅ Style extraction successful")
        print(f"   📈 Style statistics: mean shape {busi_style['mean'].shape}, std shape {busi_style['std'].shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Style extraction failed: {e}")
        return False


def test_adain_model():
    """Test the AdaIN model"""
    print("\n🔧 Testing AdaIN model...")
    
    try:
        from ccst_privacy_preserving_adain import PrivacyPreservingAdaIN
        
        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PrivacyPreservingAdaIN(device=device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_style_stats = {
            'mean': torch.randn(1, 512, 1, 1).to(device),
            'std': torch.randn(1, 512, 1, 1).to(device)
        }
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input, dummy_style_stats)
        
        print(f"   ✅ AdaIN model working")
        print(f"   📐 Input shape: {dummy_input.shape}, Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ AdaIN model test failed: {e}")
        return False


def test_madgnet_loading():
    """Test MADGNet model loading"""
    print("\n🧠 Testing MADGNet model loading...")
    
    try:
        from IS2D_models.mfmsnet import MFMSNet
        
        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MFMSNet().to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, 224, 224).to(device)
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input, mode='test')
        
        print(f"   ✅ MADGNet model working")
        print(f"   📐 Input shape: {dummy_input.shape}")
        
        if isinstance(output, list):
            print(f"   📊 Output: {len(output)} outputs")
            for i, out in enumerate(output):
                if isinstance(out, list):
                    print(f"      Output {i}: {len(out)} components")
                    for j, comp in enumerate(out):
                        print(f"        Component {j}: {comp.shape}")
                else:
                    print(f"      Output {i}: {out.shape}")
        else:
            print(f"   📊 Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ MADGNet model test failed: {e}")
        return False


def run_mini_pipeline():
    """Run a mini version of the pipeline with debug data"""
    print("\n🚀 Running mini pipeline test...")
    
    try:
        # Run style transfer with debug data
        command = (
            f"python ccst_privacy_preserving_adain.py "
            f"--busi-dir debug_data/BUSI "
            f"--bus-uclm-dir debug_data/BUS-UCLM "
            f"--output-dir test_output/combined_debug "
            f"--device cpu"  # Use CPU for testing
        )
        
        print(f"   🔄 Running: {command}")
        
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Mini pipeline successful!")
            
            # Check if output was created
            output_dir = 'test_output/combined_debug'
            if os.path.exists(output_dir):
                print(f"   📁 Output directory created: {output_dir}")
                
                # Check for CSV file
                csv_path = os.path.join(output_dir, 'combined_train_frame.csv')
                if os.path.exists(csv_path):
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    print(f"   📊 Combined dataset CSV: {len(df)} samples")
                
            return True
        else:
            print(f"   ❌ Mini pipeline failed!")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Mini pipeline test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 CCST Pipeline Test Suite")
    print("=" * 50)
    print(f"Start time: {datetime.now()}")
    
    # Setup test environment
    if not setup_test_environment():
        print("❌ Test environment setup failed!")
        return False
    
    # Run tests
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Style Extractor", test_style_extractor),
        ("AdaIN Model", test_adain_model),
        ("MADGNet Model", test_madgnet_loading),
        ("Mini Pipeline", run_mini_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CCST pipeline is ready.")
        print("\n🚀 Ready to run the full pipeline:")
        print("   python run_complete_ccst_pipeline.py")
    else:
        print(f"⚠️  {total - passed} tests failed. Please fix issues before running the full pipeline.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 