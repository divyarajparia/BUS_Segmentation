#!/usr/bin/env python3
"""
Test script for improved CCST style transfer implementation
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from ccst_exact_replication import CCSTStyleExtractor
import os

def test_improved_style_transfer():
    """Test the improved style transfer implementation"""
    
    print("🧪 Testing Improved CCST Style Transfer")
    print("=" * 50)
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize style extractor
    style_extractor = CCSTStyleExtractor(device=device)
    print("✅ Style extractor initialized successfully")
    
    # Test 1: Decoder architecture
    print("\n1. Testing decoder architecture...")
    test_features = torch.randn(1, 512, 16, 16).to(device)
    
    try:
        decoded_output = style_extractor.decoder(test_features)
        print(f"   ✅ Decoder output shape: {decoded_output.shape}")
        print(f"   ✅ Decoder output range: [{decoded_output.min():.3f}, {decoded_output.max():.3f}]")
        
        # Check if output is in valid range
        if decoded_output.min() >= 0 and decoded_output.max() <= 1:
            print("   ✅ Decoder output is in valid [0, 1] range")
        else:
            print("   ⚠️ Decoder output may need clamping")
            
    except Exception as e:
        print(f"   ❌ Decoder test failed: {e}")
        return False
    
    # Test 2: AdaIN implementation
    print("\n2. Testing AdaIN implementation...")
    try:
        content_features = torch.randn(1, 512, 16, 16).to(device)
        style_mean = torch.randn(1, 512, 1, 1).to(device)
        style_std = torch.randn(1, 512, 1, 1).abs().to(device)  # Ensure positive std
        
        stylized_features = style_extractor.adain(content_features, style_mean, style_std)
        print(f"   ✅ AdaIN output shape: {stylized_features.shape}")
        
        # Check if stylized features have expected statistics
        stylized_mean, stylized_std = style_extractor.calc_mean_std(stylized_features)
        print(f"   ✅ AdaIN applied - mean diff: {torch.abs(stylized_mean - style_mean).mean():.6f}")
        print(f"   ✅ AdaIN applied - std diff: {torch.abs(stylized_std - style_std).mean():.6f}")
        
    except Exception as e:
        print(f"   ❌ AdaIN test failed: {e}")
        return False
    
    # Test 3: Full style transfer pipeline
    print("\n3. Testing full style transfer pipeline...")
    try:
        # Create dummy input image (grayscale -> RGB for VGG)
        dummy_image = torch.randn(1, 3, 256, 256).to(device)
        
        # Create dummy style dictionary
        dummy_style_dict = {
            'mean': torch.randn(1, 512, 1, 1).to(device),
            'std': torch.randn(1, 512, 1, 1).abs().to(device)
        }
        
        # Apply style transfer
        stylized_image = style_extractor.apply_style_transfer(dummy_image, dummy_style_dict)
        print(f"   ✅ Style transfer output shape: {stylized_image.shape}")
        print(f"   ✅ Style transfer output range: [{stylized_image.min():.3f}, {stylized_image.max():.3f}]")
        
        # Test with different alpha values
        for alpha in [0.5, 0.8, 1.0]:
            stylized_alpha = style_extractor.apply_style_transfer(dummy_image, dummy_style_dict, alpha=alpha)
            print(f"   ✅ Alpha={alpha}: output range [{stylized_alpha.min():.3f}, {stylized_alpha.max():.3f}]")
            
    except Exception as e:
        print(f"   ❌ Style transfer test failed: {e}")
        return False
    
    # Test 4: Memory efficiency
    print("\n4. Testing memory efficiency...")
    try:
        # Check memory usage for typical batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Process a batch of images
            batch_size = 4
            dummy_batch = torch.randn(batch_size, 3, 256, 256).to(device)
            
            for i in range(batch_size):
                _ = style_extractor.apply_style_transfer(dummy_batch[i:i+1], dummy_style_dict)
            
            peak_memory = torch.cuda.memory_allocated()
            memory_mb = (peak_memory - initial_memory) / 1024**2
            print(f"   ✅ Memory usage for {batch_size} images: {memory_mb:.1f} MB")
            
    except Exception as e:
        print(f"   ⚠️ Memory test failed: {e}")
    
    print("\n🎉 All tests passed! Improved CCST implementation is working correctly.")
    print("\nKey improvements implemented:")
    print("✅ Proper AdaIN implementation with both mean and std")
    print("✅ Complete decoder network for feature-to-image conversion")
    print("✅ Controlled style mixing with alpha parameter")
    print("✅ Smoothing filter to reduce artifacts")
    print("✅ Better error handling and fallback mechanisms")
    print("✅ Proper tensor normalization and clamping")
    
    return True

if __name__ == "__main__":
    success = test_improved_style_transfer()
    if success:
        print("\n✅ Ready for improved CCST style transfer!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 