import os
import cv2
import numpy as np
from PIL import Image

def test_mask_conversion():
    """Test if green vs white masks produce different results after PIL conversion"""
    
    print('=== TESTING MASK PROCESSING PIPELINE ===')
    
    # Test with CycleGAN green mask
    cyclegan_mask_path = 'dataset/BioMedicalDataset/BUSI-Combined/benign/masks/style_ALWI_000.png'
    if os.path.exists(cyclegan_mask_path):
        print('🟢 Testing GREEN mask (CycleGAN):')
        
        # Load as PIL and convert to grayscale (like dataset does)
        mask_pil = Image.open(cyclegan_mask_path).convert('L')
        mask_array = np.array(mask_pil)
        
        print(f'   Original RGB: (0,255,0) for tumor, (0,0,0) for background')
        print(f'   After convert("L"): unique values = {np.unique(mask_array)}')
        print(f'   Shape: {mask_array.shape}')
        print(f'   After >=0.5 threshold: {np.unique((mask_array >= 127.5).astype(int))}')
        print()
    
    # Test with CCST white mask  
    ccst_mask_path = 'dataset/BioMedicalDataset/BUS-UCLM-Privacy-Styled/benign/mask/styled_benign ALWI_002.png'
    if os.path.exists(ccst_mask_path):
        print('⚪ Testing WHITE mask (CCST):')
        
        # Load as PIL and convert to grayscale (like dataset does)
        mask_pil = Image.open(ccst_mask_path).convert('L') 
        mask_array = np.array(mask_pil)
        
        print(f'   Original RGB: (255,255,255) for tumor, (0,0,0) for background')
        print(f'   After convert("L"): unique values = {np.unique(mask_array)}')
        print(f'   Shape: {mask_array.shape}')
        print(f'   After >=0.5 threshold: {np.unique((mask_array >= 127.5).astype(int))}')
        print()
    
    # Test conversion manually
    print('🧪 Manual conversion test:')
    
    # Create green mask (0,255,0)
    green_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    green_mask[30:70, 30:70, 1] = 255  # Green tumor region
    
    # Create white mask (255,255,255)  
    white_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    white_mask[30:70, 30:70, :] = 255  # White tumor region
    
    # Convert both to grayscale
    green_pil = Image.fromarray(green_mask).convert('L')
    white_pil = Image.fromarray(white_mask).convert('L')
    
    green_gray = np.array(green_pil)
    white_gray = np.array(white_pil)
    
    print(f'   Green (0,255,0) → convert("L") → unique: {np.unique(green_gray)}')
    print(f'   White (255,255,255) → convert("L") → unique: {np.unique(white_gray)}')
    
    # Check if they produce same binary mask
    green_binary = (green_gray >= 127.5).astype(int)
    white_binary = (white_gray >= 127.5).astype(int)
    
    print(f'   Green binary: {np.unique(green_binary)}')
    print(f'   White binary: {np.unique(white_binary)}')
    print(f'   Are they identical? {np.array_equal(green_binary, white_binary)}')
    
    return np.array_equal(green_binary, white_binary)

if __name__ == "__main__":
    identical = test_mask_conversion()
    
    print()
    print('=== CONCLUSION ===')
    if identical:
        print('✅ GREEN and WHITE masks produce IDENTICAL binary results!')
        print('   The color difference should NOT affect training.')
        print('   The performance issue is likely from style transfer quality.')
    else:
        print('❌ GREEN and WHITE masks produce DIFFERENT binary results!')
        print('   This could explain the performance degradation.')
        print('   We need to match the CycleGAN green mask approach.') 