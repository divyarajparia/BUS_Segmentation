import os
import cv2
import numpy as np

print('=== DETAILED MASK COLOR ANALYSIS ===')
print()

# Check CycleGAN styled masks (green)
cyclegan_mask_path = 'dataset/BioMedicalDataset/BUSI-Combined/benign/masks/style_ALWI_000.png'
if os.path.exists(cyclegan_mask_path):
    mask = cv2.imread(cyclegan_mask_path)
    print(f'CycleGAN styled mask: {mask.shape}')
    print(f'   Unique values per channel:')
    print(f'   B: {np.unique(mask[:,:,0])}')
    print(f'   G: {np.unique(mask[:,:,1])}') 
    print(f'   R: {np.unique(mask[:,:,2])}')
    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
    print(f'   Combined unique RGB values: {unique_colors}')
    print(f'   Number of unique colors: {len(unique_colors)}')
    print()
else:
    print('❌ CycleGAN styled mask not found')

# Check our CCST styled masks (white)
ccst_mask_path = 'dataset/BioMedicalDataset/BUS-UCLM-Privacy-Styled/benign/mask/styled_benign ALWI_002.png'
if os.path.exists(ccst_mask_path):
    mask = cv2.imread(ccst_mask_path)
    print(f'CCST styled mask: {mask.shape}')
    print(f'   Unique values per channel:')
    print(f'   B: {np.unique(mask[:,:,0])}')
    print(f'   G: {np.unique(mask[:,:,1])}')
    print(f'   R: {np.unique(mask[:,:,2])}')
    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
    print(f'   Combined unique RGB values: {unique_colors}')
    print(f'   Number of unique colors: {len(unique_colors)}')
    print()
else:
    print('❌ CCST styled mask not found')

# Check original BUSI mask for reference (white)  
busi_mask_path = 'dataset/BioMedicalDataset/BUSI/benign/mask/benign (1)_mask.png'
if os.path.exists(busi_mask_path):
    mask = cv2.imread(busi_mask_path)
    print(f'Original BUSI mask: {mask.shape}')
    print(f'   Unique values per channel:')
    print(f'   B: {np.unique(mask[:,:,0])}')
    print(f'   G: {np.unique(mask[:,:,1])}')
    print(f'   R: {np.unique(mask[:,:,2])}')
    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
    print(f'   Combined unique RGB values: {unique_colors}')
    print(f'   Number of unique colors: {len(unique_colors)}')
    print()
else:
    print('❌ Original BUSI mask not found')

print('=== ANALYSIS SUMMARY ===')
print('CycleGAN: Green masks (0,255,0) for styled data')
print('CCST: White masks (255,255,255) for styled data') 
print('BUSI: White masks (255,255,255) for original data')
print()
print('This difference could be significant for training!') 