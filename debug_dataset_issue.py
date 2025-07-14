import os
import pandas as pd
from PIL import Image
import numpy as np

def debug_dataset_issues():
    """Debug potential issues causing performance degradation"""
    
    print("🔍 DEBUGGING DATASET ISSUES")
    print("=" * 50)
    
    styled_dir = 'dataset/BioMedicalDataset/BUS-UCLM-Hybrid-Medium'
    
    # 1. Check actual filenames vs CSV entries
    print("\n📁 ACTUAL FILENAMES:")
    for class_type in ['benign', 'malignant']:
        img_dir = os.path.join(styled_dir, class_type, 'image')
        if os.path.exists(img_dir):
            files = sorted(os.listdir(img_dir))[:3]
            for file in files:
                print(f"   {class_type}/image/{file}")
    
    # 2. Check CSV entries
    print("\n📄 CSV ENTRIES:")
    csv_path = os.path.join(styled_dir, 'ccst_augmented_dataset.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Total samples: {len(df)}")
        
        # Check first few entries
        for i in range(3):
            print(f"   Row {i}:")
            print(f"     image_path: {df.iloc[i]['image_path']}")
            print(f"     mask_path:  {df.iloc[i]['mask_path']}")
            print(f"     augmentation_type: {df.iloc[i]['augmentation_type']}")
        
        # Check augmentation distribution
        print(f"\n📊 AUGMENTATION DISTRIBUTION:")
        aug_counts = df['augmentation_type'].value_counts()
        for aug_type, count in aug_counts.items():
            print(f"   {aug_type}: {count}")
    
    # 3. Check if files exist for sample entries
    print("\n🔗 PATH VERIFICATION:")
    for i in range(3):
        image_path_info = df.iloc[i]['image_path']
        mask_path_info = df.iloc[i]['mask_path']
        class_type = df.iloc[i]['class']
        aug_type = df.iloc[i]['augmentation_type']
        
        # Handle different filename formats
        if aug_type == 'original':
            # Original BUSI format: "benign (1).png"
            image_path = os.path.join('dataset/BioMedicalDataset/BUSI', 
                                    class_type, 'image', image_path_info)
            mask_path = os.path.join('dataset/BioMedicalDataset/BUSI', 
                                   class_type, 'mask', mask_path_info)
        else:
            # Styled format: "benign hybrid_medium_FILENAME.png"
            if ' ' in image_path_info:
                image_file = image_path_info.split(' ', 1)[1]
                mask_file = mask_path_info.split(' ', 1)[1]
            else:
                image_file = image_path_info
                mask_file = mask_path_info
                
            image_path = os.path.join(styled_dir, class_type, 'image', image_file)
            mask_path = os.path.join(styled_dir, class_type, 'mask', mask_file)
        
        print(f"   Sample {i} ({aug_type}):")
        print(f"     Image exists: {os.path.exists(image_path)} - {image_path}")
        print(f"     Mask exists:  {os.path.exists(mask_path)} - {mask_path}")
    
    # 4. Check image properties
    print("\n🖼️  IMAGE ANALYSIS:")
    sample_paths = []
    
    # Get one original and one styled sample
    for i in range(len(df)):
        if df.iloc[i]['augmentation_type'] == 'original' and len([p for p in sample_paths if 'original' in p]) == 0:
            image_path_info = df.iloc[i]['image_path']
            class_type = df.iloc[i]['class']
            image_path = os.path.join('dataset/BioMedicalDataset/BUSI', 
                                    class_type, 'image', image_path_info)
            sample_paths.append(('original', image_path))
        elif df.iloc[i]['augmentation_type'] == 'hybrid_medium' and len([p for p in sample_paths if 'styled' in p]) == 0:
            image_path_info = df.iloc[i]['image_path']
            class_type = df.iloc[i]['class']
            image_file = image_path_info.split(' ', 1)[1]
            image_path = os.path.join(styled_dir, class_type, 'image', image_file)
            sample_paths.append(('styled', image_path))
        
        if len(sample_paths) >= 2:
            break
    
    for sample_type, image_path in sample_paths:
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img_array = np.array(img)
                
                print(f"   {sample_type.upper()} sample:")
                print(f"     Size: {img.size}")
                print(f"     Mode: {img.mode}")
                print(f"     Array shape: {img_array.shape}")
                print(f"     Array dtype: {img_array.dtype}")
                print(f"     Value range: {img_array.min()} - {img_array.max()}")
                print(f"     Mean: {img_array.mean():.2f}")
                
            except Exception as e:
                print(f"     ERROR loading {sample_type}: {e}")
    
    # 5. Check for potential data issues
    print("\n⚠️  POTENTIAL ISSUES:")
    
    # Check if styled images are grayscale vs RGB
    styled_rgb_count = 0
    styled_gray_count = 0
    original_rgb_count = 0
    original_gray_count = 0
    
    for i in range(min(20, len(df))):  # Check first 20 samples
        try:
            image_path_info = df.iloc[i]['image_path']
            class_type = df.iloc[i]['class']
            aug_type = df.iloc[i]['augmentation_type']
            
            if aug_type == 'original':
                image_path = os.path.join('dataset/BioMedicalDataset/BUSI', 
                                        class_type, 'image', image_path_info)
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    if img.mode == 'RGB':
                        original_rgb_count += 1
                    else:
                        original_gray_count += 1
            else:
                image_file = image_path_info.split(' ', 1)[1]
                image_path = os.path.join(styled_dir, class_type, 'image', image_file)
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    if img.mode == 'RGB':
                        styled_rgb_count += 1
                    else:
                        styled_gray_count += 1
        except:
            continue
    
    print(f"   Original images - RGB: {original_rgb_count}, Grayscale: {original_gray_count}")
    print(f"   Styled images - RGB: {styled_rgb_count}, Grayscale: {styled_gray_count}")
    
    if styled_rgb_count > 0 and styled_gray_count > 0:
        print("   ⚠️  MIXED COLOR MODES in styled images!")
    if original_rgb_count > 0 and original_gray_count > 0:
        print("   ⚠️  MIXED COLOR MODES in original images!")
    
    print("\n" + "=" * 50)
    print("🎯 DEBUGGING COMPLETE")

if __name__ == "__main__":
    debug_dataset_issues() 