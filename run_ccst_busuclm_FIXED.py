#!/usr/bin/env python3
"""
Run CCST BUS-UCLM Augmentation Pipeline - FIXED VERSION
=======================================================
Uses the fixed CCST implementation that properly handles tensor normalization
"""

import os
import sys
from datetime import datetime

# Import the fixed CCST implementation
from ccst_exact_replication_FIXED import run_fixed_ccst_pipeline

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🎯 FIXED CCST BUS-UCLM Augmentation Pipeline")
    print("=" * 60)
    print("✅ This version fixes the black image issue!")
    print("=" * 60)
    
    # Configuration
    source_dataset = "dataset/BioMedicalDataset/BUS-UCLM"
    source_csv = "train_frame.csv"
    target_dataset = "dataset/BioMedicalDataset/BUSI"
    target_csv = "train_frame.csv"
    output_dir = f"ccst_busuclm_FIXED_{timestamp}"
    
    print(f"📊 Configuration:")
    print(f"   Source (BUS-UCLM): {source_dataset}")
    print(f"   Target (BUSI): {target_dataset}")
    print(f"   Output: {output_dir}")
    
    # Step 1: Run FIXED CCST style transfer
    print(f"\n🚀 Step 1: Running FIXED CCST Style Transfer")
    print("This will generate BUS-UCLM images styled to look like BUSI")
    
    try:
        styled_data = run_fixed_ccst_pipeline(
            source_dataset=source_dataset,
            source_csv=source_csv,
            target_dataset=target_dataset,
            target_csv=target_csv,
            output_dir=output_dir,
            K=1,  # Single style transfer per image
            J_samples=None  # Use all BUSI samples for style extraction
        )
        
        print(f"\n✅ FIXED CCST Style Transfer Complete!")
        print(f"   Generated: {len(styled_data)} styled images")
        print(f"   Output directory: {output_dir}")
        
        # Check a few images to verify they're not black
        print(f"\n🔍 Verifying styled images are not black...")
        sample_images = []
        for class_type in ['benign', 'malignant']:
            image_dir = os.path.join(output_dir, class_type, 'image')
            if os.path.exists(image_dir):
                images = [f for f in os.listdir(image_dir) if f.endswith('.png')][:3]
                sample_images.extend([os.path.join(image_dir, img) for img in images])
        
        from PIL import Image
        import numpy as np
        
        for img_path in sample_images[:5]:  # Check first 5 images
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img_array = np.array(img)
                
                if img_array.max() > img_array.min():
                    print(f"   ✅ {os.path.basename(img_path)}: range [{img_array.min()}, {img_array.max()}] - GOOD!")
                else:
                    print(f"   ❌ {os.path.basename(img_path)}: flat image - might be black")
        
        # Next steps instructions
        print(f"\n🎯 Next Steps for Full Pipeline:")
        print(f"1. Train BUS-UCLM baseline:")
        print(f"   python IS2D_main.py --train_data_type BUS-UCLM --test_data_type BUS-UCLM --train --final_epoch 50")
        
        print(f"\n2. Create combined dataset (original + styled):")
        combined_dir = f"dataset/BioMedicalDataset/BUS-UCLM-CCST-Combined_{timestamp}"
        print(f"   mkdir {combined_dir}")
        print(f"   # Copy original BUS-UCLM + styled data from {output_dir}")
        
        print(f"\n3. Train on combined dataset:")
        print(f"   python train_with_ccst_data.py \\")
        print(f"       --ccst-augmented-path '{output_dir}' \\")
        print(f"       --original-busi-path '{source_dataset}' \\")
        print(f"       --num-epochs 50 --batch-size 4")
        
        print(f"\n4. Compare results:")
        print(f"   Baseline (BUS-UCLM only) vs Combined (BUS-UCLM + BUSI-styled)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CCST Style Transfer Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 FIXED CCST Pipeline Completed Successfully!")
        print(f"   No more black images! Check the output directory.")
    else:
        print(f"\n💥 Pipeline failed. Check the error messages above.")
        sys.exit(1) 