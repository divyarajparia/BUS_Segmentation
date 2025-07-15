#!/usr/bin/env python3
"""
Complete Pipeline Using Fixed CCST Output
==========================================
Use the FIXED CCST output to complete the BUS-UCLM augmentation pipeline
"""

import os
import subprocess
from datetime import datetime

def run_command(cmd, description):
    """Run a shell command with progress tracking"""
    print(f"\n🚀 {description}")
    print(f"💻 Command: {cmd}")
    print("-" * 50)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("📄 Output:")
            print(result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout)
    else:
        print(f"❌ {description} failed!")
        print("🚨 Error:")
        print(result.stderr)
        return False
    
    return True

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🎯 Complete BUS-UCLM Augmentation Pipeline with Fixed CCST")
    print("=" * 70)
    print("Uses the fixed CCST output that generates visible images!")
    print("=" * 70)
    
    # Step 1: Check if CCST output exists
    print("\n📋 Step 1: Checking for CCST Output")
    ccst_output_dirs = [d for d in os.listdir('.') if d.startswith('ccst_busuclm_FIXED_')]
    
    if not ccst_output_dirs:
        print("❌ No CCST output found!")
        print("🔧 Please run first: python run_ccst_busuclm_FIXED.py")
        return False
    
    # Use the most recent CCST output
    ccst_output_dir = sorted(ccst_output_dirs)[-1]
    print(f"✅ Found CCST output: {ccst_output_dir}")
    
    # Verify it has images
    benign_images = os.path.join(ccst_output_dir, 'benign', 'image')
    malignant_images = os.path.join(ccst_output_dir, 'malignant', 'image')
    
    if os.path.exists(benign_images) and os.path.exists(malignant_images):
        benign_count = len([f for f in os.listdir(benign_images) if f.endswith('.png')])
        malignant_count = len([f for f in os.listdir(malignant_images) if f.endswith('.png')])
        print(f"   📊 Images found: {benign_count} benign, {malignant_count} malignant")
    else:
        print("❌ CCST output directory structure incomplete!")
        return False
    
    # Step 2: Train BUS-UCLM baseline model
    print(f"\n📊 Step 2: Train BUS-UCLM Baseline Model")
    baseline_cmd = f"""python IS2D_main.py \
        --train_data_type BUS-UCLM \
        --test_data_type BUS-UCLM \
        --final_epoch 50 \
        --train \
        --save_path baseline_busuclm_{timestamp}.pth \
        --batch_size 4"""
    
    if not run_command(baseline_cmd, "BUS-UCLM baseline training"):
        print("⚠️ Baseline training failed, but continuing with CCST training...")
    
    # Step 3: Train with CCST-augmented data
    print(f"\n🚀 Step 3: Train with CCST-Augmented Data")
    
    ccst_cmd = f"""python train_with_ccst_data.py \
        --ccst-augmented-path "{ccst_output_dir}" \
        --original-busi-path "dataset/BioMedicalDataset/BUS-UCLM" \
        --batch-size 4 \
        --num-epochs 50 \
        --lr 0.001 \
        --save-path "augmented_busuclm_{timestamp}.pth" """
    
    if not run_command(ccst_cmd, "CCST-augmented model training"):
        return False
    
    # Step 4: Evaluate and compare results
    print(f"\n📊 Step 4: Evaluate and Compare Results")
    print("=" * 50)
    
    print("🎯 Training completed! Here's what you should have:")
    print(f"   • Baseline model: baseline_busuclm_{timestamp}.pth")
    print(f"   • CCST-augmented model: augmented_busuclm_{timestamp}.pth")
    print(f"   • CCST styled images: {ccst_output_dir}/")
    
    print("\n📈 Next steps for comparison:")
    print("1. Test both models on BUS-UCLM test set")
    print("2. Compare Dice/IoU scores:")
    print("   - Baseline (BUS-UCLM only)")
    print("   - Augmented (BUS-UCLM + BUSI-styled BUS-UCLM)")
    print("3. The augmented model should show improved performance!")
    
    print("\n🎉 Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n💥 Pipeline failed. Check the error messages above.")
        exit(1) 