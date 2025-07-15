#!/usr/bin/env python3
"""
Simple CCST BUS-UCLM Augmentation Pipeline
==========================================

GOAL: Use BUSI knowledge to improve BUS-UCLM performance (privacy-preserving)

Pipeline:
1. Extract BUSI style statistics (privacy-preserving)
2. Style-transfer BUS-UCLM to BUSI style 
3. Train MADGNet on: original BUS-UCLM + BUSI-styled BUS-UCLM
4. Compare against BUS-UCLM-only baseline
5. Prove improvement!

This uses your existing CCST implementation - much simpler than the federated approach!
"""

import os
import sys
import subprocess
import json
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
    
    print("🎯 Simple CCST BUS-UCLM Augmentation Pipeline")
    print("=" * 60)
    print("GOAL: Prove BUSI improves BUS-UCLM performance (privacy-preserving)")
    print("=" * 60)
    
    # Step 1: Generate CCST-augmented data using existing implementation
    print(f"\n📊 STEP 1: Generate CCST-Augmented Data")
    print("This uses your existing CCST implementation!")
    
    ccst_cmd = f"""python ccst_exact_replication.py \
        --source_dataset "dataset/BioMedicalDataset/BUS-UCLM" \
        --source_csv "train_frame.csv" \
        --target_dataset "dataset/BioMedicalDataset/BUSI" \
        --target_csv "train_frame.csv" \
        --output_dir "ccst_busuclm_results_{timestamp}" \
        --K 1"""
    
    if not run_command(ccst_cmd, "CCST style transfer generation"):
        return
    
    # Step 2: Train BUS-UCLM baseline
    print(f"\n📊 STEP 2: Train BUS-UCLM Baseline Model")
    
    baseline_cmd = f"""python IS2D_main.py \
        --train_data_type BUS-UCLM \
        --test_data_type BUS-UCLM \
        --final_epoch 50 \
        --train \
        --device cuda \
        --save_path baseline_busuclm_{timestamp}.pth \
        --image_size 352 \
        --batch_size 4"""
    
    if not run_command(baseline_cmd, "BUS-UCLM baseline training"):
        return
    
    # Step 3: Create combined dataset script
    print(f"\n📁 STEP 3: Create Combined Dataset")
    
    combine_script = f"""
import os
import shutil
import pandas as pd

# Paths
original_path = "dataset/BioMedicalDataset/BUS-UCLM"
styled_path = "ccst_busuclm_results_{timestamp}/BUS-UCLM-CCST-augmented"
output_path = "dataset/BioMedicalDataset/BUS-UCLM-CCST-Combined_{timestamp}"

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Copy BUS-UCLM structure
for item in ["train_frame.csv", "val_frame.csv", "test_frame.csv"]:
    if os.path.exists(os.path.join(original_path, item)):
        shutil.copy2(os.path.join(original_path, item), os.path.join(output_path, item))

# Copy original images
for class_type in ["benign", "malignant"]:
    for data_type in ["image", "mask"]:
        src = os.path.join(original_path, class_type, data_type)
        dst = os.path.join(output_path, class_type, data_type)
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

# Add styled images to training
if os.path.exists(styled_path):
    train_df = pd.read_csv(os.path.join(output_path, "train_frame.csv"))
    
    for class_type in ["benign", "malignant"]:
        styled_img_dir = os.path.join(styled_path, class_type, "image")
        if os.path.exists(styled_img_dir):
            dst_img = os.path.join(output_path, class_type, "image") 
            dst_mask = os.path.join(output_path, class_type, "mask")
            
            for img_file in os.listdir(styled_img_dir):
                if img_file.startswith("styled_"):
                    # Copy styled image
                    shutil.copy2(os.path.join(styled_img_dir, img_file), 
                               os.path.join(dst_img, img_file))
                    
                    # Copy styled mask
                    mask_file = img_file.replace(".png", "_mask.png")
                    styled_mask_path = os.path.join(styled_path, class_type, "mask", mask_file)
                    if os.path.exists(styled_mask_path):
                        shutil.copy2(styled_mask_path, os.path.join(dst_mask, mask_file))
                        
                        # Add to training CSV
                        new_row = pd.DataFrame({{
                            "image_path": [f"{{class_type}} {{img_file}}"],
                            "mask_path": [f"{{class_type}} {{mask_file}}"],
                            "class": [class_type]
                        }})
                        train_df = pd.concat([train_df, new_row], ignore_index=True)
    
    # Save updated CSV
    train_df.to_csv(os.path.join(output_path, "train_frame.csv"), index=False)
    print(f"Combined dataset created: {{len(train_df)}} training samples")

print("✅ Combined dataset created!")
"""
    
    with open(f"create_combined_{timestamp}.py", "w") as f:
        f.write(combine_script)
    
    if not run_command(f"python create_combined_{timestamp}.py", "Creating combined dataset"):
        return
    
    # Step 4: Train on combined dataset using existing train_with_ccst_data.py
    print(f"\n🚀 STEP 4: Train on Combined Dataset (BUS-UCLM + BUSI-styled)")
    
    # Use the existing CCST training script
    combined_cmd = f"""python train_with_ccst_data.py \
        --ccst-augmented-path "ccst_busuclm_results_{timestamp}/styled_dataset.csv" \
        --original-busi-path "dataset/BioMedicalDataset/BUS-UCLM" \
        --batch-size 4 \
        --num-epochs 50 \
        --lr 0.001 \
        --device cuda \
        --save-path "augmented_busuclm_{timestamp}.pth" """
    
    if not run_command(combined_cmd, "Augmented model training (BUS-UCLM + BUSI-styled)"):
        return
    
    # Step 5: Compare results
    print(f"\n📊 STEP 5: Compare Results")
    print("=" * 50)
    
    results = {
        'timestamp': timestamp,
        'approach': 'CCST_Privacy_Preserving_Style_Transfer',
        'baseline_model': f'baseline_busuclm_{timestamp}.pth',
        'augmented_model': f'augmented_busuclm_{timestamp}.pth',
        'goal': 'Prove BUSI knowledge improves BUS-UCLM performance',
        'ccst_output': f'ccst_busuclm_results_{timestamp}',
        'combined_dataset': f'dataset/BioMedicalDataset/BUS-UCLM-CCST-Combined_{timestamp}'
    }
    
    # Save results summary
    with open(f'ccst_pipeline_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🎉 CCST Pipeline Completed!")
    print("=" * 50)
    print(f"📊 Results Summary:")
    print(f"   🔹 Baseline Model: {results['baseline_model']}")
    print(f"   🔹 Augmented Model: {results['augmented_model']}")
    print(f"   🔹 CCST Output: {results['ccst_output']}")
    print(f"   🔹 Combined Dataset: {results['combined_dataset']}")
    print(f"   🔹 Privacy: Only BUSI statistics used (no raw images shared)")
    print(f"\n📄 Detailed results: ccst_pipeline_results_{timestamp}.json")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Compare model performance metrics from training logs")
    print(f"2. Test both models on BUS-UCLM test set using evaluate_model.py")
    print(f"3. Calculate improvement: (Augmented - Baseline) performance")
    print(f"4. If Dice/IoU improved → MISSION ACCOMPLISHED! 🎉")
    
    print(f"\n🔧 Manual Evaluation Commands:")
    print(f"# Test baseline model:")
    print(f"python evaluate_model.py --model-path {results['baseline_model']} --test-data BUS-UCLM")
    print(f"")
    print(f"# Test augmented model:")
    print(f"python evaluate_model.py --model-path {results['augmented_model']} --test-data BUS-UCLM")
    
    # Cleanup
    os.remove(f"create_combined_{timestamp}.py")

if __name__ == '__main__':
    main() 