#!/usr/bin/env python3
"""
Training Script for Reverse BUS-UCLM Style Transfer

This script:
1. Runs the reverse style transfer pipeline
2. Trains on combined dataset (BUS-UCLM + styled BUSI)
3. Tests on BUS-UCLM test set
4. Evaluates cross-domain performance
"""

import os
import subprocess
import sys
import argparse
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("-" * 80)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def check_dependencies():
    """Check if required datasets exist"""
    required_paths = [
        'dataset/BioMedicalDataset/BUS-UCLM',
        'dataset/BioMedicalDataset/BUSI',
        'IS2D_main.py',
        'reverse_style_transfer_pipeline.py'
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"❌ Missing required path: {path}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Train with Reverse BUS-UCLM Style Transfer")
    parser.add_argument('--bus-uclm-path', type=str, default='dataset/BioMedicalDataset/BUS-UCLM',
                        help='Path to BUS-UCLM dataset')
    parser.add_argument('--busi-path', type=str, default='dataset/BioMedicalDataset/BUSI',
                        help='Path to BUSI dataset')
    parser.add_argument('--output-dir', type=str, default='dataset/BioMedicalDataset/Reverse-BUS-UCLM-Style',
                        help='Output directory for style transfer results')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for style transfer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--skip-style-transfer', action='store_true', 
                        help='Skip style transfer if already done')
    
    args = parser.parse_args()
    
    print("🚀 Starting Reverse BUS-UCLM Style Transfer Training")
    print("=" * 80)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 BUS-UCLM Dataset: {args.bus_uclm_path}")
    print(f"📁 BUSI Dataset: {args.busi_path}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"🔄 Training Epochs: {args.epochs}")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please check your setup.")
        return False
    
    # Step 1: Run reverse style transfer pipeline
    if not args.skip_style_transfer:
        print("\n🎨 Step 1: Running Reverse Style Transfer Pipeline...")
        style_transfer_cmd = (
            f"python reverse_style_transfer_pipeline.py "
            f"--bus-uclm-path {args.bus_uclm_path} "
            f"--busi-path {args.busi_path} "
            f"--output-dir {args.output_dir} "
            f"--batch-size {args.batch_size}"
        )
        
        if not run_command(style_transfer_cmd, "Reverse Style Transfer Pipeline"):
            print("❌ Style transfer pipeline failed")
            return False
    else:
        print("⏭️  Skipping style transfer (--skip-style-transfer flag)")
    
    # Step 2: Train model with combined dataset
    print("\n🏋️ Step 2: Training model with combined dataset...")
    train_cmd = (
        f"python IS2D_main.py "
        f"--train_data_type BUS-UCLM "
        f"--test_data_type BUS-UCLM "
        f"--ccst_augmented_path {args.output_dir} "
        f"--train "
        f"--final_epoch {args.epochs}"
    )
    
    if not run_command(train_cmd, "Training with reverse style transfer"):
        print("❌ Training failed")
        return False
    
    # Step 3: Additional evaluation on BUSI for cross-domain analysis
    print("\n📊 Step 3: Cross-domain evaluation on BUSI...")
    eval_cmd = (
        f"python IS2D_main.py "
        f"--train_data_type BUS-UCLM "
        f"--test_data_type BUSI "
        f"--ccst_augmented_path {args.output_dir} "
        f"--test "
        f"--final_epoch {args.epochs}"
    )
    
    if not run_command(eval_cmd, "Cross-domain evaluation on BUSI"):
        print("⚠️  Cross-domain evaluation failed (not critical)")
    
    # Step 4: Generate summary report
    print("\n📈 Step 4: Generating summary report...")
    
    # Look for result files
    result_files = []
    model_weights_dir = "model_weights/final_model_weights/BUS-UCLM"
    
    if os.path.exists(model_weights_dir):
        test_reports_dir = os.path.join(model_weights_dir, "test_reports")
        if os.path.exists(test_reports_dir):
            for file in os.listdir(test_reports_dir):
                if file.endswith('.txt'):
                    result_files.append(os.path.join(test_reports_dir, file))
    
    print("\n📋 Training Summary:")
    print("=" * 80)
    print(f"🎯 Approach: Reverse BUS-UCLM Style Transfer")
    print(f"🔄 Training Data: BUS-UCLM + BUSI-styled-to-BUS-UCLM")
    print(f"🧪 Test Data: BUS-UCLM test set")
    print(f"📊 Output Directory: {args.output_dir}")
    print(f"📈 Training Epochs: {args.epochs}")
    
    if result_files:
        print(f"\n📋 Result Files:")
        for file in result_files:
            print(f"  📄 {file}")
    
    print("\n✅ Reverse BUS-UCLM Style Transfer Training Complete!")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 