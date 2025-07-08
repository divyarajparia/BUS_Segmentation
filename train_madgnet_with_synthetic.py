#!/usr/bin/env python3
"""
Complete MADGNET Training Pipeline with Synthetic Data
====================================================
1. Generate 264 synthetic BUSI samples using trained GAN
2. Train MADGNET on combined dataset (BUSI + synthetic)
3. Test on original BUSI only (fair evaluation)

Usage:
    python train_madgnet_with_synthetic.py --gan_checkpoint checkpoints/busi_gan_final.pth
"""

import os
import sys
import subprocess
import argparse
import torch

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
        return True
    else:
        print(f"❌ {description} failed with exit code {result.returncode}")
        return False

def check_paths():
    """Check that required paths exist"""
    required_paths = [
        'dataset/BioMedicalDataset/BUSI',
        'IS2D_main.py',
        'synthetic_busi_gan.py'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing required files/directories:")
        for path in missing_paths:
            print(f"   - {path}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train MADGNET with synthetic BUSI data')
    parser.add_argument('--gan_checkpoint', type=str, required=True,
                       help='Path to trained GAN checkpoint (e.g., checkpoints/busi_gan_final.pth)')
    parser.add_argument('--synthetic_dir', type=str, default='synthetic_busi_madgnet',
                       help='Directory for synthetic dataset')
    parser.add_argument('--num_benign', type=int, default=175,
                       help='Number of synthetic benign samples')
    parser.add_argument('--num_malignant', type=int, default=89,
                       help='Number of synthetic malignant samples')
    parser.add_argument('--madgnet_epochs', type=int, default=200,
                       help='Number of MADGNET training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip synthetic data generation (use existing)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip MADGNET training (only generate synthetic data)')
    
    args = parser.parse_args()
    
    print("🔬 MADGNET Training with Synthetic BUSI Data")
    print("=" * 70)
    print(f"📁 GAN Checkpoint: {args.gan_checkpoint}")
    print(f"📁 Synthetic Directory: {args.synthetic_dir}")
    print(f"🎯 Synthetic Samples: {args.num_benign} benign + {args.num_malignant} malignant")
    print(f"🧠 MADGNET Epochs: {args.madgnet_epochs}")
    print(f"💻 GPU: {args.gpu}")
    
    # Check prerequisites
    if not check_paths():
        return False
    
    if not args.skip_generation:
        if not os.path.exists(args.gan_checkpoint):
            print(f"❌ GAN checkpoint not found: {args.gan_checkpoint}")
            return False
    
    # Step 1: Generate Synthetic Dataset
    if not args.skip_generation:
        print(f"\n{'='*70}")
        print("🎨 STEP 1: GENERATING SYNTHETIC DATASET")
        print("="*70)
        
        generation_cmd = (
            f"python generate_busi_synthetic_final.py "
            f"--checkpoint {args.gan_checkpoint} "
            f"--output_dir {args.synthetic_dir} "
            f"--num_benign {args.num_benign} "
            f"--num_malignant {args.num_malignant}"
        )
        
        if not run_command(generation_cmd, "Synthetic Data Generation"):
            return False
        
        # Verify synthetic data was created
        synthetic_csv = os.path.join(args.synthetic_dir, 'synthetic_dataset.csv')
        if not os.path.exists(synthetic_csv):
            print(f"❌ Synthetic dataset not created properly: {synthetic_csv}")
            return False
        
        print(f"\n✅ Synthetic dataset ready at: {args.synthetic_dir}")
    else:
        print(f"\n⏭️  Skipping synthetic data generation (using existing: {args.synthetic_dir})")
    
    if args.skip_training:
        print(f"\n⏭️  Skipping MADGNET training as requested")
        return True
    
    # Step 2: Train MADGNET on Combined Dataset
    print(f"\n{'='*70}")
    print("🧠 STEP 2: TRAINING MADGNET ON COMBINED DATASET")
    print("="*70)
    print("📊 Training Data: BUSI training + Synthetic data")
    print("🎯 Test Data: Original BUSI test only (fair evaluation)")
    
    # Set environment variable for GPU
    env_prefix = f"CUDA_VISIBLE_DEVICES={args.gpu}"
    
    training_cmd = (
        f"{env_prefix} python IS2D_main.py "
        f"--num_workers 4 "
        f"--data_path dataset/BioMedicalDataset "
        f"--save_path model_weights "
        f"--train_data_type BUSI-Synthetic-Combined "
        f"--test_data_type BUSI "
        f"--synthetic_data_dir {args.synthetic_dir} "
        f"--final_epoch {args.madgnet_epochs} "
        f"--batch_size {args.batch_size} "
        f"--train"
    )
    
    if not run_command(training_cmd, "MADGNET Training on Combined Dataset"):
        return False
    
    # Step 3: Evaluate on Original BUSI Test Set
    print(f"\n{'='*70}")
    print("📈 STEP 3: EVALUATING ON ORIGINAL BUSI TEST SET")
    print("="*70)
    
    evaluation_cmd = (
        f"{env_prefix} python IS2D_main.py "
        f"--num_workers 4 "
        f"--data_path dataset/BioMedicalDataset "
        f"--save_path model_weights "
        f"--train_data_type BUSI-Synthetic-Combined "
        f"--test_data_type BUSI "
        f"--synthetic_data_dir {args.synthetic_dir} "
        f"--final_epoch {args.madgnet_epochs}"
    )
    
    if not run_command(evaluation_cmd, "Evaluation on Original BUSI Test Set"):
        return False
    
    # Step 4: Summary and Results
    print(f"\n{'='*70}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Find and display results
    model_dir = f"model_weights/BUSI-Synthetic-Combined"
    if os.path.exists(model_dir):
        print(f"\n📁 Model saved in: {model_dir}")
        
        test_reports_dir = os.path.join(model_dir, "test_reports")
        if os.path.exists(test_reports_dir):
            print(f"📊 Test results in: {test_reports_dir}")
            
            # Try to find and display the latest test report
            test_files = [f for f in os.listdir(test_reports_dir) if f.endswith('.txt')]
            if test_files:
                latest_test = sorted(test_files)[-1]
                test_path = os.path.join(test_reports_dir, latest_test)
                
                print(f"\n📈 Latest Test Results ({latest_test}):")
                print("-" * 40)
                try:
                    with open(test_path, 'r') as f:
                        print(f.read())
                except Exception as e:
                    print(f"Could not read test results: {e}")
    
    print(f"\n🔬 Experimental Setup Summary:")
    print(f"   Training Data: BUSI training ({args.num_benign + args.num_malignant} synthetic added)")
    print(f"   Test Data: Original BUSI test only")
    print(f"   Model: MADGNET")
    print(f"   Training Epochs: {args.madgnet_epochs}")
    print(f"   Fair Evaluation: ✅ (no synthetic data in testing)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 