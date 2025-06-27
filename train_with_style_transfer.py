#!/usr/bin/env python3
"""
Complete pipeline for style transfer and training:
1. Train CycleGAN for style transfer
2. Apply style transfer to BUS-UCLM
3. Create combined dataset
4. Train MADGNet on combined dataset
5. Evaluate performance
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['torch', 'torchvision', 'pandas', 'numpy', 'PIL', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_datasets():
    """Check if required datasets exist"""
    required_paths = [
        "dataset/BioMedicalDataset/BUSI",
        "dataset/BioMedicalDataset/BUS-UCLM"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ Missing dataset paths: {missing_paths}")
        print("Please ensure the datasets are properly organized")
        return False
    
    print("✅ All required datasets found")
    return True

def main():
    parser = argparse.ArgumentParser(description='Complete style transfer and training pipeline')
    parser.add_argument('--skip_style_transfer', action='store_true', 
                       help='Skip style transfer training (use existing model)')
    parser.add_argument('--style_transfer_epochs', type=int, default=50,
                       help='Number of epochs for style transfer training')
    parser.add_argument('--madgnet_epochs', type=int, default=200,
                       help='Number of epochs for MADGNet training')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    print("🚀 Starting Style Transfer and Training Pipeline")
    print(f"Configuration:")
    print(f"  - Style Transfer Epochs: {args.style_transfer_epochs}")
    print(f"  - MADGNet Epochs: {args.madgnet_epochs}")
    print(f"  - GPU: {args.gpu}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Skip Style Transfer: {args.skip_style_transfer}")
    
    # Check dependencies and datasets
    if not check_dependencies():
        return False
    
    if not check_datasets():
        return False
    
    # Step 1: Train CycleGAN for style transfer
    if not args.skip_style_transfer:
        print("\n📊 Step 1: Training CycleGAN for style transfer...")
        style_transfer_cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} python style_transfer.py"
        if not run_command(style_transfer_cmd, "Style Transfer Training"):
            print("❌ Style transfer training failed")
            return False
    else:
        print("\n⏭️  Skipping style transfer training (using existing model)")
    
    # Step 2: Apply style transfer to BUS-UCLM
    print("\n🎨 Step 2: Applying style transfer to BUS-UCLM...")
    apply_style_cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} python apply_style_transfer.py"
    if not run_command(apply_style_cmd, "Style Transfer Application"):
        print("❌ Style transfer application failed")
        return False
    
    # Step 3: Train MADGNet on combined dataset
    print("\n🧠 Step 3: Training MADGNet on combined dataset...")
    madgnet_cmd = (f"CUDA_VISIBLE_DEVICES={args.gpu} python IS2D_main.py "
                   f"--train_data_type BUSI-Combined "
                   f"--test_data_type BUSI "
                   f"--train "
                   f"--final_epoch {args.madgnet_epochs} "
                   f"--num_workers 4 "
                   f"--data_path dataset/BioMedicalDataset "
                   f"--save_path model_weights")
    
    if not run_command(madgnet_cmd, "MADGNet Training on Combined Dataset"):
        print("❌ MADGNet training failed")
        return False
    
    # Step 4: Evaluate on BUSI
    print("\n📈 Step 4: Evaluating on BUSI dataset...")
    eval_busi_cmd = (f"CUDA_VISIBLE_DEVICES={args.gpu} python IS2D_main.py "
                     f"--train_data_type BUSI-Combined "
                     f"--test_data_type BUSI "
                     f"--final_epoch {args.madgnet_epochs} "
                     f"--num_workers 4 "
                     f"--data_path dataset/BioMedicalDataset "
                     f"--save_path model_weights")
    
    if not run_command(eval_busi_cmd, "Evaluation on BUSI"):
        print("❌ BUSI evaluation failed")
        return False
    
    # Step 5: Evaluate on BUS-UCLM
    print("\n📈 Step 5: Evaluating on BUS-UCLM dataset...")
    eval_bus_uclm_cmd = (f"CUDA_VISIBLE_DEVICES={args.gpu} python IS2D_main.py "
                         f"--train_data_type BUSI-Combined "
                         f"--test_data_type BUS-UCLM "
                         f"--final_epoch {args.madgnet_epochs} "
                         f"--num_workers 4 "
                         f"--data_path dataset/BioMedicalDataset "
                         f"--save_path model_weights")
    
    if not run_command(eval_bus_uclm_cmd, "Evaluation on BUS-UCLM"):
        print("❌ BUS-UCLM evaluation failed")
        return False
    
    print("\n🎉 Pipeline completed successfully!")
    print("\n📊 Results Summary:")
    print("  - Style transfer model trained and applied")
    print("  - Combined dataset created (BUSI + style-transferred BUS-UCLM)")
    print("  - MADGNet trained on combined dataset")
    print("  - Evaluated on both BUSI and BUS-UCLM")
    print("\n📁 Output files:")
    print("  - Style transfer model: style_transfer_epoch_*.pth")
    print("  - Combined dataset: dataset/BioMedicalDataset/BUSI-Combined/")
    print("  - MADGNet model: model_weights/BUSI-Combined/")
    print("  - Test results: model_weights/BUSI-Combined/test_reports/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 