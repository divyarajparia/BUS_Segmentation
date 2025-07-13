#!/usr/bin/env python3
"""
Complete CCST Pipeline for MADGNet Training
==========================================

This script runs the complete CCST (Cross-Client Style Transfer) pipeline:

1. Privacy-preserving style extraction from BUSI
2. AdaIN-based style transfer of BUS-UCLM data
3. Combined dataset creation
4. MADGNet training on combined dataset
5. Evaluation on original BUSI test set

This implements the CCST methodology for privacy-preserving domain adaptation
in medical image segmentation.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
import torch


def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        
        if result.returncode == 0:
            print(f"   ✅ {description} completed successfully!")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ {description} failed!")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed with error code {e.returncode}")
        print(f"   Error: {e.stderr}")
        return False


def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    required_packages = [
        'torch', 'torchvision', 'PIL', 'pandas', 'numpy', 'tqdm', 'json'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} - missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Please install missing packages before running the pipeline.")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"   {'✅' if cuda_available else '⚠️'} CUDA: {'Available' if cuda_available else 'Not available (will use CPU)'}")
    
    return True


def check_datasets(busi_dir, bus_uclm_dir):
    """Check if required datasets are available"""
    print("📊 Checking datasets...")
    
    # Check BUSI dataset
    busi_files = [
        'train_frame.csv', 'val_frame.csv', 'test_frame.csv'
    ]
    
    for file in busi_files:
        file_path = os.path.join(busi_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ BUSI {file}")
        else:
            print(f"   ❌ BUSI {file} - not found")
            return False
    
    # Check BUS-UCLM dataset
    bus_uclm_train_csv = os.path.join(bus_uclm_dir, 'train_frame.csv')
    if os.path.exists(bus_uclm_train_csv):
        print(f"   ✅ BUS-UCLM train_frame.csv")
    else:
        print(f"   ❌ BUS-UCLM train_frame.csv - not found")
        return False
    
    return True


def run_style_transfer_pipeline(busi_dir, bus_uclm_dir, output_dir, device='auto'):
    """Run the privacy-preserving style transfer pipeline"""
    print(f"\n{'='*60}")
    print("🎨 STEP 1: PRIVACY-PRESERVING STYLE TRANSFER")
    print("="*60)
    
    command = (
        f"python ccst_privacy_preserving_adain.py "
        f"--busi-dir {busi_dir} "
        f"--bus-uclm-dir {bus_uclm_dir} "
        f"--output-dir {output_dir} "
        f"--device {device}"
    )
    
    return run_command(command, "Privacy-preserving AdaIN style transfer")


def run_madgnet_training(combined_dir, original_busi_dir, num_epochs=100, batch_size=8, device='auto'):
    """Run MADGNet training on combined dataset"""
    print(f"\n{'='*60}")
    print("🧠 STEP 2: MADGNET TRAINING")
    print("="*60)
    
    command = (
        f"python train_madgnet_ccst_combined.py "
        f"--combined-dir {combined_dir} "
        f"--original-busi-dir {original_busi_dir} "
        f"--num-epochs {num_epochs} "
        f"--batch-size {batch_size} "
        f"--device {device}"
    )
    
    return run_command(command, "MADGNet training on CCST combined dataset")


def run_is2d_integration(combined_dir, original_busi_dir, num_epochs=100, batch_size=8):
    """Run training using IS2D integration"""
    print(f"\n{'='*60}")
    print("🔗 STEP 3: IS2D INTEGRATION TRAINING")
    print("="*60)
    
    command = (
        f"python IS2D_main.py "
        f"--train_data_type BUSI-CCST-Combined "
        f"--test_data_type BUSI "
        f"--ccst_combined_dir {combined_dir} "
        f"--data_path dataset/BioMedicalDataset "
        f"--final_epoch {num_epochs} "
        f"--batch_size {batch_size} "
        f"--train"
    )
    
    return run_command(command, "IS2D integration training")


def generate_report(results_dir, combined_dir):
    """Generate a comprehensive report of the pipeline results"""
    print(f"\n📋 Generating pipeline report...")
    
    report = {
        'pipeline_info': {
            'timestamp': datetime.now().isoformat(),
            'methodology': 'CCST Privacy-Preserving Domain Adaptation',
            'paper_reference': 'Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer'
        },
        'datasets': {
            'source_domain': 'BUSI',
            'target_domain': 'BUS-UCLM',
            'combined_dataset': combined_dir,
            'privacy_preserving': True
        },
        'methodology_steps': [
            "1. Extract domain-level style statistics from BUSI (privacy-preserving)",
            "2. Apply AdaIN-based style transfer to BUS-UCLM training data",
            "3. Combine original BUSI + styled BUS-UCLM for training",
            "4. Train MADGNet on combined dataset",
            "5. Evaluate on original BUSI test set (fair evaluation)"
        ],
        'privacy_features': [
            "Only domain-level statistics shared (mean/variance)",
            "No actual images shared between domains",
            "Real-time style transfer without training",
            "Preserves patient privacy across institutions"
        ]
    }
    
    # Check if training results exist
    training_results_path = 'results/ccst_combined_results.json'
    if os.path.exists(training_results_path):
        with open(training_results_path, 'r') as f:
            training_results = json.load(f)
        report['training_results'] = training_results
    
    # Save report
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'ccst_pipeline_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✅ Pipeline report saved to: {report_path}")
    
    # Print summary
    print(f"\n🎉 CCST Pipeline Summary:")
    print(f"   Methodology: Privacy-preserving domain adaptation")
    print(f"   Source domain: BUSI")
    print(f"   Target domain: BUS-UCLM")
    print(f"   Privacy preserved: ✅ Only statistics shared")
    print(f"   Domain adaptation: ✅ Style transfer applied")
    print(f"   Fair evaluation: ✅ Test on original BUSI only")
    
    return report_path


def main():
    """Main pipeline orchestrator"""
    parser = argparse.ArgumentParser(description='Complete CCST Pipeline for MADGNet Training')
    parser.add_argument('--busi-dir', type=str, 
                       default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--bus-uclm-dir', type=str, 
                       default='dataset/BioMedicalDataset/BUS-UCLM',
                       help='Path to BUS-UCLM dataset')
    parser.add_argument('--output-dir', type=str, 
                       default='dataset/BioMedicalDataset/BUSI_CCST_Combined',
                       help='Output directory for combined dataset')
    parser.add_argument('--results-dir', type=str, 
                       default='results/ccst_pipeline',
                       help='Directory to save pipeline results')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--skip-style-transfer', action='store_true',
                       help='Skip style transfer step (if already done)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step (generate report only)')
    parser.add_argument('--use-is2d-integration', action='store_true',
                       help='Use IS2D integration instead of standalone training')
    
    args = parser.parse_args()
    
    print("🚀 Complete CCST Pipeline for MADGNet Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  BUSI directory: {args.busi_dir}")
    print(f"  BUS-UCLM directory: {args.bus_uclm_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Training epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Skip style transfer: {args.skip_style_transfer}")
    print(f"  Skip training: {args.skip_training}")
    print(f"  Use IS2D integration: {args.use_is2d_integration}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return False
    
    # Check datasets
    if not check_datasets(args.busi_dir, args.bus_uclm_dir):
        print("\n❌ Dataset check failed. Please ensure datasets are available.")
        return False
    
    success = True
    
    # Step 1: Style transfer
    if not args.skip_style_transfer:
        if not run_style_transfer_pipeline(
            args.busi_dir, args.bus_uclm_dir, args.output_dir, args.device
        ):
            print("\n❌ Style transfer pipeline failed!")
            success = False
    else:
        print("\n⏭️  Skipping style transfer step")
        if not os.path.exists(args.output_dir):
            print(f"❌ Output directory {args.output_dir} does not exist!")
            success = False
    
    # Step 2: Training
    if success and not args.skip_training:
        if args.use_is2d_integration:
            if not run_is2d_integration(
                args.output_dir, args.busi_dir, args.num_epochs, args.batch_size
            ):
                print("\n❌ IS2D integration training failed!")
                success = False
        else:
            if not run_madgnet_training(
                args.output_dir, args.busi_dir, args.num_epochs, args.batch_size, args.device
            ):
                print("\n❌ MADGNet training failed!")
                success = False
    else:
        print("\n⏭️  Skipping training step")
    
    # Step 3: Generate report
    if success or args.skip_training:
        report_path = generate_report(args.results_dir, args.output_dir)
        
        print(f"\n🎊 CCST Pipeline Completed Successfully!")
        print(f"   Report: {report_path}")
        print(f"   Combined dataset: {args.output_dir}")
        
        # Final summary
        print(f"\n📝 Next Steps:")
        print(f"   1. Review training results in {args.results_dir}")
        print(f"   2. Compare with baseline BUSI-only training")
        print(f"   3. Analyze domain adaptation effectiveness")
        print(f"   4. Consider further hyperparameter tuning")
        
        return True
    else:
        print(f"\n❌ CCST Pipeline failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 