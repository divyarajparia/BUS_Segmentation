#!/usr/bin/env python3
"""
Complete CCST Pipeline: From Data Generation to Model Comparison
Automates the entire workflow:
1. Generate CCST-augmented data
2. Train baseline model (BUSI only)
3. Train CCST model (BUSI + styled BUS-UCLM)
4. Evaluate both models
5. Compare results
"""

import subprocess
import os
import sys
import argparse
import time
from datetime import datetime

def run_command(command, description, check=True):
    """Run a command with error handling"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=False)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.1f}s: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} error after {elapsed:.1f}s: {e}")
        return False

def check_prerequisites():
    """Check if required files exist"""
    required_files = [
        'ccst_exact_replication.py',
        'train_baseline_busi_only.py',
        'train_with_ccst_data.py',
        'evaluate_model.py',
        'compare_results.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def check_datasets(busi_path, bus_uclm_path):
    """Check if datasets exist"""
    if not os.path.exists(busi_path):
        print(f"❌ BUSI dataset not found at {busi_path}")
        return False
    
    if not os.path.exists(bus_uclm_path):
        print(f"❌ BUS-UCLM dataset not found at {bus_uclm_path}")
        return False
    
    return True

def create_output_directories():
    """Create output directories"""
    directories = [
        'logs',
        'models',
        'results',
        'comparison_results'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    
    print(f"✅ Output directories created")

def main():
    parser = argparse.ArgumentParser(description='Run complete CCST pipeline')
    parser.add_argument('--busi-path', type=str,
                       default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset (style source)')
    parser.add_argument('--bus-uclm-path', type=str,
                       default='dataset/BioMedicalDataset/BUS-UCLM',
                       help='Path to BUS-UCLM dataset (content source)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip data generation (if already done)')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline training')
    parser.add_argument('--skip-ccst', action='store_true',
                       help='Skip CCST training')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with reduced epochs for quick testing')
    
    args = parser.parse_args()
    
    # Quick test settings
    if args.quick_test:
        args.epochs = 5
        print("🚀 Quick test mode: Running with 5 epochs")
    
    # Print configuration
    print("🎯 CCST Complete Pipeline")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  BUSI path (style): {args.busi_path}")
    print(f"  BUS-UCLM path (content): {args.bus_uclm_path}")
    print(f"  Direction: Extract BUSI style → Apply to BUS-UCLM")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Start time: {datetime.now()}")
    
    # Check prerequisites
    print(f"\n🔍 Checking prerequisites...")
    if not check_prerequisites():
        return
    
    if not check_datasets(args.busi_path, args.bus_uclm_path):
        return
    
    # Create output directories
    create_output_directories()
    
    # Define paths
    ccst_augmented_path = "dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented"
    baseline_model_path = "models/baseline_busi_only_model.pth"
    ccst_model_path = "models/ccst_domain_adaptation_model.pth"
    baseline_results_path = "results/baseline_results.json"
    ccst_results_path = "results/ccst_results.json"
    
    # Pipeline execution
    success = True
    
    # Step 1: Generate CCST-augmented data
    if not args.skip_generation:
        if not run_command(
            f"python ccst_exact_replication.py "
            f"--source_dataset '{args.bus_uclm_path}' "
            f"--target_dataset '{args.busi_path}'",
            "Step 1: Generating CCST-augmented data (BUSI style → BUS-UCLM content)"
        ):
            success = False
    else:
        print("⏭️  Skipping data generation")
    
    # Step 2: Train baseline model
    if success and not args.skip_baseline:
        if not run_command(
            f"python train_baseline_busi_only.py "
            f"--busi-path '{args.busi_path}' "
            f"--batch-size {args.batch_size} "
            f"--num-epochs {args.epochs} "
            f"--lr {args.lr} "
            f"--device {args.device} "
            f"--save-path '{baseline_model_path}' "
            f"--results-path '{baseline_results_path}'",
            "Step 2: Training baseline model (BUSI only)"
        ):
            success = False
    else:
        print("⏭️  Skipping baseline training")
    
    # Step 3: Train CCST model
    if success and not args.skip_ccst:
        if not run_command(
            f"python train_with_ccst_data.py "
            f"--ccst-augmented-path '{ccst_augmented_path}' "
            f"--original-busi-path '{args.busi_path}' "
            f"--batch-size {args.batch_size} "
            f"--num-epochs {args.epochs} "
            f"--lr {args.lr} "
            f"--device {args.device} "
            f"--save-path '{ccst_model_path}' "
            f"--results-path '{ccst_results_path}'",
            "Step 3: Training CCST model (BUSI + styled BUS-UCLM)"
        ):
            success = False
    else:
        print("⏭️  Skipping CCST training")
    
    # Step 4: Evaluate baseline model (if not already done)
    if success and not os.path.exists(baseline_results_path):
        if not run_command(
            f"python evaluate_model.py "
            f"--model-path '{baseline_model_path}' "
            f"--dataset-path '{args.busi_path}' "
            f"--dataset-type busi "
            f"--mode test "
            f"--batch-size {args.batch_size} "
            f"--device {args.device} "
            f"--output-file '{baseline_results_path}' "
            f"--output-dir 'results/baseline_evaluation'",
            "Step 4: Evaluating baseline model"
        ):
            success = False
    
    # Step 5: Evaluate CCST model (if not already done)
    if success and not os.path.exists(ccst_results_path):
        if not run_command(
            f"python evaluate_model.py "
            f"--model-path '{ccst_model_path}' "
            f"--dataset-path '{args.busi_path}' "
            f"--dataset-type busi "
            f"--mode test "
            f"--batch-size {args.batch_size} "
            f"--device {args.device} "
            f"--output-file '{ccst_results_path}' "
            f"--output-dir 'results/ccst_evaluation'",
            "Step 5: Evaluating CCST model"
        ):
            success = False
    
    # Step 6: Compare results
    if success:
        if not run_command(
            f"python compare_results.py "
            f"--baseline-results '{baseline_results_path}' "
            f"--ccst-results '{ccst_results_path}' "
            f"--output-dir 'comparison_results' "
            f"--output-file 'comparison_results/comparison.json'",
            "Step 6: Comparing results"
        ):
            success = False
    
    # Final summary
    end_time = datetime.now()
    print(f"\n🎉 Pipeline Summary")
    print("=" * 50)
    
    if success:
        print("✅ Complete CCST pipeline executed successfully!")
        print(f"\n📁 Results:")
        print(f"  Baseline model: {baseline_model_path}")
        print(f"  CCST model: {ccst_model_path}")
        print(f"  Baseline results: {baseline_results_path}")
        print(f"  CCST results: {ccst_results_path}")
        print(f"  Comparison: comparison_results/")
        
        print(f"\n📊 Key Files Generated:")
        print(f"  - CCST-augmented data: {ccst_augmented_path}")
        print(f"  - Comparison plots: comparison_results/")
        print(f"  - Evaluation plots: results/*/")
        
        print(f"\n🔍 Next Steps:")
        print(f"  1. Check comparison_results/ for detailed comparison")
        print(f"  2. Review training logs in models/ directory")
        print(f"  3. Examine evaluation plots in results/ directory")
        
    else:
        print("❌ Pipeline failed. Check logs above for details.")
        print(f"\n🔧 Troubleshooting:")
        print(f"  - Check that all datasets exist")
        print(f"  - Verify sufficient disk space")
        print(f"  - Ensure GPU availability if using CUDA")
        print(f"  - Check individual script logs for specific errors")
    
    print(f"\n⏱️  Total execution time: {end_time}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 