#!/usr/bin/env python3
"""
Complete CCST Pipeline - Proper Implementation
Uses the IS2D infrastructure for training
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path

def check_dataset_exists(path, name):
    """Check if dataset exists"""
    if not os.path.exists(path):
        print(f"❌ {name} dataset not found at {path}")
        return False
    print(f"✅ {name} dataset found at {path}")
    return True

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully")
        if result.stdout:
            print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print("Error output:", e.stderr[:500] + "..." if len(e.stderr) > 500 else e.stderr)
        return False

def load_json_results(filepath):
    """Load JSON results file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def compare_results(baseline_results, ccst_results):
    """Compare baseline vs CCST results"""
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    if not baseline_results or not ccst_results:
        print("❌ Cannot compare results - missing result files")
        return
    
    metrics = ['dice', 'iou', 'hausdorff']
    
    for metric in metrics:
        if metric in baseline_results and metric in ccst_results:
            baseline_mean = baseline_results[metric]['mean']
            ccst_mean = ccst_results[metric]['mean']
            
            if metric == 'hausdorff':
                # Lower is better for Hausdorff
                improvement = ((baseline_mean - ccst_mean) / baseline_mean) * 100
                better = "✅" if improvement > 0 else "❌"
            else:
                # Higher is better for Dice and IoU
                improvement = ((ccst_mean - baseline_mean) / baseline_mean) * 100
                better = "✅" if improvement > 0 else "❌"
            
            print(f"{metric.upper()}:")
            print(f"  Baseline: {baseline_mean:.4f} ± {baseline_results[metric]['std']:.4f}")
            print(f"  CCST:     {ccst_mean:.4f} ± {ccst_results[metric]['std']:.4f}")
            print(f"  Change:   {improvement:+.2f}% {better}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Complete CCST Pipeline - Proper Implementation')
    parser.add_argument('--busi-path', type=str, default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--bus-uclm-path', type=str, default='dataset/BioMedicalDataset/BUS-UCLM',
                       help='Path to BUS-UCLM dataset')
    parser.add_argument('--ccst-output', type=str, default='ccst_output',
                       help='CCST output directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--skip-ccst', action='store_true', help='Skip CCST generation')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline training')
    parser.add_argument('--skip-ccst-training', action='store_true', help='Skip CCST training')
    
    args = parser.parse_args()
    
    print("🚀 Complete CCST Pipeline - Proper Implementation")
    print("=" * 60)
    print(f"BUSI Dataset: {args.busi_path}")
    print(f"BUS-UCLM Dataset: {args.bus_uclm_path}")
    print(f"CCST Output: {args.ccst_output}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)
    
    # Check if datasets exist
    if not check_dataset_exists(args.busi_path, "BUSI"):
        return
    if not check_dataset_exists(args.bus_uclm_path, "BUS-UCLM"):
        return
    
    # Step 1: Generate CCST data
    if not args.skip_ccst:
        print("\n🎯 Step 1: Generating CCST Data")
        ccst_cmd = [
            'python', 'ccst_exact_replication.py',
            '--busi-path', args.busi_path,
            '--bus-uclm-path', args.bus_uclm_path,
            '--output-dir', args.ccst_output
        ]
        
        if not run_command(ccst_cmd, "Generating CCST style-transferred data"):
            print("❌ CCST generation failed. Stopping pipeline.")
            return
    else:
        print("⏭️  Skipping CCST generation")
    
    # Step 2: Train baseline model (BUSI only)
    if not args.skip_baseline:
        print("\n🎯 Step 2: Training Baseline Model (BUSI Only)")
        baseline_cmd = [
            'python', 'train_busi_baseline_proper.py',
            '--busi-path', args.busi_path,
            '--batch-size', str(args.batch_size),
            '--final-epoch', str(args.epochs),
            '--lr', str(args.lr),
            '--num-workers', str(args.num_workers),
            '--save-path', 'model_weights_baseline'
        ]
        
        if not run_command(baseline_cmd, "Training baseline model"):
            print("❌ Baseline training failed. Stopping pipeline.")
            return
    else:
        print("⏭️  Skipping baseline training")
    
    # Step 3: Train CCST model (BUSI + CCST-augmented BUS-UCLM)
    if not args.skip_ccst_training:
        print("\n🎯 Step 3: Training CCST Model")
        ccst_train_cmd = [
            'python', 'train_ccst_proper.py',
            '--busi-path', args.busi_path,
            '--ccst-path', args.ccst_output,
            '--batch-size', str(args.batch_size),
            '--final-epoch', str(args.epochs),
            '--lr', str(args.lr),
            '--num-workers', str(args.num_workers),
            '--save-path', 'model_weights_ccst'
        ]
        
        if not run_command(ccst_train_cmd, "Training CCST model"):
            print("❌ CCST training failed. Stopping pipeline.")
            return
    else:
        print("⏭️  Skipping CCST training")
    
    # Step 4: Compare results
    print("\n🎯 Step 4: Comparing Results")
    
    baseline_results_path = 'model_weights_baseline/test_results.json'
    ccst_results_path = 'model_weights_ccst/ccst_test_results.json'
    
    baseline_results = load_json_results(baseline_results_path)
    ccst_results = load_json_results(ccst_results_path)
    
    if baseline_results and ccst_results:
        compare_results(baseline_results, ccst_results)
        
        # Save comparison results
        comparison = {
            'baseline': baseline_results,
            'ccst': ccst_results,
            'comparison': {
                'dice_improvement': ((ccst_results['dice']['mean'] - baseline_results['dice']['mean']) / baseline_results['dice']['mean']) * 100,
                'iou_improvement': ((ccst_results['iou']['mean'] - baseline_results['iou']['mean']) / baseline_results['iou']['mean']) * 100,
                'hausdorff_improvement': ((baseline_results['hausdorff']['mean'] - ccst_results['hausdorff']['mean']) / baseline_results['hausdorff']['mean']) * 100
            }
        }
        
        with open('ccst_comparison_results.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("📊 Comparison results saved to ccst_comparison_results.json")
    
    print("\n🎉 Pipeline Complete!")
    print("=" * 60)
    print("Summary:")
    print("✅ CCST style transfer completed")
    print("✅ Baseline model training completed")
    print("✅ CCST model training completed")
    print("✅ Results comparison completed")
    
    if baseline_results and ccst_results:
        dice_improvement = ((ccst_results['dice']['mean'] - baseline_results['dice']['mean']) / baseline_results['dice']['mean']) * 100
        iou_improvement = ((ccst_results['iou']['mean'] - baseline_results['iou']['mean']) / baseline_results['iou']['mean']) * 100
        hausdorff_improvement = ((baseline_results['hausdorff']['mean'] - ccst_results['hausdorff']['mean']) / baseline_results['hausdorff']['mean']) * 100
        
        print(f"\n📈 Final Improvements:")
        print(f"  Dice: {dice_improvement:+.2f}%")
        print(f"  IoU: {iou_improvement:+.2f}%")
        print(f"  Hausdorff: {hausdorff_improvement:+.2f}%")
        
        if dice_improvement > 0 and iou_improvement > 0 and hausdorff_improvement > 0:
            print("🎯 CCST shows improvement across all metrics!")
        else:
            print("⚠️  Mixed results - some metrics improved, others didn't")

if __name__ == "__main__":
    main() 