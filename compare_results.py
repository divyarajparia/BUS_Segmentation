#!/usr/bin/env python3
"""
Compare results between baseline and CCST models
Generate comprehensive comparison report with improvement metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_improvement(baseline_value, ccst_value):
    """Calculate improvement percentage"""
    if baseline_value == 0:
        return 0
    return ((ccst_value - baseline_value) / baseline_value) * 100

def create_comparison_table(baseline_results, ccst_results):
    """Create comparison table"""
    
    baseline_seg = baseline_results['segmentation_metrics']
    ccst_seg = ccst_results['segmentation_metrics']
    
    comparison = {
        'dice': {
            'baseline': baseline_seg['dice']['mean'],
            'ccst': ccst_seg['dice']['mean'],
            'improvement': calculate_improvement(baseline_seg['dice']['mean'], ccst_seg['dice']['mean'])
        },
        'iou': {
            'baseline': baseline_seg['iou']['mean'],
            'ccst': ccst_seg['iou']['mean'],
            'improvement': calculate_improvement(baseline_seg['iou']['mean'], ccst_seg['iou']['mean'])
        },
        'hausdorff': {
            'baseline': baseline_seg['hausdorff']['mean'],
            'ccst': ccst_seg['hausdorff']['mean'],
            'improvement': calculate_improvement(baseline_seg['hausdorff']['mean'], ccst_seg['hausdorff']['mean'])
        },
        'sensitivity': {
            'baseline': baseline_seg['sensitivity']['mean'],
            'ccst': ccst_seg['sensitivity']['mean'],
            'improvement': calculate_improvement(baseline_seg['sensitivity']['mean'], ccst_seg['sensitivity']['mean'])
        },
        'specificity': {
            'baseline': baseline_seg['specificity']['mean'],
            'ccst': ccst_seg['specificity']['mean'],
            'improvement': calculate_improvement(baseline_seg['specificity']['mean'], ccst_seg['specificity']['mean'])
        }
    }
    
    # Classification metrics
    baseline_cls = baseline_results['classification_metrics']
    ccst_cls = ccst_results['classification_metrics']
    
    comparison['accuracy'] = {
        'baseline': baseline_cls['accuracy'],
        'ccst': ccst_cls['accuracy'],
        'improvement': calculate_improvement(baseline_cls['accuracy'], ccst_cls['accuracy'])
    }
    
    return comparison

def print_comparison_report(comparison, baseline_results, ccst_results):
    """Print comprehensive comparison report"""
    
    print("🎯 CCST vs Baseline Comparison Report")
    print("=" * 60)
    
    # Dataset information
    print(f"📊 Dataset Information:")
    print(f"  Baseline samples: {baseline_results['total_samples']}")
    print(f"  CCST samples: {ccst_results['total_samples']}")
    
    # Segmentation metrics comparison
    print(f"\n📈 Segmentation Metrics Comparison:")
    print(f"{'Metric':<15} {'Baseline':<12} {'CCST':<12} {'Improvement':<15}")
    print("-" * 60)
    
    for metric, values in comparison.items():
        if metric in ['dice', 'iou', 'hausdorff', 'sensitivity', 'specificity']:
            baseline_val = values['baseline']
            ccst_val = values['ccst']
            improvement = values['improvement']
            
            print(f"{metric.title():<15} {baseline_val:<12.4f} {ccst_val:<12.4f} {improvement:<15.2f}%")
    
    # Classification metrics
    print(f"\n🎯 Classification Metrics:")
    acc_baseline = comparison['accuracy']['baseline']
    acc_ccst = comparison['accuracy']['ccst']
    acc_improvement = comparison['accuracy']['improvement']
    
    print(f"{'Accuracy':<15} {acc_baseline:<12.4f} {acc_ccst:<12.4f} {acc_improvement:<15.2f}%")
    
    # Summary
    print(f"\n📋 Summary:")
    improvements = [v['improvement'] for v in comparison.values()]
    positive_improvements = [imp for imp in improvements if imp > 0]
    negative_improvements = [imp for imp in improvements if imp < 0]
    
    print(f"  Metrics improved: {len(positive_improvements)}/{len(improvements)}")
    print(f"  Average improvement: {np.mean(improvements):.2f}%")
    print(f"  Best improvement: {max(improvements):.2f}%")
    if negative_improvements:
        print(f"  Worst decline: {min(improvements):.2f}%")
    
    # Significance
    print(f"\n🔍 Statistical Significance:")
    for metric in ['dice', 'iou', 'hausdorff', 'sensitivity', 'specificity']:
        baseline_values = baseline_results['segmentation_metrics'][metric]['values']
        ccst_values = ccst_results['segmentation_metrics'][metric]['values']
        
        # Simple t-test approximation
        baseline_mean = np.mean(baseline_values)
        ccst_mean = np.mean(ccst_values)
        baseline_std = np.std(baseline_values)
        ccst_std = np.std(ccst_values)
        
        improvement = ((ccst_mean - baseline_mean) / baseline_mean) * 100
        
        # Determine significance based on effect size
        effect_size = abs(ccst_mean - baseline_mean) / np.sqrt((baseline_std**2 + ccst_std**2) / 2)
        
        if effect_size > 0.8:
            significance = "Large"
        elif effect_size > 0.5:
            significance = "Medium"
        elif effect_size > 0.2:
            significance = "Small"
        else:
            significance = "Negligible"
        
        print(f"  {metric.title()}: {improvement:.2f}% ({significance} effect)")

def create_comparison_plots(comparison, baseline_results, ccst_results, output_dir):
    """Create comparison plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Metrics comparison bar plot
    metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'accuracy']
    baseline_values = [comparison[m]['baseline'] for m in metrics]
    ccst_values = [comparison[m]['ccst'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, ccst_values, width, label='CCST', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Baseline vs CCST Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Improvement percentages
    improvements = [comparison[m]['improvement'] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(metrics, improvements, color=colors, alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('CCST Improvement over Baseline')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%', ha='center', va='bottom' if imp > 0 else 'top')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_percentages.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution comparison for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Metric Distribution Comparison', fontsize=16)
    
    key_metrics = ['dice', 'iou', 'sensitivity', 'specificity']
    
    for i, metric in enumerate(key_metrics):
        ax = axes[i // 2, i % 2]
        
        baseline_values = baseline_results['segmentation_metrics'][metric]['values']
        ccst_values = ccst_results['segmentation_metrics'][metric]['values']
        
        ax.hist(baseline_values, bins=20, alpha=0.5, label='Baseline', density=True)
        ax.hist(ccst_values, bins=20, alpha=0.5, label='CCST', density=True)
        
        ax.set_title(f'{metric.title()} Distribution')
        ax.set_xlabel(metric.title())
        ax.set_ylabel('Density')
        ax.legend()
        
        # Add mean lines
        ax.axvline(np.mean(baseline_values), color='blue', linestyle='--', alpha=0.7, label='Baseline Mean')
        ax.axvline(np.mean(ccst_values), color='orange', linestyle='--', alpha=0.7, label='CCST Mean')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Compare baseline and CCST model results')
    parser.add_argument('--baseline-results', type=str, required=True,
                       help='Path to baseline results JSON file')
    parser.add_argument('--ccst-results', type=str, required=True,
                       help='Path to CCST results JSON file')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for comparison plots')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output JSON file to save comparison results')
    
    args = parser.parse_args()
    
    # Load results
    try:
        baseline_results = load_results(args.baseline_results)
        ccst_results = load_results(args.ccst_results)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    print(f"✅ Results loaded successfully")
    print(f"  Baseline: {args.baseline_results}")
    print(f"  CCST: {args.ccst_results}")
    
    # Create comparison
    comparison = create_comparison_table(baseline_results, ccst_results)
    
    # Print report
    print_comparison_report(comparison, baseline_results, ccst_results)
    
    # Create plots
    create_comparison_plots(comparison, baseline_results, ccst_results, args.output_dir)
    
    # Save comparison results
    if args.output_file:
        comparison_results = {
            'comparison_table': comparison,
            'baseline_summary': {
                'total_samples': baseline_results['total_samples'],
                'dice_mean': baseline_results['segmentation_metrics']['dice']['mean'],
                'iou_mean': baseline_results['segmentation_metrics']['iou']['mean'],
                'accuracy': baseline_results['classification_metrics']['accuracy']
            },
            'ccst_summary': {
                'total_samples': ccst_results['total_samples'],
                'dice_mean': ccst_results['segmentation_metrics']['dice']['mean'],
                'iou_mean': ccst_results['segmentation_metrics']['iou']['mean'],
                'accuracy': ccst_results['classification_metrics']['accuracy']
            }
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n📁 Comparison results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 