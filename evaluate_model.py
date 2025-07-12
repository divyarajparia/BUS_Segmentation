#!/usr/bin/env python3
"""
Model evaluation script for comparing baseline and CCST models
Can evaluate any trained model on any dataset for performance comparison.
"""

import os
import torch
import json
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from dataset.BioMedicalDataset.CCSTDataset import CCSTAugmentedDataset
from IS2D_models.mfmsnet import MFMSNet
from utils.calculate_metrics import calculate_dice, calculate_iou, calculate_hausdorff
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def create_transforms():
    """Create evaluation transforms (no augmentation)"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return transform, target_transform

def create_dataloader(dataset_path, dataset_type='busi', mode='test', batch_size=8, num_workers=4):
    """Create dataloader for evaluation"""
    
    transform, target_transform = create_transforms()
    
    if dataset_type == 'busi':
        dataset = BUSISegmentationDataset(
            dataset_path,
            mode=mode,
            transform=transform,
            target_transform=target_transform
        )
    elif dataset_type == 'ccst':
        dataset = CCSTAugmentedDataset(
            ccst_augmented_dir=dataset_path,
            original_busi_dir=None,
            mode=mode,
            transform=transform,
            target_transform=target_transform,
            combine_with_original=False
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def calculate_sensitivity_specificity(pred_masks, true_masks):
    """Calculate sensitivity and specificity"""
    pred_flat = pred_masks.flatten().cpu().numpy()
    true_flat = true_masks.flatten().cpu().numpy()
    
    tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity

def evaluate_model(model, dataloader, device, save_predictions=False, output_dir=None):
    """Comprehensive model evaluation"""
    
    model.eval()
    
    # Metrics storage
    all_dice = []
    all_iou = []
    all_hausdorff = []
    all_sensitivity = []
    all_specificity = []
    
    # For classification metrics
    all_pred_labels = []
    all_true_labels = []
    
    # For visualization
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            pred_masks = torch.sigmoid(outputs) > 0.5
            
            # Calculate metrics for each sample in batch
            for i in range(pred_masks.size(0)):
                pred_mask = pred_masks[i:i+1]
                true_mask = masks[i:i+1]
                
                # Segmentation metrics
                dice = calculate_dice(pred_mask, true_mask)
                iou = calculate_iou(pred_mask, true_mask)
                hausdorff = calculate_hausdorff(pred_mask, true_mask)
                
                # Classification metrics
                sensitivity, specificity = calculate_sensitivity_specificity(pred_mask, true_mask)
                
                # Store metrics
                all_dice.append(dice)
                all_iou.append(iou)
                all_hausdorff.append(hausdorff)
                all_sensitivity.append(sensitivity)
                all_specificity.append(specificity)
                
                # Classification labels (tumor present/absent)
                pred_label = 1 if pred_mask.sum() > 0 else 0
                true_label = 1 if true_mask.sum() > 0 else 0
                all_pred_labels.append(pred_label)
                all_true_labels.append(true_label)
                
                # Store predictions if requested
                if save_predictions:
                    predictions.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'dice': dice,
                        'iou': iou,
                        'hausdorff': hausdorff,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'pred_label': pred_label,
                        'true_label': true_label
                    })
            
            # Update progress bar
            pbar.set_postfix({
                'Dice': f'{np.mean(all_dice):.4f}',
                'IoU': f'{np.mean(all_iou):.4f}',
                'HD': f'{np.mean(all_hausdorff):.4f}',
                'Sens': f'{np.mean(all_sensitivity):.4f}',
                'Spec': f'{np.mean(all_specificity):.4f}'
            })
    
    # Calculate final statistics
    results = {
        'segmentation_metrics': {
            'dice': {
                'mean': np.mean(all_dice),
                'std': np.std(all_dice),
                'median': np.median(all_dice),
                'values': all_dice
            },
            'iou': {
                'mean': np.mean(all_iou),
                'std': np.std(all_iou),
                'median': np.median(all_iou),
                'values': all_iou
            },
            'hausdorff': {
                'mean': np.mean(all_hausdorff),
                'std': np.std(all_hausdorff),
                'median': np.median(all_hausdorff),
                'values': all_hausdorff
            },
            'sensitivity': {
                'mean': np.mean(all_sensitivity),
                'std': np.std(all_sensitivity),
                'median': np.median(all_sensitivity),
                'values': all_sensitivity
            },
            'specificity': {
                'mean': np.mean(all_specificity),
                'std': np.std(all_specificity),
                'median': np.median(all_specificity),
                'values': all_specificity
            }
        },
        'classification_metrics': {
            'accuracy': np.mean(np.array(all_pred_labels) == np.array(all_true_labels)),
            'confusion_matrix': confusion_matrix(all_true_labels, all_pred_labels).tolist(),
            'classification_report': classification_report(all_true_labels, all_pred_labels, output_dict=True)
        },
        'total_samples': len(all_dice),
        'predictions': predictions if save_predictions else None
    }
    
    # Save visualizations if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        create_evaluation_plots(results, output_dir)
    
    return results

def create_evaluation_plots(results, output_dir):
    """Create evaluation plots"""
    
    # Metrics distribution plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Evaluation Metrics Distribution', fontsize=16)
    
    metrics = ['dice', 'iou', 'hausdorff', 'sensitivity', 'specificity']
    
    for i, metric in enumerate(metrics):
        if i < 5:  # We have 5 metrics
            ax = axes[i // 3, i % 3]
            values = results['segmentation_metrics'][metric]['values']
            
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric.title()} Distribution')
            ax.set_xlabel(metric.title())
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
            ax.legend()
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix plot
    if 'confusion_matrix' in results['classification_metrics']:
        cm = np.array(results['classification_metrics']['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Tumor', 'Tumor'],
                   yticklabels=['No Tumor', 'Tumor'])
        plt.title('Confusion Matrix - Tumor Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Evaluation plots saved to: {output_dir}")

def print_results_summary(results, model_name="Model"):
    """Print comprehensive results summary"""
    
    seg_metrics = results['segmentation_metrics']
    cls_metrics = results['classification_metrics']
    
    print(f"\n🎯 {model_name} Evaluation Results")
    print("=" * 50)
    print(f"Total samples evaluated: {results['total_samples']}")
    
    print(f"\n📊 Segmentation Metrics:")
    print(f"  Dice Coefficient: {seg_metrics['dice']['mean']:.4f} ± {seg_metrics['dice']['std']:.4f}")
    print(f"  IoU:             {seg_metrics['iou']['mean']:.4f} ± {seg_metrics['iou']['std']:.4f}")
    print(f"  Hausdorff Dist:  {seg_metrics['hausdorff']['mean']:.4f} ± {seg_metrics['hausdorff']['std']:.4f}")
    print(f"  Sensitivity:     {seg_metrics['sensitivity']['mean']:.4f} ± {seg_metrics['sensitivity']['std']:.4f}")
    print(f"  Specificity:     {seg_metrics['specificity']['mean']:.4f} ± {seg_metrics['specificity']['std']:.4f}")
    
    print(f"\n🎯 Classification Metrics:")
    print(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
    print(f"  Confusion Matrix: {cls_metrics['confusion_matrix']}")
    
    if 'classification_report' in cls_metrics:
        print(f"\n📋 Classification Report:")
        report = cls_metrics['classification_report']
        print(f"  Precision: {report['weighted avg']['precision']:.4f}")
        print(f"  Recall:    {report['weighted avg']['recall']:.4f}")
        print(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained segmentation model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset for evaluation')
    parser.add_argument('--dataset-type', type=str, default='busi',
                       choices=['busi', 'ccst'],
                       help='Type of dataset (busi or ccst)')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output JSON file to save results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots and visualizations')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save individual predictions')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔍 Model Evaluation")
    print("=" * 30)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found: {args.dataset_path}")
        return
    
    # Load model
    model = MFMSNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ Model loaded successfully")
    
    # Create dataloader
    dataloader = create_dataloader(
        args.dataset_path,
        args.dataset_type,
        args.mode,
        args.batch_size,
        args.num_workers
    )
    print(f"✅ Dataset loaded: {len(dataloader.dataset)} samples")
    
    # Evaluate model
    results = evaluate_model(
        model,
        dataloader,
        device,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Print results
    model_name = os.path.basename(args.model_path).replace('.pth', '')
    print_results_summary(results, model_name)
    
    # Save results
    if args.output_file:
        # Add metadata
        results['metadata'] = {
            'model_path': args.model_path,
            'dataset_path': args.dataset_path,
            'dataset_type': args.dataset_type,
            'mode': args.mode,
            'batch_size': args.batch_size,
            'device': str(device)
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {args.output_file}")
    
    if args.output_dir:
        print(f"📁 Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 