#!/usr/bin/env python3
"""
Baseline training script: Train segmentation model on BUSI dataset only
This serves as a baseline to compare against CCST domain adaptation results.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import json
import numpy as np
from tqdm import tqdm
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from IS2D_models.mfmsnet import MFMSNet
from utils.calculate_metrics import calculate_dice, calculate_iou, calculate_hausdorff

def create_transforms():
    """Create training and validation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform, target_transform

def create_dataloaders(busi_path, batch_size=8, num_workers=4):
    """Create dataloaders for baseline training (BUSI only)"""
    
    train_transform, val_transform, target_transform = create_transforms()
    
    # Training dataset: BUSI training data only
    train_dataset = BUSISegmentationDataset(
        busi_path,
        mode='train',
        transform=train_transform,
        target_transform=target_transform
    )
    
    # Validation dataset: BUSI validation data
    val_dataset = BUSISegmentationDataset(
        busi_path,
        mode='val',
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Test dataset: BUSI test data
    test_dataset = BUSISegmentationDataset(
        busi_path,
        mode='test',
        transform=val_transform,
        target_transform=target_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, mode='train')
        # outputs is a list of 4 stages; each stage is [map, dist, boundary]
        map_output = outputs[3][0]  # Use the highest resolution map for supervision
        loss = criterion(map_output, masks)
        
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred_masks = torch.sigmoid(map_output) > 0.5
            dice = calculate_dice(pred_masks, masks)
            iou = calculate_iou(pred_masks, masks)
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}',
                'IoU': f'{iou:.4f}'
            })
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    
    return avg_loss, avg_dice, avg_iou

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images, mode='train')
            map_output = outputs[3][0]
            loss = criterion(map_output, masks)
            
            pred_masks = torch.sigmoid(map_output) > 0.5
            dice = calculate_dice(pred_masks, masks)
            iou = calculate_iou(pred_masks, masks)
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}',
                'IoU': f'{iou:.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    return avg_loss, avg_dice, avg_iou

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_dice = []
    all_iou = []
    all_hausdorff = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images, mode='test')
            pred_masks = torch.sigmoid(outputs) > 0.5
            
            # Calculate metrics for each sample in batch
            for i in range(pred_masks.size(0)):
                dice = calculate_dice(pred_masks[i:i+1], masks[i:i+1])
                iou = calculate_iou(pred_masks[i:i+1], masks[i:i+1])
                hausdorff = calculate_hausdorff(pred_masks[i:i+1], masks[i:i+1])
                
                all_dice.append(dice)
                all_iou.append(iou)
                all_hausdorff.append(hausdorff)
            
            pbar.set_postfix({
                'Dice': f'{np.mean(all_dice):.4f}',
                'IoU': f'{np.mean(all_iou):.4f}',
                'HD': f'{np.mean(all_hausdorff):.4f}'
            })
    
    results = {
        'dice': {
            'mean': np.mean(all_dice),
            'std': np.std(all_dice),
            'values': all_dice
        },
        'iou': {
            'mean': np.mean(all_iou),
            'std': np.std(all_iou),
            'values': all_iou
        },
        'hausdorff': {
            'mean': np.mean(all_hausdorff),
            'std': np.std(all_hausdorff),
            'values': all_hausdorff
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train baseline model on BUSI dataset only')
    parser.add_argument('--busi-path', type=str, 
                       default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save-path', type=str, default='baseline_busi_only_model.pth', 
                       help='Path to save model')
    parser.add_argument('--results-path', type=str, default='baseline_results.json',
                       help='Path to save results')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 Training Baseline Model (BUSI Only)")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Dataset: {args.busi_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Save path: {args.save_path}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.busi_path, args.batch_size, args.num_workers
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    model = MFMSNet(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Use BCEWithLogitsLoss on the final map output (output[3][0]) as done in existing IS2D infrastructure
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_dice = 0
    training_history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': []
    }
    
    print(f"\n🏋️ Starting Training...")
    
    for epoch in range(args.num_epochs):
        # Training
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validation
        val_loss, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['train_dice'].append(train_dice)
        training_history['train_iou'].append(train_iou)
        training_history['val_loss'].append(val_loss)
        training_history['val_dice'].append(val_dice)
        training_history['val_iou'].append(val_iou)
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), args.save_path)
            print(f"✅ New best model saved! Validation Dice: {val_dice:.4f}")
        
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print()
    
    # Load best model for testing
    model.load_state_dict(torch.load(args.save_path))
    
    # Test evaluation
    print("🧪 Evaluating on Test Set...")
    test_results = evaluate_model(model, test_loader, device)
    
    # Save results
    final_results = {
        'model_type': 'baseline_busi_only',
        'dataset': args.busi_path,
        'training_samples': len(train_loader.dataset),
        'validation_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'best_validation_dice': best_val_dice,
        'test_results': test_results,
        'training_history': training_history,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.lr,
            'device': str(device)
        }
    }
    
    with open(args.results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Training Completed!")
    print(f"📊 Final Test Results:")
    print(f"  Dice Coefficient: {test_results['dice']['mean']:.4f} ± {test_results['dice']['std']:.4f}")
    print(f"  IoU: {test_results['iou']['mean']:.4f} ± {test_results['iou']['std']:.4f}")
    print(f"  Hausdorff Distance: {test_results['hausdorff']['mean']:.4f} ± {test_results['hausdorff']['std']:.4f}")
    print(f"📁 Model saved to: {args.save_path}")
    print(f"📁 Results saved to: {args.results_path}")

if __name__ == "__main__":
    main() 