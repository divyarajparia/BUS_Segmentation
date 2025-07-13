#!/usr/bin/env python3
"""
Training script using CCST-augmented data for domain adaptation
Following the exact CCST paper methodology for federated domain generalization
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from dataset.BioMedicalDataset.CCSTDataset import CCSTAugmentedDataset
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from IS2D_models.mfmsnet import MFMSNet
from utils.calculate_metrics import calculate_dice, calculate_iou

def create_transforms():
    """Create training and validation transforms following CCST paper"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Target transforms for masks
    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform, target_transform

def create_dataloaders(ccst_augmented_path, original_busi_path, batch_size=8, num_workers=4):
    """
    Create dataloaders following CCST paper methodology:
    - Training: CCST-augmented data (combines original + style-transferred)
    - Validation/Test: Original BUSI only (fair evaluation)
    """
    
    train_transform, val_transform, target_transform = create_transforms()
    
    # Training dataset: CCST-augmented data
    print("🔄 Creating CCST-augmented training dataset...")
    train_dataset = CCSTAugmentedDataset(
        ccst_augmented_dir=ccst_augmented_path,
        original_busi_dir=original_busi_path,
        mode='train',
        transform=train_transform,
        target_transform=target_transform,
        combine_with_original=True  # Include original BUSI data too
    )
    
    # Validation dataset: Original BUSI only
    print("🔄 Creating validation dataset (original BUSI only)...")
    val_dataset = BUSISegmentationDataset(
        original_busi_path,
        mode='val',
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Test dataset: Original BUSI only
    print("🔄 Creating test dataset (original BUSI only)...")
    test_dataset = BUSISegmentationDataset(
        original_busi_path,
        mode='test',
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Create dataloaders
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
    
    print(f"📊 Dataset sizes:")
    print(f"   Training (CCST-augmented): {len(train_dataset)}")
    print(f"   Validation (original BUSI): {len(val_dataset)}")
    print(f"   Test (original BUSI): {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, *, save_path='best_ccst_model.pth', num_epochs=100, lr=0.001):
    """Train the segmentation model with CCST-augmented data"""
    
    print(f"\n🎯 Training model with CCST-augmented data...")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {lr}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, mode='train')
            
            # Ensure we always compute loss on the main segmentation map
            def get_map(t):
                if isinstance(t, (list, tuple)):
                    # Expect deepest resolution at index 3
                    candidate = t[3] if len(t) > 3 else t[-1]
                    return candidate[0] if isinstance(candidate, (list, tuple)) else candidate
                return t

            map_output = get_map(outputs)
            loss = criterion(map_output, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_samples += images.size(0)
            
            if batch_idx % 20 == 0:
                print(f'   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images, mode='test')  # Model may return tuple/list
                map_output = get_map(outputs)
                loss = criterion(map_output, masks)
                
                val_loss += loss.item()
                val_samples += images.size(0)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        
        print(f'📊 Epoch {epoch+1} Results:')
        print(f'   Train Loss: {avg_train_loss:.4f}')
        print(f'   Val Loss: {avg_val_loss:.4f}')
        print(f'   Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'   ✅ New best model saved! Val Loss: {best_val_loss:.4f}')
        
        scheduler.step()
    
    return training_history

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test set (original BUSI only for fair comparison)"""
    
    print(f"\n📊 Evaluating model on test set (original BUSI only)...")
    
    def get_map(t):
        if isinstance(t, (list, tuple)):
            candidate = t[3] if len(t) > 3 else t[-1]
            return candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        return t

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    test_loss = 0.0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images, mode='test')
            map_output = get_map(outputs)
            loss = criterion(map_output, masks)
            
            test_loss += loss.item()
            preds = torch.sigmoid(map_output) > 0.5
            for i in range(preds.size(0)):
                dice = calculate_dice(preds[i:i+1], masks[i:i+1])
                iou  = calculate_iou(preds[i:i+1], masks[i:i+1])
                dice_scores.append(dice)
                iou_scores.append(iou)

    avg_test_loss = test_loss / len(test_loader)
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    mean_iou  = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    print(f'🎯 Final Test Results (CCST Paper Methodology):')
    print(f'   Test Loss: {avg_test_loss:.4f}')
    print(f'   Mean Dice: {mean_dice:.4f}')
    print(f'   Mean IoU: {mean_iou:.4f}')
    print(f'   Evaluated on {len(test_loader)} batches of original BUSI data')

    return {
        'test_loss': avg_test_loss,
        'mean_dice': mean_dice,
        'mean_iou': mean_iou
    }

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model with CCST-augmented data')
    parser.add_argument('--ccst-augmented-path', type=str, 
                       default='dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented',
                       help='Path to CCST-augmented dataset')
    parser.add_argument('--original-busi-path', type=str, 
                       default='dataset/BioMedicalDataset/BUSI',
                       help='Path to original BUSI dataset')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    # Optional paths so pipeline can override defaults without raising errors
    parser.add_argument('--save-path', type=str, default='best_ccst_model.pth',
                        help='Path to save best model weights (default: best_ccst_model.pth)')
    parser.add_argument('--results-path', type=str, default='ccst_results.json',
                        help='(Optional) Path to save results JSON – currently unused but accepted to avoid CLI errors')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 Training with CCST-Augmented Data")
    print("=" * 50)
    print(f"Following CCST paper methodology for federated domain generalization")
    print(f"Configuration:")
    print(f"  CCST-augmented path: {args.ccst_augmented_path}")
    print(f"  Original BUSI path: {args.original_busi_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    
    # Check if CCST data exists
    if not os.path.exists(args.ccst_augmented_path):
        print(f"❌ CCST-augmented data not found at {args.ccst_augmented_path}")
        print("Please run ccst_exact_replication.py first to generate CCST data")
        return
    
    if not os.path.exists(args.original_busi_path):
        print(f"❌ Original BUSI data not found at {args.original_busi_path}")
        return
    
    # Create dataloaders
    print(f"\n📊 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.ccst_augmented_path,
        args.original_busi_path, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print(f"\n🏗️  Initializing MFMSNet model...")
    model = MFMSNet().to(device)
    
    # Display dataset statistics
    print(f"\n📈 CCST Dataset Analysis:")
    sample_dataset = CCSTAugmentedDataset(
        ccst_augmented_dir=args.ccst_augmented_path,
        original_busi_dir=args.original_busi_path,
        mode='train',
        combine_with_original=True
    )
    stats = sample_dataset.get_style_transfer_stats()
    print(f"   Total training samples: {stats['total_samples']}")
    print(f"   Original samples: {stats['original_samples']}")
    print(f"   Style-transferred samples: {stats['styled_samples']}")
    print(f"   Style transfer combinations:")
    for combo, count in stats['client_combinations'].items():
        print(f"     {combo}: {count} samples")
    
    # Train model
    print(f"\n🎯 Starting training with CCST methodology...")
    training_history = train_model(
        model, train_loader, val_loader, device,
        save_path=args.save_path,
        num_epochs=args.num_epochs,
        lr=args.lr
    )
    
    # Load best model and evaluate
    print(f"\n📊 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(args.save_path))
    
    # Final evaluation on test set (original BUSI only)
    test_results = evaluate_model(model, test_loader, device)

    # Save results JSON if results_path provided
    try:
        import json, os
        with open(args.results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save results to {args.results_path}: {e}")
    
    # Save training history
    import json
    with open('ccst_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n🎉 CCST Training Completed!")
    print(f"=" * 50)
    print(f"Final Results (following CCST paper evaluation):")
    print(f"  Test Loss: {test_results['test_loss']:.4f}")
    print(f"  Mean Dice: {test_results['mean_dice']:.4f}")
    print(f"  Mean IoU: {test_results['mean_iou']:.4f}")
    print(f"\nFiles saved:")
    print(f"  Best model: {args.save_path}")
    print(f"  Training history: ccst_training_history.json")
    print(f"\n✅ Model trained with CCST domain generalization methodology!")

if __name__ == "__main__":
    main() 