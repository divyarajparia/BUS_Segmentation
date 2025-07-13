#!/usr/bin/env python3
"""
Train MADGNet on BUSI CCST Combined Dataset
==========================================

Complete training pipeline for MADGNet using the BUSI CCST Combined dataset.
This script follows the CCST methodology for domain adaptation:

1. Training: Original BUSI + Style-transferred BUS-UCLM data
2. Validation: Original BUSI validation set
3. Testing: Original BUSI test set

The style-transferred data is generated using privacy-preserving AdaIN
following the CCST paper methodology.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import argparse
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add paths for imports
sys.path.append('.')
sys.path.append('IS2D_Experiment')
sys.path.append('dataset/BioMedicalDataset')

from dataset.BioMedicalDataset.BUSICCSTCombinedDataset import BUSICCSTCombinedDataset, BUSICCSTTrainingDataset
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from IS2D_models.mfmsnet import MFMSNet
from utils.calculate_metrics import calculate_metrics


def create_transforms(image_size=224):
    """Create training and validation transforms"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Target transforms for masks
    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform, target_transform


def create_dataloaders(combined_dir, original_busi_dir, batch_size=8, num_workers=4, image_size=224):
    """Create dataloaders for training, validation, and testing"""
    
    train_transform, val_transform, target_transform = create_transforms(image_size)
    
    # Training dataset: BUSI CCST Combined (original + styled)
    print("🔄 Creating BUSI CCST Combined training dataset...")
    train_dataset = BUSICCSTCombinedDataset(
        combined_dir=combined_dir,
        original_busi_dir=original_busi_dir,
        mode='train',
        transform=train_transform,
        target_transform=target_transform
    )
    
    # Validation dataset: Original BUSI validation set
    print("🔄 Creating BUSI validation dataset...")
    val_dataset = BUSICCSTCombinedDataset(
        combined_dir=combined_dir,
        original_busi_dir=original_busi_dir,
        mode='val',
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Test dataset: Original BUSI test set
    print("🔄 Creating BUSI test dataset...")
    test_dataset = BUSICCSTCombinedDataset(
        combined_dir=combined_dir,
        original_busi_dir=original_busi_dir,
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
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, 
                save_path="model_weights/ccst_combined_model.pth", log_dir="logs"):
    """Train the MADGNet model"""
    
    print(f"🚀 Starting MADGNet training with BUSI CCST Combined dataset...")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training history
    train_losses = []
    val_losses = []
    val_metrics = []
    
    best_val_loss = float('inf')
    best_val_dice = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, mode='train')
            
            # Calculate loss (handle multiple outputs from MFMSNet)
            if isinstance(outputs, list):
                # Multiple outputs from different decoder stages
                loss = 0
                for output in outputs:
                    if isinstance(output, list):
                        # Each output is [map, distance, boundary]
                        loss += criterion(output[0], masks)  # Use segmentation map
                    else:
                        loss += criterion(output, masks)
                loss /= len(outputs)
            else:
                loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_samples += images.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        all_predictions = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_pbar):
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                outputs = model(images, mode='test')
                
                # Calculate loss
                if isinstance(outputs, list):
                    # Use the first output for validation
                    main_output = outputs[0]
                    if isinstance(main_output, list):
                        main_output = main_output[0]  # Segmentation map
                    loss = criterion(main_output, masks)
                    predictions = torch.sigmoid(main_output)
                else:
                    loss = criterion(outputs, masks)
                    predictions = torch.sigmoid(outputs)
                
                val_loss += loss.item()
                val_samples += images.size(0)
                
                # Store predictions and targets for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(masks.cpu())
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate validation metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_predictions, all_targets)
        val_metrics.append(metrics)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Dice: {metrics['dice']:.4f}")
        print(f"  Val IoU: {metrics['iou']:.4f}")
        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_dice = metrics['dice']
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_metrics': metrics,
                'training_args': {
                    'num_epochs': num_epochs,
                    'lr': lr,
                    'batch_size': train_loader.batch_size
                }
            }, save_path)
            
            print(f"  ✅ New best model saved! (Val Loss: {avg_val_loss:.4f}, Dice: {metrics['dice']:.4f})")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice
        }
        
        history_path = os.path.join(log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Best validation Dice: {best_val_dice:.4f}")
    print(f"   Model saved to: {save_path}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'best_val_dice': best_val_dice
    }


def evaluate_model(model, test_loader, device, model_path=None):
    """Evaluate the trained model on test set"""
    
    print(f"📊 Evaluating model on test set...")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Load best model if path provided
    if model_path and os.path.exists(model_path):
        print(f"   Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Model loaded from epoch {checkpoint['epoch']}")
    
    model.eval()
    test_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images, mode='test')
            
            # Calculate loss
            if isinstance(outputs, list):
                main_output = outputs[0]
                if isinstance(main_output, list):
                    main_output = main_output[0]
                loss = criterion(main_output, masks)
                predictions = torch.sigmoid(main_output)
            else:
                loss = criterion(outputs, masks)
                predictions = torch.sigmoid(outputs)
            
            test_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
    
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate test metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    test_metrics = calculate_metrics(all_predictions, all_targets)
    
    print(f"\n📈 Test Results:")
    print(f"   Test Loss: {avg_test_loss:.4f}")
    print(f"   Test Dice: {test_metrics['dice']:.4f}")
    print(f"   Test IoU: {test_metrics['iou']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test Precision: {test_metrics['precision']:.4f}")
    print(f"   Test Recall: {test_metrics['recall']:.4f}")
    print(f"   Test F1-Score: {test_metrics['f1']:.4f}")
    
    return test_metrics


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train MADGNet on BUSI CCST Combined Dataset')
    parser.add_argument('--combined-dir', type=str, 
                       default='dataset/BioMedicalDataset/BUSI_CCST_Combined',
                       help='Path to BUSI CCST Combined dataset')
    parser.add_argument('--original-busi-dir', type=str, 
                       default='dataset/BioMedicalDataset/BUSI',
                       help='Path to original BUSI dataset')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, 
                       help='Image size for training')
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--save-path', type=str, 
                       default='model_weights/ccst_combined_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--log-dir', type=str, default='logs/ccst_combined',
                       help='Directory to save training logs')
    parser.add_argument('--results-path', type=str, 
                       default='results/ccst_combined_results.json',
                       help='Path to save test results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🚀 MADGNet Training with BUSI CCST Combined Dataset")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Combined dataset: {args.combined_dir}")
    print(f"  Original BUSI: {args.original_busi_dir}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image size: {args.image_size}")
    print(f"  Save path: {args.save_path}")
    
    # Check if combined dataset exists
    if not os.path.exists(args.combined_dir):
        print(f"❌ Combined dataset not found at {args.combined_dir}")
        print("   Please run ccst_privacy_preserving_adain.py first to generate the dataset")
        return
    
    # Create dataloaders
    print(f"\n📊 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.combined_dir,
        args.original_busi_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Print dataset statistics
    print(f"\n📈 Dataset Statistics:")
    train_dataset = train_loader.dataset
    train_dataset.print_dataset_info()
    
    # Initialize model
    print(f"\n🏗️  Initializing MADGNet model...")
    model = MFMSNet().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Train model
    print(f"\n🎯 Starting training...")
    training_history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        save_path=args.save_path,
        log_dir=args.log_dir
    )
    
    # Test model
    print(f"\n🧪 Testing model...")
    test_metrics = evaluate_model(
        model, test_loader, device,
        model_path=args.save_path
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    results = {
        'training_config': {
            'combined_dir': args.combined_dir,
            'original_busi_dir': args.original_busi_dir,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'lr': args.lr,
            'image_size': args.image_size,
            'device': str(device)
        },
        'dataset_stats': train_dataset.get_dataset_statistics(),
        'training_history': training_history,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(args.results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Training and evaluation completed!")
    print(f"   Final test Dice score: {test_metrics['dice']:.4f}")
    print(f"   Model saved to: {args.save_path}")
    print(f"   Results saved to: {args.results_path}")
    
    # Summary of CCST methodology benefits
    print(f"\n🔍 CCST Methodology Summary:")
    print(f"   Privacy-preserving: ✅ Only domain statistics were shared")
    print(f"   Domain adaptation: ✅ BUS-UCLM data styled to match BUSI")
    print(f"   Fair evaluation: ✅ Testing on original BUSI only")
    print(f"   Data augmentation: ✅ Training data significantly increased")


if __name__ == "__main__":
    main() 