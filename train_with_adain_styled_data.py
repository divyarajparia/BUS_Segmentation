#!/usr/bin/env python3
"""
Training script using AdaIN-styled data for domain adaptation
Following CCST methodology for single-institution use
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from dataset.BioMedicalDataset.AdaINStyleTransferDataset import AdaINStyleTransferDataset
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from IS2D_models.mfmsnet import MFMSNet

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

def create_dataloaders(busi_path, adain_styled_path, batch_size=8, num_workers=4):
    """Create training and validation dataloaders"""
    
    train_transform, val_transform, target_transform = create_transforms()
    
    # Training dataset: BUSI + AdaIN-styled BUS-UCLM
    train_dataset = AdaINStyleTransferDataset(
        busi_dir=busi_path,
        adain_styled_dir=adain_styled_path,
        mode='train',
        transform=train_transform,
        target_transform=target_transform
    )
    
    # Validation dataset: Original BUSI only
    val_dataset = BUSISegmentationDataset(
        busi_path,
        mode='val',
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Test dataset: Original BUSI only  
    test_dataset = BUSISegmentationDataset(
        busi_path,
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
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001):
    """Train the segmentation model"""
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, mode='train')
            
            # Calculate loss (for MFMSNet with multiple outputs)
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
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images, mode='test')  # Single output in test mode
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_samples += images.size(0)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_adain_model.pth')
            print(f'  ✅ New best model saved! Val Loss: {best_val_loss:.4f}')
        
        scheduler.step()
        print()

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test set"""
    
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    test_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images, mode='test')
            loss = criterion(outputs, masks)
            
            test_loss += loss.item()
            
            # Calculate pixel accuracy
            predictions = torch.sigmoid(outputs) > 0.5
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_test_loss = test_loss / len(test_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    print(f'Test Results:')
    print(f'  Test Loss: {avg_test_loss:.4f}')
    print(f'  Pixel Accuracy: {pixel_accuracy:.4f}')
    
    return avg_test_loss, pixel_accuracy

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model with AdaIN-styled data')
    parser.add_argument('--busi-path', type=str, default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--adain-styled-path', type=str, default='dataset/BioMedicalDataset/BUS-UCLM-AdaIN-styled',
                       help='Path to AdaIN-styled BUS-UCLM dataset')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting training with AdaIN-styled data on {device}")
    print(f"Configuration:")
    print(f"  BUSI path: {args.busi_path}")
    print(f"  AdaIN-styled path: {args.adain_styled_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    
    # Check if styled data exists
    if not os.path.exists(args.adain_styled_path):
        print(f"❌ AdaIN-styled data not found at {args.adain_styled_path}")
        print("Please run adain_style_transfer.py first to generate styled data")
        return
    
    # Create dataloaders
    print("\n📊 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.busi_path, 
        args.adain_styled_path,
        batch_size=args.batch_size
    )
    
    # Initialize model
    print("\n🏗️  Initializing model...")
    model = MFMSNet().to(device)
    
    # Train model
    print("\n🎯 Starting training...")
    train_model(model, train_loader, val_loader, device, args.num_epochs, args.lr)
    
    # Load best model and evaluate
    print("\n📊 Evaluating best model...")
    model.load_state_dict(torch.load('best_adain_model.pth'))
    test_loss, pixel_accuracy = evaluate_model(model, test_loader, device)
    
    print(f"\n🎉 Training completed!")
    print(f"Final Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Pixel Accuracy: {pixel_accuracy:.4f}")

if __name__ == "__main__":
    main() 