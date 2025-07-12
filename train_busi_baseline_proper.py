#!/usr/bin/env python3
"""
Proper BUSI Baseline Training Script
Replicates the existing IS2D infrastructure for BUSI-only training
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import json

# Add the required imports following the existing pattern
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from IS2D_models import IS2D_model
from utils.get_functions import get_deivce
from utils.calculate_metrics import calculate_dice, calculate_iou, calculate_hausdorff

class BUSIBaselineTrainer:
    def __init__(self, args):
        self.args = args
        self.args.device = get_deivce()
        
        # Set BUSI-specific configuration (following IS2D_main.py)
        self.args.num_channels = 3
        self.args.image_size = 352
        self.args.num_classes = 1
        
        # Model configuration (following IS2D_main.py defaults)
        self.args.cnn_backbone = getattr(args, 'cnn_backbone', 'resnest50')
        self.args.scale_branches = getattr(args, 'scale_branches', 3)
        self.args.frequency_branches = getattr(args, 'frequency_branches', 16)
        self.args.frequency_selection = getattr(args, 'frequency_selection', 'top')
        self.args.block_repetition = getattr(args, 'block_repetition', 1)
        self.args.min_channel = getattr(args, 'min_channel', 32)
        self.args.min_resolution = getattr(args, 'min_resolution', 8)
        
        self.setup_datasets()
        self.setup_model()
        
    def setup_datasets(self):
        """Setup train, val, test datasets following existing pattern"""
        print("Setting up BUSI datasets...")
        
        # Create transforms (following biomedical_2dimage_segmentation_experiment.py)
        transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]
        
        target_transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]
        
        self.image_transform = transforms.Compose(transform_list)
        self.target_transform = transforms.Compose(target_transform_list)
        
        # Create datasets
        self.train_dataset = BUSISegmentationDataset(
            self.args.busi_path, 
            mode='train', 
            transform=self.image_transform, 
            target_transform=self.target_transform
        )
        
        self.val_dataset = BUSISegmentationDataset(
            self.args.busi_path, 
            mode='val', 
            transform=self.image_transform, 
            target_transform=self.target_transform
        )
        
        self.test_dataset = BUSISegmentationDataset(
            self.args.busi_path, 
            mode='test', 
            transform=self.image_transform, 
            target_transform=self.target_transform
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers, 
            pin_memory=True, 
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=True
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        
    def setup_model(self):
        """Setup model following existing pattern"""
        print("Setting up MFMSNet model...")
        
        # Load model using IS2D_model function
        self.model = IS2D_model(self.args)
        self.model.to(self.args.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        print("Model setup complete!")
        
    def train_epoch(self, epoch):
        """Train for one epoch following existing pattern"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (image, target) in enumerate(pbar):
            image, target = image.to(self.args.device), target.to(self.args.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (following the existing pattern)
            with torch.cuda.amp.autocast(enabled=True):
                output = self.model(image, mode='train')
                loss = self.model._calculate_criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * image.size(0)
            
            if (batch_idx + 1) % self.args.step == 0:
                pbar.set_description(f"Epoch {epoch} | {batch_idx + 1}/{len(self.train_loader)} ({(batch_idx + 1) / len(self.train_loader) * 100:.1f}%)")
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
        
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.val_loader):
                image, target = image.to(self.args.device), target.to(self.args.device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.model(image, mode='test')
                    loss = self.model._calculate_criterion(output, target)
                
                total_loss += loss.item() * image.size(0)
                
                # Calculate metrics using the main prediction
                predict = torch.sigmoid(output[3][0]).squeeze()
                
                dice = calculate_dice(predict.unsqueeze(0), target)
                iou = calculate_iou(predict.unsqueeze(0), target)
                
                total_dice += dice
                total_iou += iou
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_dice = total_dice / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        
        return avg_loss, avg_dice, avg_iou
        
    def test_model(self):
        """Test the model following existing pattern"""
        print("Testing model...")
        self.model.eval()
        
        all_dice = []
        all_iou = []
        all_hausdorff = []
        
        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(tqdm(self.test_loader, desc="Testing")):
                image, target = image.to(self.args.device), target.to(self.args.device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.model(image, mode='test')
                
                # Use the main prediction (output[3][0])
                predict = torch.sigmoid(output[3][0]).squeeze()
                
                dice = calculate_dice(predict.unsqueeze(0), target)
                iou = calculate_iou(predict.unsqueeze(0), target)
                hausdorff = calculate_hausdorff(predict.unsqueeze(0), target)
                
                all_dice.append(dice)
                all_iou.append(iou)
                all_hausdorff.append(hausdorff)
        
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
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        save_dir = os.path.join(self.args.save_path, "BUSI", "model_weights")
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': epoch,
            'args': self.args
        }
        
        save_path = os.path.join(save_dir, f"model_weight(EPOCH {epoch}).pth.tar")
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pth.tar")
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
        
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        best_dice = 0.0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }
        
        for epoch in range(1, self.args.final_epoch + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate_epoch(epoch)
            
            # Save history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['val_dice'].append(val_dice)
            training_history['val_iou'].append(val_iou)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save checkpoint
            is_best = val_dice > best_dice
            if is_best:
                best_dice = val_dice
            
            self.save_checkpoint(epoch, is_best)
            
            # Save training history
            with open(os.path.join(self.args.save_path, 'training_history.json'), 'w') as f:
                json.dump(training_history, f, indent=2)
        
        print("Training complete!")
        
        # Test the best model
        test_results = self.test_model()
        
        # Save test results
        with open(os.path.join(self.args.save_path, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print("Test Results:")
        print(f"Dice: {test_results['dice']['mean']:.4f} ± {test_results['dice']['std']:.4f}")
        print(f"IoU: {test_results['iou']['mean']:.4f} ± {test_results['iou']['std']:.4f}")
        print(f"Hausdorff: {test_results['hausdorff']['mean']:.4f} ± {test_results['hausdorff']['std']:.4f}")
        
        return test_results

def main():
    parser = argparse.ArgumentParser(description='BUSI Baseline Training - Proper Implementation')
    parser.add_argument('--busi-path', type=str, default='dataset/BioMedicalDataset/BUSI',
                       help='Path to BUSI dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--final-epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--step', type=int, default=10, help='Print step interval')
    parser.add_argument('--save-path', type=str, default='model_weights', help='Save path')
    
    # MFMSNet parameters
    parser.add_argument('--cnn-backbone', type=str, default='resnest50', help='CNN backbone')
    parser.add_argument('--scale-branches', type=int, default=3, help='Scale branches')
    parser.add_argument('--frequency-branches', type=int, default=16, help='Frequency branches')
    parser.add_argument('--frequency-selection', type=str, default='top', help='Frequency selection')
    parser.add_argument('--block-repetition', type=int, default=1, help='Block repetition')
    parser.add_argument('--min-channel', type=int, default=32, help='Min channel')
    parser.add_argument('--min-resolution', type=int, default=8, help='Min resolution')
    
    args = parser.parse_args()
    
    print("🚀 BUSI Baseline Training - Proper Implementation")
    print("=" * 60)
    print(f"Dataset: {args.busi_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.final_epoch}")
    print(f"Learning rate: {args.lr}")
    print(f"Backbone: {args.cnn_backbone}")
    print("=" * 60)
    
    # Create trainer and start training
    trainer = BUSIBaselineTrainer(args)
    results = trainer.train()
    
    print("\n🎉 Training Complete!")
    print(f"Final Results:")
    print(f"  Dice: {results['dice']['mean']:.4f} ± {results['dice']['std']:.4f}")
    print(f"  IoU: {results['iou']['mean']:.4f} ± {results['iou']['std']:.4f}")
    print(f"  Hausdorff: {results['hausdorff']['mean']:.4f} ± {results['hausdorff']['std']:.4f}")

if __name__ == "__main__":
    main() 