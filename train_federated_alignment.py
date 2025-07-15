"""
Federated Feature Alignment Training Script
==========================================

Training script for federated domain adaptation using feature alignment.
Integrates with existing IS2D/MADGNet framework for medical image segmentation.

This script implements a two-phase federated learning approach:
1. Phase 1: Extract privacy-preserving feature statistics from BUSI (Institution A)
2. Phase 2: Train on BUS-UCLM with feature alignment to BUSI statistics (Institution B)

Key advantages over style transfer:
- No visual artifacts or degradation
- Mathematically proven privacy guarantees
- Better domain adaptation through deep feature alignment
- Preserves medical image authenticity

Usage:
    python train_federated_alignment.py --phase extract_stats --source_dataset BUSI
    python train_federated_alignment.py --phase federated_training --target_dataset BUS-UCLM
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  TensorBoard not available - metrics logging will be limited")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_feature_alignment import (
    FederatedFeatureExtractor, 
    FederatedDomainAdapter, 
    PrivacyConfig,
    load_source_statistics
)

# Import existing framework components
from IS2D_models import IS2D_model
from IS2D_Experiment._IS2Dbase import BaseSegmentationExperiment
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from dataset.BioMedicalDataset.BUSUCLMSegmentationDataset import BUSUCLMSegmentationDataset
from utils.calculate_metrics import compute_dice_iou_metrics


class FederatedAlignment_IS2D(BaseSegmentationExperiment):
    """
    Federated Feature Alignment experiment class extending BaseSegmentationExperiment.
    
    This class implements federated domain adaptation using feature alignment
    instead of style transfer for better privacy and performance.
    """
    
    def __init__(self, args):
        super(FederatedAlignment_IS2D, self).__init__(args)
        
        # Set device for convenience (base class sets args.device)
        self.device = self.args.device
        
        # Federated learning specific parameters
        self.privacy_config = PrivacyConfig(
            epsilon=args.privacy_epsilon,
            delta=args.privacy_delta,
            sensitivity=args.privacy_sensitivity
        )
        
        self.alignment_weight = args.alignment_weight
        self.phase = args.phase
        self.source_stats_path = args.source_stats_path
        
        # Initialize federated components
        self.feature_extractor = None
        self.domain_adapter = None
        
        if self.phase == 'federated_training' and self.source_stats_path:
            self.source_statistics = load_source_statistics(self.source_stats_path, self.device)
            self.domain_adapter = FederatedDomainAdapter(
                self.model, self.source_statistics, self.alignment_weight, self.device
            )
    
    def transform_generator(self):
        """Generate image and target transforms for dataset loading."""
        import torchvision.transforms as transforms
        
        transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]

        target_transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)
    
    def setup_datasets(self):
        """Setup datasets based on training phase"""
        print(f"🔄 Setting up datasets for phase: {self.phase}")
        
        # Generate transforms
        train_transform, target_transform = self.transform_generator()
        test_transform, test_target_transform = self.transform_generator()
        
        if self.phase == 'extract_stats':
            # Phase 1: Extract statistics from BUSI (Institution A)
            self.train_dataset = BUSISegmentationDataset(
                dataset_dir=self.args.train_dataset_dir,
                mode='train',
                transform=train_transform,
                target_transform=target_transform
            )
            self.val_dataset = BUSISegmentationDataset(
                dataset_dir=self.args.test_dataset_dir,
                mode='test',
                transform=test_transform,
                target_transform=test_target_transform
            )
            print(f"   ✅ BUSI dataset loaded for statistics extraction")
            
        elif self.phase == 'federated_training':
            # Phase 2: Train on BUS-UCLM with feature alignment (Institution B)
            self.train_dataset = BUSUCLMSegmentationDataset(
                dataset_dir=self.args.train_dataset_dir,
                mode='train',
                transform=train_transform,
                target_transform=target_transform
            )
            self.val_dataset = BUSUCLMSegmentationDataset(
                dataset_dir=self.args.test_dataset_dir,
                mode='test',
                transform=test_transform,
                target_transform=test_target_transform
            )
            print(f"   ✅ BUS-UCLM dataset loaded for federated training")
            
        # Create data loaders
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        
        self.val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        
        print(f"   📊 Training samples: {len(self.train_dataset)}")
        print(f"   📊 Validation samples: {len(self.val_dataset)}")
    
    def extract_source_statistics(self):
        """Phase 1: Extract privacy-preserving feature statistics from source domain"""
        print("=" * 60)
        print("🔒 PHASE 1: EXTRACTING PRIVACY-PRESERVING FEATURE STATISTICS")
        print("=" * 60)
        
        # Initialize feature extractor
        self.feature_extractor = FederatedFeatureExtractor(
            self.model, self.privacy_config, self.device
        )
        
        # Extract statistics
        source_statistics = self.feature_extractor.extract_domain_statistics(
            self.train_data_loader, 
            domain_name='BUSI',
            compute_higher_moments=True
        )
        
        # Save statistics
        stats_output_path = f"federated_stats_{self.args.dataset_name}_{int(time.time())}.json"
        self.feature_extractor.save_statistics(source_statistics, stats_output_path)
        
        # Validate statistics on test set
        print("\n🔍 Validating statistics extraction...")
        test_statistics = self.feature_extractor.extract_domain_statistics(
            self.val_data_loader,
            domain_name='BUSI_test',
            compute_higher_moments=False
        )
        
        # Compare train vs test statistics (should be similar for good generalization)
        self._compare_statistics(source_statistics, test_statistics)
        
        # Cleanup
        self.feature_extractor.cleanup()
        
        print(f"\n✅ Statistics extraction completed!")
        print(f"📁 Statistics saved to: {stats_output_path}")
        print(f"🔒 Privacy budget spent: {self.feature_extractor.dp_mechanism.privacy_spent:.3f}")
        
        return stats_output_path
    
    def _compare_statistics(self, train_stats, test_stats):
        """Compare training and test statistics for validation"""
        print("   📊 Statistics comparison (train vs test):")
        
        for layer_name in train_stats:
            if layer_name in test_stats:
                train_mean = train_stats[layer_name].mean
                test_mean = test_stats[layer_name].mean
                
                # Compute relative difference
                mean_diff = torch.abs(train_mean - test_mean) / (torch.abs(train_mean) + 1e-8)
                avg_diff = mean_diff.mean().item()
                
                print(f"      {layer_name}: Mean difference = {avg_diff:.4f}")
                
                if avg_diff > 0.5:
                    print(f"         ⚠️  Large difference detected - possible overfitting")
                else:
                    print(f"         ✅ Good consistency")
    
    def federated_training_epoch(self, epoch):
        """Training epoch with federated feature alignment"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_align_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        for batch_idx, (images, masks) in enumerate(self.train_data_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Federated training step with feature alignment
            total_loss, metrics = self.domain_adapter.federated_training_step(
                images, masks, self.criterion
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += metrics['total_loss']
            epoch_seg_loss += metrics['segmentation_loss']
            epoch_align_loss += metrics['alignment_loss']
            
            # Compute segmentation metrics
            with torch.no_grad():
                predictions = self.model(images, mode='test')
                if isinstance(predictions, list):
                    predictions = predictions[-1]  # Use final prediction
                if isinstance(predictions, list):
                    predictions = predictions[0]  # Use map output
                
                dice, iou = compute_dice_iou_metrics(predictions, masks)
                epoch_dice += dice
                epoch_iou += iou
            
            # Logging
            if batch_idx % self.args.logging_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_data_loader)}: "
                      f"Loss={metrics['total_loss']:.4f} "
                      f"(Seg={metrics['segmentation_loss']:.4f}, "
                      f"Align={metrics['alignment_loss']:.4f})")
        
        # Average metrics
        num_batches = len(self.train_data_loader)
        epoch_metrics = {
            'train_loss': epoch_loss / num_batches,
            'train_seg_loss': epoch_seg_loss / num_batches,
            'train_align_loss': epoch_align_loss / num_batches,
            'train_dice': epoch_dice / num_batches,
            'train_iou': epoch_iou / num_batches
        }
        
        return epoch_metrics
    
    def validation_epoch(self, epoch):
        """Validation epoch"""
        self.model.eval()
        
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_data_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predictions = self.model(images, mode='test')
                if isinstance(predictions, list):
                    predictions = predictions[0]  # Use map output
                
                # Compute loss
                loss = self.criterion(predictions, masks)
                val_loss += loss.item()
                
                # Compute metrics
                dice, iou = compute_dice_iou_metrics(predictions, masks)
                val_dice += dice
                val_iou += iou
        
        # Average metrics
        num_batches = len(self.val_data_loader)
        val_metrics = {
            'val_loss': val_loss / num_batches,
            'val_dice': val_dice / num_batches,
            'val_iou': val_iou / num_batches
        }
        
        return val_metrics
    
    def run_federated_training(self):
        """Phase 2: Run federated training with feature alignment"""
        print("=" * 60)
        print("🚀 PHASE 2: FEDERATED TRAINING WITH FEATURE ALIGNMENT")
        print("=" * 60)
        
        if not self.domain_adapter:
            raise ValueError("Domain adapter not initialized! Ensure source statistics are loaded.")
        
        # Training loop
        best_dice = 0.0
        best_model_path = None
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print("-" * 40)
            
            # Training
            train_metrics = self.federated_training_epoch(epoch)
            
            # Validation
            val_metrics = self.validation_epoch(epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Logging
            print(f"Train - Loss: {train_metrics['train_loss']:.4f}, "
                  f"Dice: {train_metrics['train_dice']:.4f}, "
                  f"IoU: {train_metrics['train_iou']:.4f}")
            print(f"  Segmentation Loss: {train_metrics['train_seg_loss']:.4f}")
            print(f"  Alignment Loss: {train_metrics['train_align_loss']:.4f}")
            print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, "
                  f"Dice: {val_metrics['val_dice']:.4f}, "
                  f"IoU: {val_metrics['val_iou']:.4f}")
            
            # TensorBoard logging
            if hasattr(self, 'writer'):
                for key, value in epoch_metrics.items():
                    self.writer.add_scalar(key, value, epoch)
            
            # Save best model
            if val_metrics['val_dice'] > best_dice:
                best_dice = val_metrics['val_dice']
                best_model_path = f"federated_alignment_best_model_dice_{best_dice:.4f}_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': best_dice,
                    'metrics': epoch_metrics,
                    'privacy_config': vars(self.privacy_config),
                    'alignment_weight': self.alignment_weight
                }, best_model_path)
                print(f"   ✅ New best model saved: {best_model_path}")
            
            # Learning rate scheduling
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
        
        # Final results
        print("\n" + "=" * 60)
        print("🎯 FEDERATED TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best Validation Dice: {best_dice:.4f}")
        print(f"Best Model: {best_model_path}")
        
        # Cleanup
        self.domain_adapter.cleanup()
        
        return best_model_path, best_dice
    
    def run(self):
        """Main execution method"""
        if self.phase == 'extract_stats':
            return self.extract_source_statistics()
        elif self.phase == 'federated_training':
            return self.run_federated_training()
        else:
            raise ValueError(f"Unknown phase: {self.phase}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Federated Feature Alignment Training')
    
    # Phase selection
    parser.add_argument('--phase', type=str, required=True, 
                       choices=['extract_stats', 'federated_training'],
                       help='Training phase: extract_stats or federated_training')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='FederatedAlignment',
                       help='Dataset name for logging')
    parser.add_argument('--train_dataset_dir', type=str, required=True,
                       help='Training dataset directory')
    parser.add_argument('--test_dataset_dir', type=str, required=True,
                       help='Test dataset directory')
    parser.add_argument('--test_data_type', type=str, default='BUSI',
                       choices=['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB',
                               'DSB2018', 'MonuSeg2018', 'ISIC2018', 'PH2', 'COVID19', 'COVID19_2', 
                               'BUSI', 'STU', 'BUS-UCLM'],
                       help='Type of test dataset for evaluation')
    
    # Base class requirements
    parser.add_argument('--seed_fix', default=False, action='store_true',
                       help='Fix random seed for reproducibility')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of segmentation classes')
    parser.add_argument('--cnn_backbone', type=str, default='resnet50',
                       choices=['resnet50', 'res2net50_v1b_26w_4s', 'resnest50'],
                       help='CNN backbone architecture')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size for transforms')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1,
                       help='Test batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Federated learning arguments
    parser.add_argument('--alignment_weight', type=float, default=0.5,
                       help='Weight for feature alignment loss')
    parser.add_argument('--source_stats_path', type=str, default=None,
                       help='Path to source domain statistics (for federated training)')
    
    # Privacy arguments
    parser.add_argument('--privacy_epsilon', type=float, default=1.0,
                       help='Privacy budget (epsilon)')
    parser.add_argument('--privacy_delta', type=float, default=1e-5,
                       help='Privacy failure probability (delta)')
    parser.add_argument('--privacy_sensitivity', type=float, default=1.0,
                       help='L2 sensitivity for differential privacy')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--logging_interval', type=int, default=10,
                       help='Logging interval for training batches')
    
    # Model architecture arguments (for IS2D compatibility)
    parser.add_argument('--scale_branches', type=int, default=2)
    parser.add_argument('--frequency_branches', type=int, default=16)
    parser.add_argument('--frequency_selection', type=str, default='top')
    parser.add_argument('--block_repetition', type=int, default=1)
    parser.add_argument('--min_channel', type=int, default=64)
    parser.add_argument('--min_resolution', type=int, default=8)
    
    return parser.parse_args()


def main():
    """Main function"""
    print("🚀 Federated Feature Alignment Training")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if args.phase == 'federated_training' and not args.source_stats_path:
        raise ValueError("--source_stats_path required for federated_training phase")
    
    if args.phase == 'federated_training' and not os.path.exists(args.source_stats_path):
        raise FileNotFoundError(f"Source statistics file not found: {args.source_stats_path}")
    
    # Create experiment
    experiment = FederatedAlignment_IS2D(args)
    
    # Setup datasets
    experiment.setup_datasets()
    
    # Run experiment
    result = experiment.run()
    
    print(f"\n✅ Experiment completed successfully!")
    if args.phase == 'extract_stats':
        print(f"📁 Statistics saved to: {result}")
    else:
        model_path, best_dice = result
        print(f"🏆 Best model: {model_path}")
        print(f"🎯 Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main() 