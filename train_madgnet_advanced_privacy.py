#!/usr/bin/env python3
"""
Advanced Privacy-Preserving Training for MADGNet
=================================================

This script trains MADGNet using advanced privacy-preserving methods:
1. Frequency-Domain Privacy-Preserving Adaptation (FDA-PPA)
2. Enhanced loss functions with boundary detection
3. Real-time adaptation during training

Usage:
    python train_madgnet_advanced_privacy.py --config configs/train_config.py --method frequency
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import MADGNet components
from IS2D_models.mfmsnet import MFMSNet
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from dataset.BioMedicalDataset.BUSUCLMSegmentationDataset import BUSUCLMSegmentationDataset
from torchvision import transforms
from advanced_privacy_methods import (
    AdvancedPrivacyPreservingFramework,
    FrequencyDomainPrivacyAdapter,
    EnhancedLossFunction
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPrivacyTrainer:
    """
    Advanced Privacy-Preserving Trainer for MADGNet
    """
    
    def __init__(self, 
                 model_config,
                 train_config,
                 privacy_config,
                 device='cuda'):
        
        self.device = device
        self.model_config = model_config
        self.train_config = train_config
        self.privacy_config = privacy_config
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize privacy framework
        self.privacy_framework = AdvancedPrivacyPreservingFramework(
            model=self.model,
            **privacy_config
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        
        # Metrics tracking
        self.train_metrics = {'loss': [], 'dice': [], 'iou': []}
        self.val_metrics = {'loss': [], 'dice': [], 'iou': []}
        
        logger.info("✅ Advanced Privacy Trainer initialized")
        
    def _initialize_model(self):
        """Initialize MADGNet model"""
        model = MFMSNet(
            num_classes=self.model_config.get('num_classes', 1),
            scale_branches=self.model_config.get('scale_branches', 2),
            frequency_branches=self.model_config.get('frequency_branches', 16),
            frequency_selection=self.model_config.get('frequency_selection', 'top'),
            block_repetition=self.model_config.get('block_repetition', 1),
            min_channel=self.model_config.get('min_channel', 64),
            min_resolution=self.model_config.get('min_resolution', 8),
            cnn_backbone=self.model_config.get('cnn_backbone', 'resnet50')
        )
        
        model = model.to(self.device)
        
        # Load pretrained weights if available
        if self.train_config.get('pretrained_path'):
            checkpoint = torch.load(self.train_config['pretrained_path'], map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"✅ Loaded pretrained weights from {self.train_config['pretrained_path']}")
            
        return model
    
    def _initialize_optimizer(self):
        """Initialize optimizer"""
        optimizer_type = self.train_config.get('optimizer', 'adam')
        lr = self.train_config.get('learning_rate', 1e-4)
        weight_decay = self.train_config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        return optimizer
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler"""
        scheduler_type = self.train_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.train_config.get('epochs', 100)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.train_config.get('step_size', 30),
                gamma=self.train_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
            
        return scheduler
    
    def prepare_source_statistics(self, source_dataloader, stats_save_path):
        """
        Extract and save source domain frequency statistics.
        This is done ONCE on the source domain (BUSI).
        """
        logger.info("🔊 Extracting source domain frequency statistics...")
        self.privacy_framework.prepare_source_statistics(source_dataloader, stats_save_path)
        logger.info(f"✅ Source statistics saved to {stats_save_path}")
    
    def load_source_statistics(self, stats_path):
        """Load pre-computed source domain statistics"""
        self.privacy_framework.load_source_statistics(stats_path)
        logger.info(f"✅ Loaded source statistics from {stats_path}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with privacy-preserving adaptations"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Apply privacy-preserving adaptations
            if self.privacy_config.get('use_frequency_adaptation', True):
                images = self.privacy_framework.adapt_batch(images, masks)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # MFMSNet returns [map1, dist1, bound1], [map2, dist2, bound2], [map3, dist3, bound3], [map4, dist4, bound4]
            outputs = self.model(images, mode='train')
            
            # Extract main predictions (maps from all stages)
            main_prediction = outputs[3][0]  # Final stage map output
            deep_predictions = [stage[0] for stage in outputs[:3]]  # Earlier stage map outputs
            
            # Compute enhanced loss for main prediction
            if hasattr(self.privacy_framework, 'compute_enhanced_loss'):
                main_loss, loss_components = self.privacy_framework.compute_enhanced_loss(
                    main_prediction, masks
                )
            else:
                # Fallback to standard loss
                main_loss = self._compute_standard_loss(main_prediction, masks)
                loss_components = {'total_loss': main_loss.item()}
            
            # Add deep supervision loss from earlier stages
            total_loss = main_loss
            if deep_predictions:
                deep_loss = 0
                for deep_pred in deep_predictions:
                    deep_pred_resized = torch.nn.functional.interpolate(
                        deep_pred, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                    if hasattr(self.privacy_framework, 'compute_enhanced_loss'):
                        dl, _ = self.privacy_framework.compute_enhanced_loss(deep_pred_resized, masks)
                    else:
                        dl = self._compute_standard_loss(deep_pred_resized, masks)
                    deep_loss += dl
                
                total_loss = main_loss + 0.3 * deep_loss  # Weighted deep supervision
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.train_config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                prediction_prob = torch.sigmoid(main_prediction)
                prediction_binary = (prediction_prob > 0.5).float()
                
                dice_score = self._compute_dice(prediction_binary, masks)
                iou_score = self._compute_iou(prediction_binary, masks)
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_dice += dice_score
            epoch_iou += iou_score
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Dice': f'{dice_score:.4f}',
                'IoU': f'{iou_score:.4f}'
            })
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_iou = epoch_iou / num_batches
        
        return avg_loss, avg_dice, avg_iou
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Validation {epoch+1}')
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Apply privacy-preserving adaptations (same as training)
                if self.privacy_config.get('use_frequency_adaptation', True):
                    images = self.privacy_framework.adapt_batch(images, masks)
                
                # Forward pass (use test mode for single output)
                outputs = self.model(images, mode='test')
                prediction = outputs  # In test mode, returns single map output
                
                # Compute loss
                if hasattr(self.privacy_framework, 'compute_enhanced_loss'):
                    loss, _ = self.privacy_framework.compute_enhanced_loss(prediction, masks)
                else:
                    loss = self._compute_standard_loss(prediction, masks)
                
                # Compute metrics
                prediction_prob = torch.sigmoid(prediction)
                prediction_binary = (prediction_prob > 0.5).float()
                
                dice_score = self._compute_dice(prediction_binary, masks)
                iou_score = self._compute_iou(prediction_binary, masks)
                
                epoch_loss += loss.item()
                epoch_dice += dice_score
                epoch_iou += iou_score
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice_score:.4f}',
                    'IoU': f'{iou_score:.4f}'
                })
        
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_iou = epoch_iou / num_batches
        
        return avg_loss, avg_dice, avg_iou
    
    def _compute_standard_loss(self, pred, target):
        """Compute standard segmentation loss (fallback)"""
        # Dice loss
        pred_prob = torch.sigmoid(pred)
        smooth = 1e-6
        
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice
        
        # BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
        
        return dice_loss + bce_loss
    
    def _compute_dice(self, pred, target):
        """Compute Dice coefficient"""
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return dice.item()
    
    def _compute_iou(self, pred, target):
        """Compute IoU (Jaccard index)"""
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.item()
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        logger.info(f"🚀 Starting training for {num_epochs} epochs")
        
        best_dice = 0.0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_dice, train_iou = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            self.train_metrics['loss'].append(train_loss)
            self.train_metrics['dice'].append(train_dice)
            self.train_metrics['iou'].append(train_iou)
            
            self.val_metrics['loss'].append(val_loss)
            self.val_metrics['dice'].append(val_dice)
            self.val_metrics['iou'].append(val_iou)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"✅ New best model! Dice: {best_dice:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.train_config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info(f"🎉 Training complete! Best Dice: {best_dice:.4f} at epoch {best_epoch+1}")
        return best_dice
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'privacy_config': self.privacy_config,
            'model_config': self.model_config,
            'train_config': self.train_config
        }
        
        # Create output directory
        output_dir = Path(self.train_config.get('output_dir', 'checkpoints'))
        output_dir.mkdir(exist_ok=True)
        
        # Save checkpoint
        if is_best:
            save_path = output_dir / 'best_model.pth'
            torch.save(checkpoint, save_path)
            logger.info(f"💾 Best model saved to {save_path}")
        else:
            save_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, save_path)
            logger.info(f"💾 Checkpoint saved to {save_path}")
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        epochs = range(1, len(self.train_metrics['loss']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss
        axes[0].plot(epochs, self.train_metrics['loss'], label='Train')
        axes[0].plot(epochs, self.val_metrics['loss'], label='Validation')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid()
        
        # Dice
        axes[1].plot(epochs, self.train_metrics['dice'], label='Train')
        axes[1].plot(epochs, self.val_metrics['dice'], label='Validation')
        axes[1].set_title('Dice Score')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice')
        axes[1].legend()
        axes[1].grid()
        
        # IoU
        axes[2].plot(epochs, self.train_metrics['iou'], label='Train')
        axes[2].plot(epochs, self.val_metrics['iou'], label='Validation')
        axes[2].set_title('IoU Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('IoU')
        axes[2].legend()
        axes[2].grid()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Metrics plot saved to {save_path}")
        
        plt.show()

def create_dataloaders(config):
    """Create train and validation dataloaders"""
    
    # Dataset paths
    dataset_root = config['dataset_root']
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Determine dataset class based on dataset type
    if 'BUSI' in dataset_root:
        DatasetClass = BUSISegmentationDataset
    else:
        DatasetClass = BUSUCLMSegmentationDataset
    
    # Create datasets
    train_dataset = DatasetClass(
        dataset_dir=dataset_root,
        mode='train',
        transform=train_transform,
        target_transform=val_transform
    )
    
    val_dataset = DatasetClass(
        dataset_dir=dataset_root,
        mode='val',
        transform=val_transform,
        target_transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 2),  # Reduced for stability
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 2),  # Reduced for stability
        pin_memory=True
    )
    
    logger.info(f"✅ Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Advanced Privacy-Preserving Training for MADGNet')
    parser.add_argument('--dataset', choices=['busi', 'bus_uclm'], default='bus_uclm',
                       help='Target dataset to train on')
    parser.add_argument('--source_stats', default='privacy_style_stats/busi_privacy_stats.json',
                       help='Path to source domain frequency statistics')
    parser.add_argument('--output_dir', default='advanced_privacy_training_output',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (use 3 for testing, 100+ for full training)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--adaptation_strength', type=float, default=0.7,
                       help='Frequency adaptation strength (0.0-1.0)')
    parser.add_argument('--method', choices=['frequency', 'knowledge', 'self_supervised', 'all'], 
                       default='frequency', help='Privacy-preserving method to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    model_config = {
        'num_classes': 1,
        'scale_branches': 2,
        'frequency_branches': 16,
        'frequency_selection': 'top',
        'block_repetition': 1,
        'min_channel': 64,
        'min_resolution': 8,
        'cnn_backbone': 'resnet50'
    }
    
    train_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'batch_size': args.batch_size,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'save_every': max(1, args.epochs // 10),  # Save 10 times during training
        'output_dir': str(output_dir),
        'dataset_root': f'dataset/BioMedicalDataset/{args.dataset.upper()}' if args.dataset == 'busi' else 'dataset/BioMedicalDataset/BUS-UCLM'
    }
    
    privacy_config = {
        'use_frequency_adaptation': args.method in ['frequency', 'all'],
        'use_knowledge_distillation': args.method in ['knowledge', 'all'],
        'use_domain_alignment': args.method in ['self_supervised', 'all'],
        'num_frequency_bands': 8,
        'adaptation_strength': args.adaptation_strength
    }
    
    # Initialize trainer
    trainer = AdvancedPrivacyTrainer(
        model_config=model_config,
        train_config=train_config,
        privacy_config=privacy_config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load source domain statistics if using frequency adaptation
    if privacy_config['use_frequency_adaptation']:
        if os.path.exists(args.source_stats):
            trainer.load_source_statistics(args.source_stats)
            logger.info(f"✅ Loaded source frequency statistics from {args.source_stats}")
        else:
            logger.warning(f"⚠️  Source statistics not found: {args.source_stats}")
            logger.info("   Run prepare_source_statistics.py first to extract BUSI statistics")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_config)
    
    # Start training
    logger.info(f"🚀 Starting Advanced Privacy-Preserving Training")
    logger.info(f"   Target Dataset: {args.dataset.upper()}")
    logger.info(f"   Method: {args.method}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Adaptation Strength: {args.adaptation_strength}")
    
    best_dice = trainer.train(train_loader, val_loader, args.epochs)
    
    # Plot and save metrics
    trainer.plot_metrics(save_path=output_dir / 'training_metrics.png')
    
    # Save final metrics
    metrics_path = output_dir / 'final_metrics.json'
    final_metrics = {
        'best_dice': best_dice,
        'final_train_dice': trainer.train_metrics['dice'][-1],
        'final_val_dice': trainer.val_metrics['dice'][-1],
        'final_train_iou': trainer.train_metrics['iou'][-1],
        'final_val_iou': trainer.val_metrics['iou'][-1],
        'config': {
            'model_config': model_config,
            'train_config': train_config,
            'privacy_config': privacy_config
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"🎉 Training completed successfully!")
    logger.info(f"   Best Dice Score: {best_dice:.4f}")
    logger.info(f"   Results saved to: {output_dir}")

if __name__ == '__main__':
    main() 