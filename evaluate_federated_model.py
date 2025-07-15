"""
Federated Model Test Evaluation Script
=====================================

Evaluate the best federated alignment model on test set for final metrics.
Run this after federated training completes.

Usage:
    python evaluate_federated_model.py \
        --model_path federated_alignment_best_model_dice_0.XXXX_epoch_XX.pth \
        --test_dataset_dir dataset/BioMedicalDataset/BUS-UCLM \
        --test_data_type BUS-UCLM
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from IS2D_models import IS2D_model
from dataset.BioMedicalDataset.BUSUCLMSegmentationDataset import BUSUCLMSegmentationDataset
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
from utils.calculate_metrics import compute_dice_iou_metrics
from utils.get_functions import get_deivce

class FederatedModelEvaluator:
    """Evaluate federated alignment model on test set"""
    
    def __init__(self, args):
        self.args = args
        self.device = get_deivce()
        
        # Initialize model
        self.model = IS2D_model(
            cnn_backbone=args.cnn_backbone,
            num_classes=args.num_classes,
            scale_branches=args.scale_branches,
            frequency_branches=args.frequency_branches,
            frequency_selection=args.frequency_selection,
            block_repetition=args.block_repetition,
            min_channel=args.min_channel,
            min_resolution=args.min_resolution
        )
        self.model.to(self.device)
        
        # Load trained model
        self.load_model()
        
        # Setup test dataset
        self.setup_test_dataset()
    
    def load_model(self):
        """Load the best trained federated model"""
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")
        
        print(f"🔄 Loading trained model: {self.args.model_path}")
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"   ✅ Model loaded successfully")
    
    def setup_test_dataset(self):
        """Setup test dataset"""
        # Image transforms
        transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]
        target_transform_list = [
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ]
        
        image_transform = transforms.Compose(transform_list)
        target_transform = transforms.Compose(target_transform_list)
        
        # Create test dataset
        if self.args.test_data_type == 'BUS-UCLM':
            test_dataset = BUSUCLMSegmentationDataset(
                dataset_dir=self.args.test_dataset_dir,
                mode='test',
                transform=image_transform,
                target_transform=target_transform
            )
        elif self.args.test_data_type == 'BUSI':
            test_dataset = BUSISegmentationDataset(
                dataset_dir=self.args.test_dataset_dir,
                mode='test',
                transform=image_transform,
                target_transform=target_transform
            )
        else:
            raise ValueError(f"Unsupported test data type: {self.args.test_data_type}")
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=True
        )
        
        print(f"🔄 Test dataset setup complete")
        print(f"   📊 Test samples: {len(test_dataset)}")
        print(f"   📂 Dataset: {self.args.test_data_type}")
    
    def evaluate(self):
        """Evaluate model on test set"""
        print(f"\n🧪 FEDERATED MODEL TEST EVALUATION")
        print(f"=" * 50)
        
        self.model.eval()
        test_loss = 0.0
        test_dice = 0.0
        test_iou = 0.0
        criterion = nn.BCEWithLogitsLoss()
        
        all_dice_scores = []
        all_iou_scores = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.test_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predictions = self.model(images, mode='test')
                
                # Handle different prediction formats
                if isinstance(predictions, list):
                    predictions = predictions[0]  # Use map output
                elif isinstance(predictions, tuple):
                    predictions = predictions[0]  # Use first element
                
                # Compute loss (with raw logits)
                loss = criterion(predictions, masks)
                test_loss += loss.item()
                
                # Compute metrics (convert logits to probabilities)
                predictions_prob = torch.sigmoid(predictions)
                metrics_dict = compute_dice_iou_metrics(predictions_prob, masks)
                dice_score = metrics_dict['dice']
                iou_score = metrics_dict['iou']
                
                test_dice += dice_score
                test_iou += iou_score
                all_dice_scores.append(dice_score)
                all_iou_scores.append(iou_score)
                
                if batch_idx % 10 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(self.test_loader)} samples...")
        
        # Calculate final metrics
        num_samples = len(self.test_loader)
        final_loss = test_loss / num_samples
        final_dice = test_dice / num_samples
        final_iou = test_iou / num_samples
        
        # Calculate standard deviations
        import numpy as np
        dice_std = np.std(all_dice_scores)
        iou_std = np.std(all_iou_scores)
        
        # Print final results
        print(f"\n🎯 FINAL TEST RESULTS")
        print(f"=" * 50)
        print(f"📊 Test Samples: {num_samples}")
        print(f"📈 Test Loss: {final_loss:.4f}")
        print(f"🎯 Test Dice Score: {final_dice:.4f} ± {dice_std:.4f}")
        print(f"🎯 Test IoU Score: {final_iou:.4f} ± {iou_std:.4f}")
        print(f"📂 Dataset: {self.args.test_data_type}")
        print(f"🏆 Model: {os.path.basename(self.args.model_path)}")
        
        return {
            'test_loss': final_loss,
            'test_dice': final_dice,
            'test_iou': final_iou,
            'dice_std': dice_std,
            'iou_std': iou_std,
            'num_samples': num_samples
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Federated Model Test Evaluation')
    
    # Model and data arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained federated model (.pth file)')
    parser.add_argument('--test_dataset_dir', type=str, required=True,
                       help='Test dataset directory')
    parser.add_argument('--test_data_type', type=str, default='BUS-UCLM',
                       choices=['BUSI', 'BUS-UCLM'],
                       help='Type of test dataset')
    
    # Model configuration (should match training config)
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of segmentation classes')
    parser.add_argument('--cnn_backbone', type=str, default='resnet50',
                       choices=['resnet50', 'res2net50_v1b_26w_4s', 'resnest50'],
                       help='CNN backbone architecture')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # MADGNet parameters (should match training)
    parser.add_argument('--scale_branches', type=int, default=3)
    parser.add_argument('--frequency_branches', type=int, default=16)
    parser.add_argument('--frequency_selection', type=str, default='top')
    parser.add_argument('--block_repetition', type=int, default=1)
    parser.add_argument('--min_channel', type=int, default=32)
    parser.add_argument('--min_resolution', type=int, default=8)
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"🧪 Federated Model Test Evaluation")
    print(f"=" * 50)
    
    # Create evaluator
    evaluator = FederatedModelEvaluator(args)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    print(f"\n✅ Test evaluation completed!")
    print(f"🎯 Final Dice: {results['test_dice']:.4f}")

if __name__ == "__main__":
    main() 