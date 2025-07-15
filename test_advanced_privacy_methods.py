#!/usr/bin/env python3
"""
Test Advanced Privacy-Preserving Methods
=========================================

This script tests and validates the advanced privacy-preserving methods:
1. Performance comparison (before/after adaptation)
2. Privacy analysis and metrics
3. Frequency domain analysis
4. Quality assessment

Usage:
    python test_advanced_privacy_methods.py --model_path checkpoints/best_model.pth --dataset bus_uclm
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cv2
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple

# Import components
from IS2D_models.mfmsnet import MFMSNet
from utils.load_functions import BUSIDataset
from torch.utils.data import DataLoader
from advanced_privacy_methods import (
    AdvancedPrivacyPreservingFramework,
    FrequencyDomainPrivacyAdapter
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPrivacyTester:
    """
    Comprehensive tester for advanced privacy-preserving methods
    """
    
    def __init__(self, model_path, config, device='cuda'):
        self.device = device
        self.config = config
        self.model_path = model_path
        
        # Load model
        self.model = self._load_model()
        
        # Initialize privacy framework
        self.privacy_framework = AdvancedPrivacyPreservingFramework(
            model=self.model,
            **config.get('privacy_config', {})
        )
        
        # Results storage
        self.results = {
            'performance': {},
            'privacy': {},
            'quality': {},
            'frequency_analysis': {}
        }
        
        logger.info("✅ Advanced Privacy Tester initialized")
    
    def _load_model(self):
        """Load trained model"""
        # Initialize model
        model = MFMSNet(
            num_classes=self.config.get('num_classes', 1),
            input_channels=self.config.get('input_channels', 1),
            deep_supervision=self.config.get('deep_supervision', True)
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("📊 Loaded model with training history")
        else:
            model.load_state_dict(checkpoint)
            logger.info("📊 Loaded model weights only")
        
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"✅ Model loaded from {self.model_path}")
        return model
    
    def test_without_adaptation(self, test_loader):
        """Test model performance without privacy adaptations"""
        logger.info("🔍 Testing without privacy adaptations...")
        
        metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Testing baseline"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass without adaptation
                outputs = self.model(images)
                prediction = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                
                # Compute metrics
                prediction_prob = torch.sigmoid(prediction)
                prediction_binary = (prediction_prob > 0.5).float()
                
                batch_metrics = self._compute_batch_metrics(prediction_binary, masks)
                
                for key in metrics:
                    metrics[key].extend(batch_metrics[key])
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        logger.info(f"📊 Baseline Results:")
        for key, value in avg_metrics.items():
            logger.info(f"   {key.upper()}: {value:.4f}")
        
        return avg_metrics
    
    def test_with_adaptation(self, test_loader, source_stats_path):
        """Test model performance with privacy adaptations"""
        logger.info("🔊 Testing with privacy adaptations...")
        
        # Load source statistics
        self.privacy_framework.load_source_statistics(source_stats_path)
        
        metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Testing with adaptation"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Apply privacy adaptations
                adapted_images = self.privacy_framework.adapt_batch(images, masks)
                
                # Forward pass with adapted images
                outputs = self.model(adapted_images)
                prediction = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                
                # Compute metrics
                prediction_prob = torch.sigmoid(prediction)
                prediction_binary = (prediction_prob > 0.5).float()
                
                batch_metrics = self._compute_batch_metrics(prediction_binary, masks)
                
                for key in metrics:
                    metrics[key].extend(batch_metrics[key])
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        logger.info(f"📊 Privacy-Adapted Results:")
        for key, value in avg_metrics.items():
            logger.info(f"   {key.upper()}: {value:.4f}")
        
        return avg_metrics
    
    def _compute_batch_metrics(self, pred, target):
        """Compute metrics for a batch"""
        batch_size = pred.shape[0]
        metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
        
        for i in range(batch_size):
            p = pred[i].cpu().numpy().flatten()
            t = target[i].cpu().numpy().flatten()
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(t, p, labels=[0, 1]).ravel()
            
            # Dice
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            metrics['dice'].append(dice)
            
            # IoU
            iou = tp / (tp + fp + fn + 1e-8)
            metrics['iou'].append(iou)
            
            # Precision
            precision = tp / (tp + fp + 1e-8)
            metrics['precision'].append(precision)
            
            # Recall
            recall = tp / (tp + fn + 1e-8)
            metrics['recall'].append(recall)
            
            # F1
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            metrics['f1'].append(f1)
        
        return metrics
    
    def analyze_privacy_guarantees(self, source_stats_path):
        """Analyze privacy guarantees of the frequency adaptation method"""
        logger.info("🔒 Analyzing privacy guarantees...")
        
        # Load frequency statistics
        with open(source_stats_path, 'r') as f:
            stats = json.load(f)
        
        # Calculate privacy metrics
        num_stats = len(stats)
        typical_image_pixels = 256 * 256 * 1  # Assuming grayscale 256x256
        compression_ratio = typical_image_pixels / num_stats
        information_loss = (1 - num_stats / typical_image_pixels) * 100
        
        privacy_analysis = {
            'statistics_count': num_stats,
            'typical_image_pixels': typical_image_pixels,
            'compression_ratio': f"{compression_ratio:,.0f}:1",
            'information_loss_percent': f"{information_loss:.3f}%",
            'shared_data_size_kb': len(json.dumps(stats).encode('utf-8')) / 1024,
            'privacy_level': 'HIGH' if compression_ratio > 1000 else 'MEDIUM' if compression_ratio > 100 else 'LOW'
        }
        
        logger.info("🔒 Privacy Analysis:")
        logger.info(f"   Statistics shared: {privacy_analysis['statistics_count']} numbers")
        logger.info(f"   Compression ratio: {privacy_analysis['compression_ratio']}")
        logger.info(f"   Information loss: {privacy_analysis['information_loss_percent']}")
        logger.info(f"   Data size: {privacy_analysis['shared_data_size_kb']:.2f} KB")
        logger.info(f"   Privacy level: {privacy_analysis['privacy_level']}")
        
        self.results['privacy'] = privacy_analysis
        
        return privacy_analysis
    
    def analyze_frequency_domain(self, test_loader, source_stats_path, num_samples=5):
        """Analyze frequency domain transformations"""
        logger.info("🌊 Analyzing frequency domain transformations...")
        
        # Load source statistics
        self.privacy_framework.load_source_statistics(source_stats_path)
        
        analysis_results = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                if batch_idx >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # Original image frequency analysis
                original_freq_stats = self._analyze_image_frequency(images[0:1])
                
                # Adapted image frequency analysis
                adapted_images = self.privacy_framework.adapt_batch(images[0:1])
                adapted_freq_stats = self._analyze_image_frequency(adapted_images)
                
                analysis_results.append({
                    'sample_id': batch_idx,
                    'original': original_freq_stats,
                    'adapted': adapted_freq_stats,
                    'adaptation_effect': self._compute_adaptation_effect(
                        original_freq_stats, adapted_freq_stats
                    )
                })
        
        # Aggregate results
        freq_analysis = self._aggregate_frequency_analysis(analysis_results)
        
        logger.info("🌊 Frequency Analysis:")
        logger.info(f"   Average frequency shift: {freq_analysis['avg_frequency_shift']:.4f}")
        logger.info(f"   Spectral similarity: {freq_analysis['spectral_similarity']:.4f}")
        logger.info(f"   Adaptation strength: {freq_analysis['effective_adaptation_strength']:.4f}")
        
        self.results['frequency_analysis'] = freq_analysis
        
        return freq_analysis
    
    def _analyze_image_frequency(self, image):
        """Analyze frequency domain characteristics of an image"""
        # Convert to numpy
        img_np = image.cpu().numpy().squeeze()
        
        # 2D FFT
        freq_domain = np.fft.fft2(img_np)
        magnitude = np.abs(freq_domain)
        phase = np.angle(freq_domain)
        
        # Frequency statistics
        stats = {
            'mean_magnitude': np.mean(magnitude),
            'std_magnitude': np.std(magnitude),
            'max_magnitude': np.max(magnitude),
            'mean_phase': np.mean(phase),
            'std_phase': np.std(phase),
            'spectral_energy': np.sum(magnitude ** 2),
            'spectral_entropy': self._compute_spectral_entropy(magnitude)
        }
        
        return stats
    
    def _compute_spectral_entropy(self, magnitude):
        """Compute spectral entropy"""
        # Normalize magnitude spectrum
        normalized = magnitude / (np.sum(magnitude) + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(normalized * np.log(normalized + 1e-8))
        
        return entropy
    
    def _compute_adaptation_effect(self, original, adapted):
        """Compute adaptation effect metrics"""
        effects = {}
        
        for key in original:
            if key in adapted:
                original_val = original[key]
                adapted_val = adapted[key]
                
                if original_val != 0:
                    relative_change = (adapted_val - original_val) / original_val
                    effects[f'{key}_relative_change'] = relative_change
                else:
                    effects[f'{key}_absolute_change'] = adapted_val - original_val
        
        return effects
    
    def _aggregate_frequency_analysis(self, analysis_results):
        """Aggregate frequency analysis results"""
        # Collect all adaptation effects
        all_effects = []
        spectral_similarities = []
        
        for result in analysis_results:
            effect = result['adaptation_effect']
            
            # Calculate frequency shift (magnitude change)
            if 'mean_magnitude_relative_change' in effect:
                all_effects.append(abs(effect['mean_magnitude_relative_change']))
            
            # Calculate spectral similarity
            orig_energy = result['original']['spectral_energy']
            adapt_energy = result['adapted']['spectral_energy']
            
            if orig_energy > 0:
                similarity = min(orig_energy, adapt_energy) / max(orig_energy, adapt_energy)
                spectral_similarities.append(similarity)
        
        aggregated = {
            'avg_frequency_shift': np.mean(all_effects) if all_effects else 0.0,
            'spectral_similarity': np.mean(spectral_similarities) if spectral_similarities else 0.0,
            'effective_adaptation_strength': np.mean(all_effects) if all_effects else 0.0,
            'num_samples_analyzed': len(analysis_results)
        }
        
        return aggregated
    
    def assess_visual_quality(self, test_loader, source_stats_path, num_samples=3):
        """Assess visual quality of adapted images"""
        logger.info("🎨 Assessing visual quality...")
        
        # Load source statistics
        self.privacy_framework.load_source_statistics(source_stats_path)
        
        quality_metrics = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                if batch_idx >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # Get adapted images
                adapted_images = self.privacy_framework.adapt_batch(images)
                
                # Compute quality metrics for first image in batch
                original = images[0].cpu().numpy().squeeze()
                adapted = adapted_images[0].cpu().numpy().squeeze()
                
                # Normalize to [0, 1]
                original = (original - original.min()) / (original.max() - original.min() + 1e-8)
                adapted = (adapted - adapted.min()) / (adapted.max() - adapted.min() + 1e-8)
                
                # Compute quality metrics
                mse = np.mean((original - adapted) ** 2)
                psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
                
                # Structural similarity (simplified)
                ssim = self._compute_simple_ssim(original, adapted)
                
                quality_metrics.append({
                    'sample_id': batch_idx,
                    'mse': mse,
                    'psnr': psnr,
                    'ssim': ssim,
                    'adaptation_preserves_structure': ssim > 0.8
                })
        
        # Aggregate quality metrics
        avg_quality = {
            'avg_mse': np.mean([m['mse'] for m in quality_metrics]),
            'avg_psnr': np.mean([m['psnr'] for m in quality_metrics]),
            'avg_ssim': np.mean([m['ssim'] for m in quality_metrics]),
            'structure_preservation_rate': np.mean([m['adaptation_preserves_structure'] for m in quality_metrics]) * 100
        }
        
        logger.info("🎨 Visual Quality Assessment:")
        logger.info(f"   Average PSNR: {avg_quality['avg_psnr']:.2f} dB")
        logger.info(f"   Average SSIM: {avg_quality['avg_ssim']:.4f}")
        logger.info(f"   Structure preservation: {avg_quality['structure_preservation_rate']:.1f}%")
        
        self.results['quality'] = avg_quality
        
        return avg_quality
    
    def _compute_simple_ssim(self, img1, img2, k1=0.01, k2=0.03):
        """Compute simplified SSIM"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (k1) ** 2
        c2 = (k2) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def create_comparison_visualizations(self, test_loader, source_stats_path, save_dir):
        """Create visual comparisons of original vs adapted images"""
        logger.info("📊 Creating comparison visualizations...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Load source statistics
        self.privacy_framework.load_source_statistics(source_stats_path)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                if batch_idx >= 3:  # Limit to 3 examples
                    break
                
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Get adapted images
                adapted_images = self.privacy_framework.adapt_batch(images)
                
                # Get predictions for both
                orig_pred = self.model(images)
                adapt_pred = self.model(adapted_images)
                
                if isinstance(orig_pred, (list, tuple)):
                    orig_pred = orig_pred[0]
                if isinstance(adapt_pred, (list, tuple)):
                    adapt_pred = adapt_pred[0]
                
                orig_pred = torch.sigmoid(orig_pred)
                adapt_pred = torch.sigmoid(adapt_pred)
                
                # Create visualization
                self._create_sample_visualization(
                    images[0], adapted_images[0], masks[0],
                    orig_pred[0], adapt_pred[0],
                    save_dir / f'comparison_sample_{batch_idx}.png'
                )
        
        logger.info(f"📊 Visualizations saved to {save_dir}")
    
    def _create_sample_visualization(self, orig_img, adapt_img, mask, orig_pred, adapt_pred, save_path):
        """Create a single sample visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Convert tensors to numpy
        orig_img = orig_img.cpu().numpy().squeeze()
        adapt_img = adapt_img.cpu().numpy().squeeze()
        mask = mask.cpu().numpy().squeeze()
        orig_pred = orig_pred.cpu().numpy().squeeze()
        adapt_pred = adapt_pred.cpu().numpy().squeeze()
        
        # Original row
        axes[0, 0].imshow(orig_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(orig_pred, cmap='gray')
        axes[0, 2].set_title(f'Original Prediction')
        axes[0, 2].axis('off')
        
        # Adapted row
        axes[1, 0].imshow(adapt_img, cmap='gray')
        axes[1, 0].set_title('Adapted Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('Ground Truth')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(adapt_pred, cmap='gray')
        axes[1, 2].set_title(f'Adapted Prediction')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, save_path):
        """Generate comprehensive test report"""
        logger.info("📋 Generating comprehensive report...")
        
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'config': self.config,
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Comprehensive report saved to {save_path}")
        
        return report
    
    def _generate_summary(self):
        """Generate summary of test results"""
        summary = {
            'performance_improvement': {},
            'privacy_assessment': 'UNKNOWN',
            'quality_assessment': 'UNKNOWN',
            'recommendations': []
        }
        
        # Performance comparison
        if 'baseline' in self.results['performance'] and 'adapted' in self.results['performance']:
            baseline = self.results['performance']['baseline']
            adapted = self.results['performance']['adapted']
            
            for metric in ['dice', 'iou', 'f1']:
                if metric in baseline and metric in adapted:
                    improvement = adapted[metric] - baseline[metric]
                    improvement_pct = (improvement / baseline[metric]) * 100
                    summary['performance_improvement'][metric] = {
                        'absolute': improvement,
                        'relative_percent': improvement_pct
                    }
        
        # Privacy assessment
        if 'privacy' in self.results:
            privacy = self.results['privacy']
            if 'privacy_level' in privacy:
                summary['privacy_assessment'] = privacy['privacy_level']
        
        # Quality assessment
        if 'quality' in self.results:
            quality = self.results['quality']
            if 'avg_ssim' in quality:
                if quality['avg_ssim'] > 0.9:
                    summary['quality_assessment'] = 'EXCELLENT'
                elif quality['avg_ssim'] > 0.8:
                    summary['quality_assessment'] = 'GOOD'
                elif quality['avg_ssim'] > 0.6:
                    summary['quality_assessment'] = 'ACCEPTABLE'
                else:
                    summary['quality_assessment'] = 'POOR'
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        if 'performance' in self.results:
            perf = self.results['performance']
            if 'baseline' in perf and 'adapted' in perf:
                dice_improvement = perf['adapted'].get('dice', 0) - perf['baseline'].get('dice', 0)
                
                if dice_improvement > 0.02:
                    recommendations.append("EXCELLENT: Significant performance improvement achieved")
                elif dice_improvement > 0.005:
                    recommendations.append("GOOD: Moderate performance improvement achieved")
                elif dice_improvement > -0.005:
                    recommendations.append("ACCEPTABLE: Performance maintained with privacy benefits")
                else:
                    recommendations.append("CONCERN: Performance degraded, consider tuning adaptation strength")
        
        # Privacy recommendations
        if 'privacy' in self.results:
            privacy_level = self.results['privacy'].get('privacy_level', 'UNKNOWN')
            if privacy_level == 'HIGH':
                recommendations.append("EXCELLENT: High privacy level achieved")
            elif privacy_level == 'MEDIUM':
                recommendations.append("GOOD: Acceptable privacy level")
            else:
                recommendations.append("IMPROVE: Consider reducing shared information for better privacy")
        
        # Quality recommendations
        if 'quality' in self.results:
            ssim = self.results['quality'].get('avg_ssim', 0)
            if ssim < 0.8:
                recommendations.append("TUNE: Consider reducing adaptation strength to preserve image quality")
        
        return recommendations

def create_test_dataloader(config):
    """Create test dataloader"""
    test_csv = config.get('test_csv', config.get('val_csv'))  # Use val if test not available
    dataset_root = config['dataset_root']
    
    test_dataset = BUSIDataset(
        csv_file=test_csv,
        dataset_dir=dataset_root,
        transform_prob=0.0,  # No augmentation for testing
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch size 1 for detailed analysis
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"✅ Test dataloader created with {len(test_dataset)} samples")
    
    return test_loader

def main():
    parser = argparse.ArgumentParser(description='Test Advanced Privacy-Preserving Methods')
    parser.add_argument('--model_path', required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', choices=['busi', 'bus_uclm'], default='bus_uclm',
                       help='Test dataset')
    parser.add_argument('--source_stats', default='privacy_style_stats/busi_privacy_stats.json',
                       help='Path to source domain frequency statistics')
    parser.add_argument('--output_dir', default='test_results',
                       help='Output directory for test results')
    parser.add_argument('--adaptation_strength', type=float, default=0.7,
                       help='Frequency adaptation strength for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'num_classes': 1,
        'input_channels': 1,
        'deep_supervision': True,
        'dataset_root': f'dataset/{args.dataset.upper()}' if args.dataset == 'busi' else 'dataset/BUS-UCLM',
        'test_csv': f'dataset/{args.dataset.upper()}/test_frame.csv' if args.dataset == 'busi' else 'dataset/BUS-UCLM/val_frame.csv',
        'val_csv': f'dataset/{args.dataset.upper()}/val_frame.csv' if args.dataset == 'busi' else 'dataset/BUS-UCLM/val_frame.csv',
        'privacy_config': {
            'use_frequency_adaptation': True,
            'adaptation_strength': args.adaptation_strength,
            'num_frequency_bands': 8
        }
    }
    
    # Initialize tester
    tester = AdvancedPrivacyTester(
        model_path=args.model_path,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create test dataloader
    test_loader = create_test_dataloader(config)
    
    logger.info(f"🧪 Starting comprehensive testing")
    logger.info(f"   Model: {args.model_path}")
    logger.info(f"   Dataset: {args.dataset.upper()}")
    logger.info(f"   Source stats: {args.source_stats}")
    
    # Test without adaptation (baseline)
    baseline_results = tester.test_without_adaptation(test_loader)
    tester.results['performance']['baseline'] = baseline_results
    
    # Test with adaptation
    if os.path.exists(args.source_stats):
        adapted_results = tester.test_with_adaptation(test_loader, args.source_stats)
        tester.results['performance']['adapted'] = adapted_results
        
        # Privacy analysis
        tester.analyze_privacy_guarantees(args.source_stats)
        
        # Frequency domain analysis
        tester.analyze_frequency_domain(test_loader, args.source_stats)
        
        # Visual quality assessment
        tester.assess_visual_quality(test_loader, args.source_stats)
        
        # Create visualizations
        tester.create_comparison_visualizations(
            test_loader, args.source_stats, output_dir / 'visualizations'
        )
        
    else:
        logger.warning(f"⚠️  Source statistics not found: {args.source_stats}")
        logger.info("   Skipping adaptation tests")
    
    # Generate comprehensive report
    report = tester.generate_comprehensive_report(output_dir / 'test_report.json')
    
    # Print summary
    logger.info("🎉 Testing completed!")
    logger.info("=" * 50)
    
    if 'baseline' in tester.results['performance']:
        baseline = tester.results['performance']['baseline']
        logger.info(f"📊 Baseline Performance:")
        logger.info(f"   Dice: {baseline['dice']:.4f}")
        logger.info(f"   IoU:  {baseline['iou']:.4f}")
    
    if 'adapted' in tester.results['performance']:
        adapted = tester.results['performance']['adapted']
        logger.info(f"📊 Privacy-Adapted Performance:")
        logger.info(f"   Dice: {adapted['dice']:.4f}")
        logger.info(f"   IoU:  {adapted['iou']:.4f}")
        
        # Show improvement
        if 'baseline' in tester.results['performance']:
            dice_improvement = adapted['dice'] - baseline['dice']
            iou_improvement = adapted['iou'] - baseline['iou']
            logger.info(f"📈 Improvement:")
            logger.info(f"   Dice: {dice_improvement:+.4f}")
            logger.info(f"   IoU:  {iou_improvement:+.4f}")
    
    if 'privacy' in tester.results:
        privacy = tester.results['privacy']
        logger.info(f"🔒 Privacy Level: {privacy.get('privacy_level', 'UNKNOWN')}")
        logger.info(f"🔒 Compression: {privacy.get('compression_ratio', 'N/A')}")
    
    logger.info(f"📋 Detailed results saved to: {output_dir}")

if __name__ == '__main__':
    main() 