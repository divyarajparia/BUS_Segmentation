#!/usr/bin/env python3
"""
CCST BUS-UCLM Augmentation Pipeline
===================================

Uses privacy-preserving CCST style transfer to:
1. Extract BUSI style statistics (privacy-preserving)  
2. Style-transfer BUS-UCLM training data to BUSI style
3. Train MADGNet on: BUS-UCLM + styled-BUS-UCLM  
4. Test on BUS-UCLM test set
5. Compare against BUS-UCLM-only baseline

Goal: Prove BUSI knowledge improves BUS-UCLM performance while preserving privacy.
"""

import os
import sys
import torch
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ccst_privacy_preserving_adain import (
    PrivacyPreservingStyleExtractor,
    AdaINStyleTransfer, 
    StyleTransferDatasetGenerator
)
from IS2D_main import main as train_is2d
from utils.calculate_metrics import compute_dice_iou_metrics

class CCSTargetedPipeline:
    """
    CCST Pipeline specifically for BUS-UCLM augmentation using BUSI knowledge
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Paths
        self.busi_path = "dataset/BioMedicalDataset/BUSI"
        self.bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
        self.output_base = f"ccst_busuclm_augmentation_{self.timestamp}"
        
        # Results storage
        self.results = {
            'timestamp': self.timestamp,
            'approach': 'CCST_Privacy_Preserving',
            'target_domain': 'BUS-UCLM',
            'source_domain': 'BUSI',
            'privacy_preserving': True
        }
        
        print("🎯 CCST BUS-UCLM Augmentation Pipeline")
        print("=" * 50)
        print(f"📂 BUSI path: {self.busi_path}")
        print(f"📂 BUS-UCLM path: {self.bus_uclm_path}")
        print(f"📂 Output: {self.output_base}")
        print(f"🖥️  Device: {self.device}")
        
    def step1_extract_busi_statistics(self):
        """Step 1: Extract privacy-preserving BUSI style statistics"""
        print("\n" + "="*60)
        print("🔐 STEP 1: EXTRACTING BUSI STYLE STATISTICS (Privacy-Preserving)")
        print("="*60)
        
        # Initialize style extractor
        extractor = PrivacyPreservingStyleExtractor(device=self.device)
        
        # Extract BUSI domain statistics
        print("📊 Extracting BUSI domain-level style statistics...")
        busi_stats = extractor.extract_domain_statistics(
            dataset_path=self.busi_path,
            csv_file='train_frame.csv',
            save_path=f"{self.output_base}_busi_stats.json"
        )
        
        self.results['busi_statistics'] = {
            'num_images_processed': busi_stats.get('num_images', 0),
            'feature_dimensions': busi_stats.get('feature_shape', None),
            'privacy_preserving': True,
            'only_statistics_shared': True
        }
        
        print(f"✅ BUSI statistics extracted")
        print(f"   📈 Images processed: {busi_stats.get('num_images', 0)}")
        print(f"   🔒 Privacy preserved: Only mean/variance shared")
        
        return f"{self.output_base}_busi_stats.json"
    
    def step2_create_augmented_dataset(self, busi_stats_path):
        """Step 2: Create BUS-UCLM dataset augmented with BUSI-style images"""
        print("\n" + "="*60)
        print("🎨 STEP 2: CREATING AUGMENTED BUS-UCLM DATASET")
        print("="*60)
        
        # Initialize style transfer
        style_transfer = AdaINStyleTransfer(device=self.device)
        
        # Create dataset generator
        generator = StyleTransferDatasetGenerator(
            style_transfer_model=style_transfer,
            device=self.device
        )
        
        # Generate styled BUS-UCLM dataset
        augmented_path = f"{self.output_base}_augmented_busuclm"
        print(f"🔄 Applying BUSI style to BUS-UCLM training images...")
        
        styled_info = generator.generate_styled_dataset(
            source_path=self.bus_uclm_path,
            target_stats_path=busi_stats_path,
            output_path=augmented_path,
            csv_file='train_frame.csv'
        )
        
        # Create combined dataset (original + styled)
        combined_path = f"{self.output_base}_combined_dataset"
        self._create_combined_dataset(
            original_path=self.bus_uclm_path,
            styled_path=augmented_path, 
            output_path=combined_path
        )
        
        self.results['augmentation'] = {
            'original_busuclm_images': styled_info.get('original_count', 0),
            'styled_images_generated': styled_info.get('styled_count', 0),
            'combined_training_size': styled_info.get('original_count', 0) + styled_info.get('styled_count', 0),
            'style_source': 'BUSI_privacy_preserving_statistics'
        }
        
        print(f"✅ Augmented dataset created")
        print(f"   📊 Original BUS-UCLM: {styled_info.get('original_count', 0)} images")
        print(f"   🎨 BUSI-styled: {styled_info.get('styled_count', 0)} images")
        print(f"   📈 Total training: {self.results['augmentation']['combined_training_size']} images")
        
        return combined_path
    
    def step3_train_baseline(self):
        """Step 3: Train BUS-UCLM-only baseline for comparison"""
        print("\n" + "="*60)
        print("📊 STEP 3: TRAINING BUS-UCLM-ONLY BASELINE")
        print("="*60)
        
        baseline_args = [
            '--train_data_type', 'BUS-UCLM',
            '--test_data_type', 'BUS-UCLM', 
            '--final_epoch', '100',
            '--train',
            '--device', 'cuda',
            '--save_path', f'{self.output_base}_baseline_model.pth',
            '--image_size', '352',
            '--batch_size', '4'
        ]
        
        print("🚀 Training baseline model (BUS-UCLM only)...")
        try:
            # Store original sys.argv
            original_argv = sys.argv
            sys.argv = ['IS2D_main.py'] + baseline_args
            
            baseline_metrics = train_is2d()
            
            self.results['baseline'] = {
                'training_data': 'BUS-UCLM_only',
                'test_data': 'BUS-UCLM', 
                'epochs': 100,
                'metrics': baseline_metrics
            }
            
            print(f"✅ Baseline training completed")
            if baseline_metrics:
                print(f"   📈 Best Dice: {baseline_metrics.get('best_dice', 'N/A'):.4f}")
                print(f"   📈 Best IoU: {baseline_metrics.get('best_iou', 'N/A'):.4f}")
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
        return f'{self.output_base}_baseline_model.pth'
    
    def step4_train_augmented(self, combined_dataset_path):
        """Step 4: Train on augmented dataset (BUS-UCLM + BUSI-styled)"""
        print("\n" + "="*60)
        print("🚀 STEP 4: TRAINING ON AUGMENTED DATASET")
        print("="*60)
        
        # Create custom dataset class for combined data
        self._register_combined_dataset(combined_dataset_path)
        
        augmented_args = [
            '--train_data_type', 'BUS-UCLM-CCST-Combined',
            '--test_data_type', 'BUS-UCLM',
            '--final_epoch', '100', 
            '--train',
            '--device', 'cuda',
            '--save_path', f'{self.output_base}_augmented_model.pth',
            '--image_size', '352',
            '--batch_size', '4'
        ]
        
        print("🚀 Training augmented model (BUS-UCLM + BUSI-styled)...")
        try:
            # Store original sys.argv
            original_argv = sys.argv
            sys.argv = ['IS2D_main.py'] + augmented_args
            
            augmented_metrics = train_is2d()
            
            self.results['augmented'] = {
                'training_data': 'BUS-UCLM + BUSI-styled',
                'test_data': 'BUS-UCLM',
                'epochs': 100,
                'metrics': augmented_metrics
            }
            
            print(f"✅ Augmented training completed")
            if augmented_metrics:
                print(f"   📈 Best Dice: {augmented_metrics.get('best_dice', 'N/A'):.4f}")
                print(f"   📈 Best IoU: {augmented_metrics.get('best_iou', 'N/A'):.4f}")
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
        return f'{self.output_base}_augmented_model.pth'
    
    def step5_compare_results(self):
        """Step 5: Compare baseline vs augmented performance"""
        print("\n" + "="*60)
        print("📊 STEP 5: PERFORMANCE COMPARISON")
        print("="*60)
        
        baseline_metrics = self.results.get('baseline', {}).get('metrics', {})
        augmented_metrics = self.results.get('augmented', {}).get('metrics', {})
        
        # Calculate improvement
        if baseline_metrics and augmented_metrics:
            baseline_dice = baseline_metrics.get('best_dice', 0)
            augmented_dice = augmented_metrics.get('best_dice', 0)
            baseline_iou = baseline_metrics.get('best_iou', 0)
            augmented_iou = augmented_metrics.get('best_iou', 0)
            
            dice_improvement = augmented_dice - baseline_dice
            iou_improvement = augmented_iou - baseline_iou
            dice_improvement_pct = (dice_improvement / baseline_dice * 100) if baseline_dice > 0 else 0
            iou_improvement_pct = (iou_improvement / baseline_iou * 100) if baseline_iou > 0 else 0
            
            self.results['comparison'] = {
                'baseline_dice': baseline_dice,
                'augmented_dice': augmented_dice,
                'dice_improvement': dice_improvement,
                'dice_improvement_percentage': dice_improvement_pct,
                'baseline_iou': baseline_iou,
                'augmented_iou': augmented_iou,
                'iou_improvement': iou_improvement,
                'iou_improvement_percentage': iou_improvement_pct,
                'approach_successful': dice_improvement > 0 and iou_improvement > 0
            }
            
            print("📊 PERFORMANCE COMPARISON RESULTS")
            print("-" * 40)
            print(f"🔹 Baseline (BUS-UCLM only):")
            print(f"   Dice: {baseline_dice:.4f}")
            print(f"   IoU:  {baseline_iou:.4f}")
            print(f"🔹 Augmented (BUS-UCLM + BUSI-styled):")
            print(f"   Dice: {augmented_dice:.4f}")
            print(f"   IoU:  {augmented_iou:.4f}")
            print(f"🔹 Improvement:")
            print(f"   Dice: {dice_improvement:+.4f} ({dice_improvement_pct:+.2f}%)")
            print(f"   IoU:  {iou_improvement:+.4f} ({iou_improvement_pct:+.2f}%)")
            
            if dice_improvement > 0 and iou_improvement > 0:
                print("\n🎉 SUCCESS: BUSI knowledge improved BUS-UCLM performance!")
            else:
                print("\n⚠️  No improvement detected. May need parameter tuning.")
        
        # Save results
        results_file = f"{self.output_base}_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed results saved: {results_file}")
    
    def _create_combined_dataset(self, original_path, styled_path, output_path):
        """Create combined dataset with original + styled images"""
        import shutil
        import pandas as pd
        
        os.makedirs(output_path, exist_ok=True)
        
        # Copy original BUS-UCLM data
        for split in ['train_frame.csv', 'val_frame.csv', 'test_frame.csv']:
            if os.path.exists(os.path.join(original_path, split)):
                shutil.copy2(
                    os.path.join(original_path, split),
                    os.path.join(output_path, split)
                )
        
        # Copy original images and masks
        for class_type in ['benign', 'malignant']:
            for data_type in ['image', 'mask']:
                src_dir = os.path.join(original_path, class_type, data_type)
                dst_dir = os.path.join(output_path, class_type, data_type)
                if os.path.exists(src_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for filename in os.listdir(src_dir):
                        shutil.copy2(
                            os.path.join(src_dir, filename),
                            os.path.join(dst_dir, filename)
                        )
        
        # Add styled images to training set
        if os.path.exists(styled_path):
            train_df = pd.read_csv(os.path.join(output_path, 'train_frame.csv'))
            
            for class_type in ['benign', 'malignant']:
                styled_img_dir = os.path.join(styled_path, class_type, 'image')
                styled_mask_dir = os.path.join(styled_path, class_type, 'mask')
                
                if os.path.exists(styled_img_dir):
                    # Copy styled images
                    dst_img_dir = os.path.join(output_path, class_type, 'image')
                    dst_mask_dir = os.path.join(output_path, class_type, 'mask')
                    
                    for filename in os.listdir(styled_img_dir):
                        if filename.startswith('styled_'):
                            # Copy image
                            shutil.copy2(
                                os.path.join(styled_img_dir, filename),
                                os.path.join(dst_img_dir, filename)
                            )
                            
                            # Copy corresponding mask
                            mask_filename = filename.replace('.png', '_mask.png')
                            if os.path.exists(os.path.join(styled_mask_dir, mask_filename)):
                                shutil.copy2(
                                    os.path.join(styled_mask_dir, mask_filename),
                                    os.path.join(dst_mask_dir, mask_filename)
                                )
                            
                            # Add to training CSV
                            new_row = pd.DataFrame({
                                'image_path': [f"{class_type} {filename}"],
                                'mask_path': [f"{class_type} {mask_filename}"],
                                'class': [class_type]
                            })
                            train_df = pd.concat([train_df, new_row], ignore_index=True)
            
            # Save updated training CSV
            train_df.to_csv(os.path.join(output_path, 'train_frame.csv'), index=False)
    
    def _register_combined_dataset(self, dataset_path):
        """Register combined dataset for IS2D training"""
        # This would need to be implemented to register the custom dataset
        # For now, we'll modify the existing dataset loading logic
        pass
    
    def run_complete_pipeline(self):
        """Run the complete CCST augmentation pipeline"""
        print("\n🚀 Starting CCST BUS-UCLM Augmentation Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Extract BUSI statistics
            busi_stats_path = self.step1_extract_busi_statistics()
            
            # Step 2: Create augmented dataset  
            combined_dataset_path = self.step2_create_augmented_dataset(busi_stats_path)
            
            # Step 3: Train baseline
            baseline_model_path = self.step3_train_baseline()
            
            # Step 4: Train augmented model
            augmented_model_path = self.step4_train_augmented(combined_dataset_path)
            
            # Step 5: Compare results
            self.step5_compare_results()
            
            print("\n🎉 CCST Pipeline completed successfully!")
            print("=" * 60)
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description='CCST BUS-UCLM Augmentation Pipeline')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline training')
    parser.add_argument('--skip-augmented', action='store_true', help='Skip augmented training')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CCSTargetedPipeline(device=args.device)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    if results and results.get('comparison', {}).get('approach_successful', False):
        print("\n🎯 MISSION ACCOMPLISHED!")
        print("BUSI knowledge successfully improved BUS-UCLM performance while preserving privacy!")
    else:
        print("\n📈 Pipeline completed. Check results for detailed analysis.")

if __name__ == '__main__':
    main() 