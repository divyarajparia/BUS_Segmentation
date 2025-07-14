#!/usr/bin/env python3
"""
Advanced Privacy-Preserving Style Transfer Pipeline
High-performance methods that should match CycleGAN results while preserving privacy.
"""

import os
import time
from advanced_privacy_style_transfer import (
    AdvancedPrivacyPreservingStyleTransfer,
    generate_advanced_styled_dataset
)

def run_advanced_pipeline():
    """Run the advanced privacy-preserving style transfer pipeline."""
    print("🚀 Advanced Privacy-Preserving Style Transfer Pipeline")
    print("=" * 70)
    print("🎯 Goal: Match CycleGAN performance (+9.5%) while preserving privacy")
    print()
    
    # Step 1: Extract advanced BUSI statistics
    print("🧠 Step 1: Extracting Advanced BUSI Statistics")
    
    advanced_stats_path = 'privacy_style_stats/busi_advanced_stats.json'
    if not os.path.exists(advanced_stats_path):
        print("   Extracting advanced BUSI statistics...")
        extractor = AdvancedPrivacyPreservingStyleTransfer()
        extractor.extract_advanced_domain_statistics(
            dataset_path='dataset/BioMedicalDataset/BUSI',
            csv_file='train_frame.csv',
            save_path=advanced_stats_path
        )
    else:
        print(f"   ✅ Using existing advanced BUSI statistics: {advanced_stats_path}")
    
    # Step 2: Test multiple advanced methods
    methods = [
        'adaptive_multi_scale',
        'structure_preserving', 
        'ensemble'  # Combination of all methods
    ]
    
    for method in methods:
        print(f"\n🎨 Step 2: Generating Dataset with {method.title().replace('_', ' ')} Method")
        
        output_dir = f'dataset/BioMedicalDataset/BUS-UCLM-Advanced-{method.title().replace("_", "")}'
        
        print(f"   Method: {method}")
        print(f"   Output: {output_dir}")
        
        start_time = time.time()
        
        styled_samples = generate_advanced_styled_dataset(
            source_dataset_path='dataset/BioMedicalDataset/BUS-UCLM',
            source_csv='train_frame.csv',
            target_stats_path=advanced_stats_path,
            output_dir=output_dir,
            method=method
        )
        
        generation_time = time.time() - start_time
        
        print(f"   ⏱️  Generation time: {generation_time:.1f} seconds")
        print(f"   📊 Generated: {len(styled_samples)} styled images")
        
        # Create combined CSV for training
        create_advanced_combined_csv(output_dir, method)
    
    print(f"\n🎉 Advanced Privacy-Preserving Pipeline Complete!")
    print("📋 Next Steps:")
    print("   1. Test each method with training")
    print("   2. Compare results to CycleGAN baseline")
    print("   3. Select best performing method")
    

def create_advanced_combined_csv(styled_dir, method):
    """Create combined CSV for training with advanced styled data."""
    import pandas as pd
    
    print(f"📝 Creating combined CSV for {method} method...")
    
    combined_samples = []
    
    # 1. Add original BUSI data
    busi_dir = 'dataset/BioMedicalDataset/BUSI'
    busi_csv = os.path.join(busi_dir, 'train_frame.csv')
    
    if os.path.exists(busi_csv):
        busi_df = pd.read_csv(busi_csv)
        
        for idx in range(len(busi_df)):
            image_path = busi_df.iloc[idx]['image_path']
            mask_path = busi_df.iloc[idx]['mask_path']
            class_type = image_path.split()[0]
            
            combined_samples.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'class': class_type,
                'source_client': 'BUSI',
                'style_client': 'BUSI',
                'augmentation_type': 'original'
            })
    
    # 2. Add advanced styled data
    styled_csv = os.path.join(styled_dir, 'advanced_styled_dataset.csv')
    if os.path.exists(styled_csv):
        styled_df = pd.read_csv(styled_csv)
        combined_samples.extend(styled_df.to_dict('records'))
    
    # Save combined CSV
    combined_csv = os.path.join(styled_dir, 'ccst_augmented_dataset.csv')
    combined_df = pd.DataFrame(combined_samples)
    combined_df.to_csv(combined_csv, index=False)
    
    # Print statistics
    original_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'original')
    styled_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'advanced_styled')
    benign_count = sum(1 for s in combined_samples if s['class'] == 'benign')
    malignant_count = sum(1 for s in combined_samples if s['class'] == 'malignant')
    
    print(f"   ✅ Combined dataset created: {len(combined_samples)} total samples")
    print(f"   📊 Breakdown:")
    print(f"      Original BUSI: {original_count}")
    print(f"      Advanced Styled: {styled_count}")
    print(f"      Benign: {benign_count}")
    print(f"      Malignant: {malignant_count}")
    print(f"   📄 Saved to: {combined_csv}")


if __name__ == "__main__":
    run_advanced_pipeline() 