#!/usr/bin/env python3
"""
Medical Image-Specific Privacy-Preserving Style Transfer Pipeline
Optimized for grayscale ultrasound images - NO VGG dependency.
"""

import os
import time
from medical_privacy_style_transfer import (
    MedicalPrivacyStyleTransfer,
    generate_medical_styled_dataset
)

def run_medical_privacy_pipeline():
    """Run the medical image-specific privacy-preserving style transfer pipeline."""
    print("🏥 Medical Privacy-Preserving Style Transfer Pipeline")
    print("=" * 70)
    print("🎯 Optimized for grayscale ultrasound images")
    print("🔬 Medical image analysis techniques (no VGG)")
    print()
    
    # Step 1: Extract medical BUSI statistics
    print("🔬 Step 1: Extracting Medical Domain Statistics from BUSI")
    
    medical_stats_path = 'privacy_style_stats/busi_medical_stats.json'
    if not os.path.exists(medical_stats_path):
        print("   Extracting medical-specific BUSI statistics...")
        extractor = MedicalPrivacyStyleTransfer()
        extractor.extract_medical_domain_statistics(
            dataset_path='dataset/BioMedicalDataset/BUSI',
            csv_file='train_frame.csv',
            save_path=medical_stats_path
        )
    else:
        print(f"   ✅ Using existing medical BUSI statistics: {medical_stats_path}")
    
    # Step 2: Test medical-specific methods
    methods = [
        'medical_adaptive',     # Adaptive transfer preserving anatomical structures
        'ultrasound_optimized', # Ultrasound speckle-aware transfer
        'structure_aware'       # Morphology-based structure preservation
    ]
    
    for method in methods:
        print(f"\n🎨 Step 2: Generating Dataset with {method.title().replace('_', ' ')} Method")
        
        output_dir = f'dataset/BioMedicalDataset/BUS-UCLM-Medical-{method.title().replace("_", "")}'
        
        print(f"   Method: {method}")
        print(f"   Output: {output_dir}")
        print(f"   Features: Medical texture analysis, ultrasound speckle, anatomical patterns")
        
        start_time = time.time()
        
        styled_samples = generate_medical_styled_dataset(
            source_dataset_path='dataset/BioMedicalDataset/BUS-UCLM',
            source_csv='train_frame.csv',
            target_stats_path=medical_stats_path,
            output_dir=output_dir,
            method=method
        )
        
        generation_time = time.time() - start_time
        
        print(f"   ⏱️  Generation time: {generation_time:.1f} seconds")
        print(f"   📊 Generated: {len(styled_samples)} medical styled images")
        
        # Create combined CSV for training
        create_medical_combined_csv(output_dir, method)
    
    print(f"\n🎉 Medical Privacy-Preserving Pipeline Complete!")
    print("📋 Next Steps:")
    print("   1. Test each medical method with training")
    print("   2. Compare results to CycleGAN baseline")
    print("   3. Medical methods should preserve ultrasound characteristics better")
    

def create_medical_combined_csv(styled_dir, method):
    """Create combined CSV for training with medical styled data."""
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
    
    # 2. Add medical styled data
    styled_csv = os.path.join(styled_dir, 'medical_styled_dataset.csv')
    if os.path.exists(styled_csv):
        styled_df = pd.read_csv(styled_csv)
        combined_samples.extend(styled_df.to_dict('records'))
    
    # Save combined CSV
    combined_csv = os.path.join(styled_dir, 'ccst_augmented_dataset.csv')
    combined_df = pd.DataFrame(combined_samples)
    combined_df.to_csv(combined_csv, index=False)
    
    # Print statistics
    original_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'original')
    styled_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'medical_styled')
    benign_count = sum(1 for s in combined_samples if s['class'] == 'benign')
    malignant_count = sum(1 for s in combined_samples if s['class'] == 'malignant')
    
    print(f"   ✅ Combined dataset created: {len(combined_samples)} total samples")
    print(f"   📊 Breakdown:")
    print(f"      Original BUSI: {original_count}")
    print(f"      Medical Styled: {styled_count}")
    print(f"      Benign: {benign_count}")
    print(f"      Malignant: {malignant_count}")
    print(f"   📄 Saved to: {combined_csv}")


if __name__ == "__main__":
    run_medical_privacy_pipeline() 