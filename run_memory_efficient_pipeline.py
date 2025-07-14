#!/usr/bin/env python3
"""
Memory-Efficient Privacy-Preserving Style Transfer Pipeline
Optimized to prevent OOM errors on server systems.
"""

import os
import time
from memory_efficient_medical_style_transfer import (
    MemoryEfficientMedicalStyleTransfer,
    generate_memory_efficient_styled_dataset
)
import pandas as pd

def run_memory_efficient_pipeline():
    """Run the memory-efficient privacy-preserving style transfer pipeline."""
    print("🚀 Memory-Efficient Privacy-Preserving Style Transfer Pipeline")
    print("=" * 70)
    print("💾 Optimized to prevent OOM errors")
    print("🔬 Medical image analysis with minimal memory usage")
    print()
    
    # Step 1: Extract BUSI statistics with memory optimization
    print("🔬 Step 1: Extracting BUSI Statistics (Memory-Efficient)")
    
    stats_path = 'privacy_style_stats/busi_efficient_stats.json'
    if not os.path.exists(stats_path):
        print("   Extracting efficient BUSI statistics...")
        print("   Processing in batches of 50 images to prevent OOM")
        
        extractor = MemoryEfficientMedicalStyleTransfer(batch_size=50)
        extractor.extract_medical_domain_statistics(
            dataset_path='dataset/BioMedicalDataset/BUSI',
            csv_file='train_frame.csv',
            save_path=stats_path
        )
    else:
        print(f"   ✅ Using existing efficient BUSI statistics: {stats_path}")
    
    # Step 2: Generate styled dataset with memory optimization
    print("\n🎨 Step 2: Generating Memory-Efficient Styled Dataset")
    
    output_dir = 'dataset/BioMedicalDataset/BUS-UCLM-Efficient-Styled'
    
    print(f"   Output: {output_dir}")
    print("   Processing images one-by-one to minimize memory usage")
    print("   Garbage collection every 10 images")
    
    start_time = time.time()
    
    styled_samples = generate_memory_efficient_styled_dataset(
        source_dataset_path='dataset/BioMedicalDataset/BUS-UCLM',
        source_csv='train_frame.csv',
        target_stats_path=stats_path,
        output_dir=output_dir
    )
    
    generation_time = time.time() - start_time
    
    print(f"   ⏱️  Generation time: {generation_time:.1f} seconds")
    print(f"   📊 Generated: {len(styled_samples)} styled images")
    
    # Step 3: Create combined CSV for training
    print("\n📝 Step 3: Creating Combined Training CSV")
    create_efficient_combined_csv(output_dir)
    
    print(f"\n🎉 Memory-Efficient Pipeline Complete!")
    print("📋 Next Steps:")
    print("   Run training with the generated efficient dataset")
    print("   Expected to use significantly less memory")
    

def create_efficient_combined_csv(styled_dir):
    """Create combined CSV for training with efficient styled data."""
    print(f"📝 Creating combined CSV for efficient method...")
    
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
    
    # 2. Add efficient styled data
    styled_csv = os.path.join(styled_dir, 'efficient_styled_dataset.csv')
    if os.path.exists(styled_csv):
        styled_df = pd.read_csv(styled_csv)
        combined_samples.extend(styled_df.to_dict('records'))
    
    # Save combined CSV
    combined_csv = os.path.join(styled_dir, 'ccst_augmented_dataset.csv')
    combined_df = pd.DataFrame(combined_samples)
    combined_df.to_csv(combined_csv, index=False)
    
    # Print statistics
    original_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'original')
    styled_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'efficient_styled')
    benign_count = sum(1 for s in combined_samples if s['class'] == 'benign')
    malignant_count = sum(1 for s in combined_samples if s['class'] == 'malignant')
    
    print(f"   ✅ Combined dataset created: {len(combined_samples)} total samples")
    print(f"   📊 Breakdown:")
    print(f"      Original BUSI: {original_count}")
    print(f"      Efficient Styled: {styled_count}")
    print(f"      Benign: {benign_count}")
    print(f"      Malignant: {malignant_count}")
    print(f"   📄 Saved to: {combined_csv}")


if __name__ == "__main__":
    run_memory_efficient_pipeline() 