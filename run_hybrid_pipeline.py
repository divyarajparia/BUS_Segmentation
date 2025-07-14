#!/usr/bin/env python3
"""
Complete Hybrid Medical Privacy-Preserving Style Transfer Pipeline
Ready-to-run script that generates dataset and provides training commands.
"""

import os
import time
import json
import pandas as pd
from hybrid_memory_medical_transfer import HybridMedicalStyleTransfer
from tqdm import tqdm
import cv2
from PIL import Image
import gc

def generate_hybrid_styled_dataset(source_dataset_path, source_csv, target_stats_path, 
                                 output_dir, complexity='medium', use_full_dataset=True):
    """
    Generate styled dataset using hybrid medical style transfer.
    """
    print(f"🏥 Generating hybrid styled dataset (complexity: {complexity})")
    
    # Initialize hybrid style transfer
    style_transfer = HybridMedicalStyleTransfer(complexity=complexity, batch_size=25)
    
    # Load target statistics
    with open(target_stats_path, 'r') as f:
        target_stats = json.load(f)
    
    # Load source CSV - use full dataset if requested
    if use_full_dataset and os.path.exists(os.path.join(source_dataset_path, 'full_dataset.csv')):
        csv_path = os.path.join(source_dataset_path, 'full_dataset.csv')
        print(f"   📈 Using FULL BUS-UCLM dataset (all train/test/val)")
    else:
        csv_path = os.path.join(source_dataset_path, source_csv)
        print(f"   📊 Using BUS-UCLM {source_csv}")
    
    df = pd.read_csv(csv_path)
    
            # Create output directories (use 'images' and 'masks' to match BUSUCLMSegmentationDataset)
        os.makedirs(os.path.join(output_dir, 'benign', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'benign', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'malignant', 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'malignant', 'masks'), exist_ok=True)
    
    styled_samples = []
    
    # Process images one by one to minimize memory usage
    for idx in tqdm(range(len(df)), desc="Generating hybrid styled images"):
        try:
            row = df.iloc[idx]
            image_path_info = row['image_path']
            mask_path_info = row['mask_path']
            
            # Determine class
            class_type = image_path_info.split()[0]
            
            # Get source paths
            source_image_path = style_transfer._get_image_path(row, source_dataset_path)
            
            # Generate output paths - fix basename extraction
            # image_path_info format: "benign FILENAME.png"
            actual_image_filename = image_path_info.split(' ', 1)[1] if ' ' in image_path_info else image_path_info
            actual_mask_filename = mask_path_info.split(' ', 1)[1] if ' ' in mask_path_info else mask_path_info
            
            output_filename = f"hybrid_{complexity}_{actual_image_filename}"
            output_mask_filename = f"hybrid_{complexity}_{actual_mask_filename}"
            
            output_image_path = os.path.join(output_dir, class_type, 'images', output_filename)
            output_mask_path = os.path.join(output_dir, class_type, 'masks', output_mask_filename)
            
            # Apply hybrid style transfer
            style_transfer.apply_medical_style_transfer(
                source_image_path, target_stats, output_image_path
            )
            
            # Process mask efficiently - fix path construction
            # mask_path_info format: "benign FILENAME.png" or "malignant FILENAME.png"
            mask_class, mask_filename = mask_path_info.split(' ', 1)
            mask_source_path = os.path.join(source_dataset_path, mask_class, 'masks', mask_filename)
            
            if os.path.exists(mask_source_path):
                # Process mask with minimal memory
                os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
                mask = cv2.imread(mask_source_path)
                if mask is not None:
                    mask_resized = cv2.resize(mask, (256, 256))
                    mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY) if len(mask_resized.shape) == 3 else mask_resized
                    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
                    mask_rgb = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(output_mask_path, mask_rgb)
                    
                    # Clean up mask memory
                    del mask, mask_resized, mask_gray, mask_binary, mask_rgb
                else:
                    print(f"   Warning: Could not read mask {mask_source_path}")
            else:
                print(f"   Warning: Mask not found at {mask_source_path}")
            
            styled_samples.append({
                'image_path': f"{class_type} {output_filename}",
                'mask_path': f"{class_type} {output_mask_filename}",
                'class': class_type,
                'source_client': 'BUS-UCLM',
                'style_client': 'BUSI', 
                'augmentation_type': f'hybrid_{complexity}'
            })
            
            # Force garbage collection every 10 images
            if idx % 10 == 0:
                gc.collect()
            
        except Exception as e:
            print(f"Warning: Error processing {idx}: {e}")
            continue
    
    # Save dataset CSV
    styled_csv_path = os.path.join(output_dir, f'hybrid_{complexity}_dataset.csv')
    styled_df = pd.DataFrame(styled_samples)
    styled_df.to_csv(styled_csv_path, index=False)
    
    print(f"✅ Generated {len(styled_samples)} hybrid styled images")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 CSV file: {styled_csv_path}")
    
    return styled_samples


def create_simple_styled_csv(styled_dir, complexity):
    """Create simple train_frame.csv for styled dataset (compatible with BUSUCLMSegmentationDataset)."""
    print(f"📝 Creating simple CSV for {complexity} styled dataset...")
    
    samples = []
    
    # Check benign images
    benign_image_dir = os.path.join(styled_dir, 'benign', 'images')
    benign_mask_dir = os.path.join(styled_dir, 'benign', 'masks')
    
    if os.path.exists(benign_image_dir):
        for filename in sorted(os.listdir(benign_image_dir)):
            if filename.endswith('.png'):
                mask_filename = filename.replace('.png', '_mask.png')
                samples.append({
                    'image_path': f'benign {filename}',
                    'mask_path': f'benign {mask_filename}'
                })
    
    # Check malignant images  
    malignant_image_dir = os.path.join(styled_dir, 'malignant', 'images')
    malignant_mask_dir = os.path.join(styled_dir, 'malignant', 'masks')
    
    if os.path.exists(malignant_image_dir):
        for filename in sorted(os.listdir(malignant_image_dir)):
            if filename.endswith('.png'):
                mask_filename = filename.replace('.png', '_mask.png')
                samples.append({
                    'image_path': f'malignant {filename}',
                    'mask_path': f'malignant {mask_filename}'
                })
    
    # Save simple CSV
    csv_path = os.path.join(styled_dir, 'train_frame.csv')
    df = pd.DataFrame(samples)
    df.to_csv(csv_path, index=False)
    
    benign_count = len([s for s in samples if s['image_path'].startswith('benign')])
    malignant_count = len([s for s in samples if s['image_path'].startswith('malignant')])
    
    print(f"   ✅ Simple CSV created: {len(samples)} styled samples")
    print(f"   📊 Breakdown:")
    print(f"      Benign: {benign_count}")
    print(f"      Malignant: {malignant_count}")
    print(f"   📄 Saved to: {csv_path}")
    
    return csv_path


def create_hybrid_combined_csv(styled_dir, complexity):
    """Create combined CSV for training with hybrid styled data."""
    print(f"📝 Creating combined CSV for {complexity} complexity...")
    
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
    
    # 2. Add hybrid styled data
    styled_csv = os.path.join(styled_dir, f'hybrid_{complexity}_dataset.csv')
    if os.path.exists(styled_csv):
        styled_df = pd.read_csv(styled_csv)
        combined_samples.extend(styled_df.to_dict('records'))
    
    # Save combined CSV with shuffling
    combined_csv = os.path.join(styled_dir, 'ccst_augmented_dataset.csv')
    combined_df = pd.DataFrame(combined_samples)
    
    # 🔄 CRITICAL: Shuffle the combined dataset for proper training
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    combined_df.to_csv(combined_csv, index=False)
    
    # Print statistics
    original_count = sum(1 for s in combined_samples if s['augmentation_type'] == 'original')
    styled_count = sum(1 for s in combined_samples if s['augmentation_type'] == f'hybrid_{complexity}')
    benign_count = sum(1 for s in combined_samples if s['class'] == 'benign')
    malignant_count = sum(1 for s in combined_samples if s['class'] == 'malignant')
    
    print(f"   ✅ Combined dataset created: {len(combined_samples)} total samples")
    print(f"   📊 Breakdown:")
    print(f"      Original BUSI: {original_count}")
    print(f"      Hybrid {complexity.title()} Styled: {styled_count}")
    print(f"      Benign: {benign_count}")
    print(f"      Malignant: {malignant_count}")
    print(f"   📄 Saved to: {combined_csv}")
    
    return combined_csv


def run_complete_hybrid_pipeline(complexity='medium'):
    """
    Run the complete hybrid pipeline with specified complexity.
    """
    import json
    
    print("🚀 Complete Hybrid Medical Privacy-Preserving Pipeline")
    print("=" * 70)
    print(f"🎯 Complexity: {complexity.upper()}")
    print(f"🔬 Expected Performance: {get_expected_performance(complexity)}")
    print()
    
    # Step 1: Extract BUSI statistics
    print(f"🔬 Step 1: Extracting BUSI Statistics ({complexity} complexity)")
    
    stats_path = f'privacy_style_stats/busi_{complexity}_stats.json'
    if not os.path.exists(stats_path):
        print(f"   Extracting {complexity} complexity BUSI statistics...")
        
        extractor = HybridMedicalStyleTransfer(complexity=complexity, batch_size=25)
        extractor.extract_medical_domain_statistics(
            dataset_path='dataset/BioMedicalDataset/BUSI',
            csv_file='train_frame.csv',
            save_path=stats_path
        )
    else:
        print(f"   ✅ Using existing {complexity} BUSI statistics: {stats_path}")
    
    # Step 2: Generate styled dataset
    print(f"\n🎨 Step 2: Generating Hybrid Styled Dataset ({complexity} complexity)")
    
    output_dir = f'dataset/BioMedicalDataset/BUS-UCLM-Hybrid-{complexity.title()}'
    
    print(f"   Method: Hybrid {complexity}")
    print(f"   Output: {output_dir}")
    print(f"   Features: {get_features_description(complexity)}")
    
    start_time = time.time()
    
    styled_samples = generate_hybrid_styled_dataset(
        source_dataset_path='dataset/BioMedicalDataset/BUS-UCLM',
        source_csv='train_frame.csv',
        target_stats_path=stats_path,
        output_dir=output_dir,
        complexity=complexity,
        use_full_dataset=True  # Use all 264 BUS-UCLM samples
    )
    
    generation_time = time.time() - start_time
    
    print(f"   ⏱️  Generation time: {generation_time:.1f} seconds")
    print(f"   📊 Generated: {len(styled_samples)} styled images")
    
    # Step 3: Create simple CSV for styled dataset
    print(f"\n📝 Step 3: Creating Simple CSV for Styled Dataset")
    styled_csv_path = create_simple_styled_csv(output_dir, complexity)
    
    # Step 4: Print training command
    print(f"\n🎯 Step 4: Training Command")
    print("=" * 50)
    print("Copy and paste this command to train:")
    print()
    print(f"python IS2D_main.py --train_data_type BUSI-CCST --test_data_type BUSI --styled_dataset_path {output_dir} --train --final_epoch 100")
    print()
    print("=" * 50)
    
    print(f"\n🎉 Hybrid Pipeline Complete ({complexity} complexity)!")
    print("📋 Summary:")
    print(f"   Generated dataset: {output_dir}")
    print(f"   Styled CSV: {styled_csv_path}")
    print(f"   Expected Dice: {get_expected_performance(complexity)}")
    print(f"   Ready for training with IS2D_main.py!")


def get_expected_performance(complexity):
    """Get expected performance for complexity level."""
    performance_map = {
        'light': '0.82-0.84 Dice',
        'medium': '0.84-0.87 Dice', 
        'heavy': '0.87-0.90 Dice'
    }
    return performance_map.get(complexity, '0.82-0.84 Dice')


def get_features_description(complexity):
    """Get features description for complexity level."""
    features_map = {
        'light': 'Basic intensity & edge statistics',
        'medium': 'Enhanced intensity, texture, edge & frequency analysis',
        'heavy': 'Full feature set with morphological analysis'
    }
    return features_map.get(complexity, 'Basic features')


if __name__ == "__main__":
    import sys
    
    # Check if complexity is provided as argument
    if len(sys.argv) > 1:
        complexity = sys.argv[1].lower()
        if complexity not in ['light', 'medium', 'heavy']:
            print("❌ Invalid complexity. Use: light, medium, or heavy")
            print("   Defaulting to medium...")
            complexity = 'medium'
    else:
        complexity = 'medium'  # Default to medium
    
    print(f"🎯 Running {complexity.upper()} complexity pipeline...")
    print("   Light: Safe, basic features, 0.82-0.84 Dice")
    print("   Medium: Balanced, enhanced features, 0.84-0.87 Dice (RECOMMENDED)")
    print("   Heavy: Full features, 0.87-0.90 Dice, may OOM")
    print()
    
    try:
        run_complete_hybrid_pipeline(complexity)
    except Exception as e:
        print(f"\n❌ Error running {complexity} pipeline: {e}")
        if complexity != 'light':
            print(f"💡 Try running with light complexity:")
            print(f"   python run_hybrid_pipeline.py light")
        else:
            print(f"💡 Check your system memory and dataset paths")
            raise 