import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset(dataset_name, dataset_path):
    """Analyze a single dataset"""
    print(f"\n{'='*50}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'='*50}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return None
    
    analysis = {
        'name': dataset_name,
        'path': dataset_path,
        'total_samples': 0,
        'train_samples': 0,
        'test_samples': 0,
        'val_samples': 0,
        'benign_samples': 0,
        'malignant_samples': 0,
        'image_sizes': [],
        'file_formats': [],
        'class_distribution': {}
    }
    
    # Analyze CSV files
    csv_files = ['train_frame.csv', 'test_frame.csv', 'val_frame.csv']
    
    for csv_file in csv_files:
        csv_path = os.path.join(dataset_path, csv_file)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            split_name = csv_file.replace('_frame.csv', '')
            analysis[f'{split_name}_samples'] = len(df)
            
            print(f"\n{split_name.upper()} Split:")
            print(f"  Total samples: {len(df)}")
            
            # Analyze class distribution
            if 'image_path' in df.columns:
                benign_count = len(df[df['image_path'].str.contains('benign', case=False)])
                malignant_count = len(df[df['image_path'].str.contains('malignant', case=False)])
                
                print(f"  Benign: {benign_count} ({benign_count/len(df)*100:.1f}%)")
                print(f"  Malignant: {malignant_count} ({malignant_count/len(df)*100:.1f}%)")
                
                if split_name == 'train':
                    analysis['benign_samples'] = benign_count
                    analysis['malignant_samples'] = malignant_count
    
    # Calculate total samples
    analysis['total_samples'] = analysis['train_samples'] + analysis['test_samples'] + analysis['val_samples']
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Train/Test/Val: {analysis['train_samples']}/{analysis['test_samples']}/{analysis['val_samples']}")
    
    if analysis['total_samples'] > 0:
        benign_pct = analysis['benign_samples'] / analysis['train_samples'] * 100 if analysis['train_samples'] > 0 else 0
        malignant_pct = analysis['malignant_samples'] / analysis['train_samples'] * 100 if analysis['train_samples'] > 0 else 0
        print(f"  Class distribution (train): Benign {benign_pct:.1f}%, Malignant {malignant_pct:.1f}%")
    
    # Check for actual image directories
    subdirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    
    if subdirs:
        print(f"\nDIRECTORY STRUCTURE:")
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            print(f"  {subdir}/")
            
            # Check for images and masks subdirectories
            for sub_item in ['images', 'masks', 'image', 'mask']:
                sub_path = os.path.join(subdir_path, sub_item)
                if os.path.exists(sub_path) and os.path.isdir(sub_path):
                    files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"    {sub_item}/: {len(files)} files")
                    
                    # Sample a few images to check sizes
                    if len(files) > 0 and sub_item in ['images', 'image']:
                        sample_files = files[:min(5, len(files))]
                        for sample_file in sample_files:
                            try:
                                img_path = os.path.join(sub_path, sample_file)
                                with Image.open(img_path) as img:
                                    analysis['image_sizes'].append(img.size)
                                    analysis['file_formats'].append(img.format)
                            except Exception as e:
                                print(f"      Error reading {sample_file}: {e}")
    
    # Analyze image characteristics
    if analysis['image_sizes']:
        sizes = analysis['image_sizes']
        print(f"\nIMAGE CHARACTERISTICS:")
        print(f"  Sample size analysis (from {len(sizes)} images):")
        print(f"    Sizes found: {list(set(sizes))}")
        
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"    Width range: {min(widths)} - {max(widths)}")
        print(f"    Height range: {min(heights)} - {max(heights)}")
        
        format_counts = Counter(analysis['file_formats'])
        print(f"    File formats: {dict(format_counts)}")
    
    return analysis

def compare_datasets(busi_analysis, bus_uclm_analysis):
    """Compare the two datasets"""
    print(f"\n{'='*60}")
    print(f"DATASET COMPARISON")
    print(f"{'='*60}")
    
    if not busi_analysis or not bus_uclm_analysis:
        print("Cannot compare - one or both datasets not found!")
        return
    
    # Sample size comparison
    print(f"\nSAMPLE SIZE COMPARISON:")
    print(f"  BUSI total: {busi_analysis['total_samples']}")
    print(f"  BUS-UCLM total: {bus_uclm_analysis['total_samples']}")
    print(f"  Ratio (BUS-UCLM/BUSI): {bus_uclm_analysis['total_samples']/busi_analysis['total_samples']:.2f}x")
    
    # Class distribution comparison
    print(f"\nCLASS DISTRIBUTION COMPARISON:")
    
    if busi_analysis['train_samples'] > 0:
        busi_benign_pct = busi_analysis['benign_samples'] / busi_analysis['train_samples'] * 100
        busi_malignant_pct = busi_analysis['malignant_samples'] / busi_analysis['train_samples'] * 100
        print(f"  BUSI: Benign {busi_benign_pct:.1f}%, Malignant {busi_malignant_pct:.1f}%")
    
    if bus_uclm_analysis['train_samples'] > 0:
        bus_uclm_benign_pct = bus_uclm_analysis['benign_samples'] / bus_uclm_analysis['train_samples'] * 100
        bus_uclm_malignant_pct = bus_uclm_analysis['malignant_samples'] / bus_uclm_analysis['train_samples'] * 100
        print(f"  BUS-UCLM: Benign {bus_uclm_benign_pct:.1f}%, Malignant {bus_uclm_malignant_pct:.1f}%")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS FOR SYNTHETIC GENERATION:")
    
    if bus_uclm_analysis['total_samples'] > busi_analysis['total_samples']:
        print(f"  ✓ BUS-UCLM has more samples ({bus_uclm_analysis['total_samples']} vs {busi_analysis['total_samples']})")
        print(f"    → Good candidate for synthetic generation (more training data)")
    else:
        print(f"  ✗ BUSI has more samples ({busi_analysis['total_samples']} vs {bus_uclm_analysis['total_samples']})")
        print(f"    → Direct BUSI synthesis might be better")
    
    # Check class balance
    if busi_analysis['train_samples'] > 0 and bus_uclm_analysis['train_samples'] > 0:
        busi_balance = min(busi_analysis['benign_samples'], busi_analysis['malignant_samples']) / max(busi_analysis['benign_samples'], busi_analysis['malignant_samples'])
        bus_uclm_balance = min(bus_uclm_analysis['benign_samples'], bus_uclm_analysis['malignant_samples']) / max(bus_uclm_analysis['benign_samples'], bus_uclm_analysis['malignant_samples'])
        
        print(f"\nCLASS BALANCE ANALYSIS:")
        print(f"  BUSI balance ratio: {busi_balance:.2f} (1.0 = perfect balance)")
        print(f"  BUS-UCLM balance ratio: {bus_uclm_balance:.2f}")
        
        if bus_uclm_balance > busi_balance:
            print(f"  ✓ BUS-UCLM has better class balance")
            print(f"    → Generate synthetic samples to improve BUSI balance")
        else:
            print(f"  ✓ BUSI has better class balance")

def main():
    # Paths to datasets
    busi_path = "dataset/BioMedicalDataset/BUSI"
    bus_uclm_path = "dataset/BioMedicalDataset/BUS-UCLM"
    
    # Analyze both datasets
    busi_analysis = analyze_dataset("BUSI", busi_path)
    bus_uclm_analysis = analyze_dataset("BUS-UCLM", bus_uclm_path)
    
    # Compare datasets
    compare_datasets(busi_analysis, bus_uclm_analysis)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()