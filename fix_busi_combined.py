    """
    Fix BUSI-Combined Dataset - Proper Train/Test Separation
    =====================================================

    This script fixes the BUSI-Combined dataset to ensure proper experimental design:
    - Training: Original BUSI train + Style-transferred BUS-UCLM 
    - Testing: ONLY original BUSI test (no style-transferred samples)
    - Validation: ONLY original BUSI validation (no style-transferred samples)

    Usage:
        python fix_busi_combined.py
    """

    import os
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    def fix_busi_combined_dataset(combined_dir="dataset/BioMedicalDataset/BUSI-Combined"):
        """Fix the BUSI-Combined dataset with proper train/test separation"""
        
        print(f"🔧 Fixing BUSI-Combined dataset at: {combined_dir}")
        
        if not os.path.exists(combined_dir):
            print(f"❌ Directory {combined_dir} does not exist!")
            return
        
        # Separate original BUSI and style-transferred samples
        busi_pairs = []
        style_pairs = []
        
        for class_name in ['benign', 'malignant']:
            img_dir = os.path.join(combined_dir, class_name, 'images')
            mask_dir = os.path.join(combined_dir, class_name, 'masks')
            
            if not os.path.exists(img_dir):
                print(f"⚠️ Directory {img_dir} does not exist, skipping...")
                continue
                
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if img_file.startswith('busi_'):
                        # Original BUSI samples
                        mask_file = img_file.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
                        mask_path = os.path.join(mask_dir, mask_file)
                        if os.path.exists(mask_path):
                            busi_pairs.append({
                                'image_path': img_file,
                                'mask_path': mask_file,
                                'class': class_name
                            })
                    elif img_file.startswith('style_'):
                        # Style-transferred samples (only for training)
                        mask_file = img_file
                        mask_path = os.path.join(mask_dir, mask_file)
                        if os.path.exists(mask_path):
                            style_pairs.append({
                                'image_path': img_file,
                                'mask_path': mask_file,
                                'class': class_name
                            })
        
        print(f"📊 Found {len(busi_pairs)} original BUSI samples")
        print(f"📊 Found {len(style_pairs)} style-transferred samples")
        
        # Count by class
        busi_by_class = defaultdict(int)
        style_by_class = defaultdict(int)
        
        for pair in busi_pairs:
            busi_by_class[pair['class']] += 1
        for pair in style_pairs:
            style_by_class[pair['class']] += 1
        
        print(f"\n📈 Original BUSI breakdown:")
        for class_name, count in busi_by_class.items():
            print(f"  - {class_name}: {count} samples")
        
        print(f"\n📈 Style-transferred breakdown:")
        for class_name, count in style_by_class.items():
            print(f"  - {class_name}: {count} samples")
        
        # Shuffle BUSI samples for random split (with seed for reproducibility)
        np.random.seed(42)
        np.random.shuffle(busi_pairs)
        
        # Split BUSI samples into train/test/val (70/20/10)
        total_busi = len(busi_pairs)
        busi_train_size = int(0.7 * total_busi)
        busi_test_size = int(0.2 * total_busi)
        
        busi_train = busi_pairs[:busi_train_size]
        busi_test = busi_pairs[busi_train_size:busi_train_size + busi_test_size]
        busi_val = busi_pairs[busi_train_size + busi_test_size:]
        
        # Combine training data: BUSI train + ALL style-transferred samples
        # Test/Val data: ONLY original BUSI samples
        train_pairs = busi_train + style_pairs  # Augmented training set
        test_pairs = busi_test                  # Pure BUSI test set
        val_pairs = busi_val                    # Pure BUSI validation set
        
        # Shuffle training set (contains both BUSI and style-transferred)
        np.random.shuffle(train_pairs)
        
        print(f"\n🎯 NEW SPLIT BREAKDOWN:")
        
        # Save CSV files and show breakdown
        for split_name, pairs in [('train', train_pairs), ('test', test_pairs), ('val', val_pairs)]:
            # Remove class column before saving (not needed in CSV)
            csv_pairs = [{'image_path': p['image_path'], 'mask_path': p['mask_path']} for p in pairs]
            
            df = pd.DataFrame(csv_pairs)
            csv_path = os.path.join(combined_dir, f'{split_name}_frame.csv')
            df.to_csv(csv_path, index=False)
            
            # Count sample types in each split
            busi_count = len([p for p in pairs if p['image_path'].startswith('busi_')])
            style_count = len([p for p in pairs if p['image_path'].startswith('style_')])
            
            # Count by class
            class_counts = defaultdict(int)
            for pair in pairs:
                class_counts[pair['class']] += 1
            
            print(f"\n{split_name.upper()}: {len(pairs)} total samples")
            print(f"  📁 Original BUSI: {busi_count}")
            print(f"  🎨 Style-transferred: {style_count}")
            print(f"  📊 Class breakdown:")
            for class_name, count in class_counts.items():
                print(f"    - {class_name}: {count}")
        
        print(f"\n✅ FIXED! PROPER EXPERIMENTAL SETUP ACHIEVED:")
        print(f"  🏋️ Training: Original BUSI + Style-transferred (augmented data)")
        print(f"  🧪 Testing: ONLY original BUSI (clean evaluation)")
        print(f"  🔬 Validation: ONLY original BUSI (clean evaluation)")
        print(f"\n📄 Updated CSV files saved in: {combined_dir}")
        
        return {
            'train_total': len(train_pairs),
            'train_busi': len(busi_train),
            'train_style': len(style_pairs),
            'test_total': len(test_pairs),
            'val_total': len(val_pairs)
        }

    def verify_fix(combined_dir="dataset/BioMedicalDataset/BUSI-Combined"):
        """Verify the fix was applied correctly"""
        
        print(f"\n🔍 VERIFICATION:")
        
        for split in ['train', 'test', 'val']:
            csv_path = os.path.join(combined_dir, f'{split}_frame.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                busi_count = len(df[df['image_path'].str.startswith('busi_')])
                style_count = len(df[df['image_path'].str.startswith('style_')])
                
                print(f"  {split}: {len(df)} samples ({busi_count} BUSI + {style_count} style)")
                
                if split in ['test', 'val'] and style_count > 0:
                    print(f"    ❌ ERROR: {split} contains {style_count} style-transferred samples!")
                elif split in ['test', 'val'] and style_count == 0:
                    print(f"    ✅ GOOD: {split} contains only original BUSI samples")
                elif split == 'train' and style_count > 0:
                    print(f"    ✅ GOOD: {split} contains both BUSI and style-transferred samples")

    if __name__ == "__main__":
        # Fix the dataset
        results = fix_busi_combined_dataset()
        
        # Verify the fix
        verify_fix()
        
        print(f"\n🎉 BUSI-Combined dataset fixed successfully!")
        print(f"   Now ready for fair comparison with synthetic generation approach!") 