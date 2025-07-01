# 🔧 MANUAL STEPS TO CREATE DEBUG DATASET

## 1. ON YOUR SERVER, run these commands:

```bash
# Navigate to your project
cd /path/to/your/BUS_Segmentation

# Copy a few sample images (5-10 samples total)
mkdir -p debug_data/BUSI/benign/image debug_data/BUSI/benign/mask
mkdir -p debug_data/BUSI/malignant/image debug_data/BUSI/malignant/mask
mkdir -p debug_data/checkpoints

# Copy 3 benign samples
ls dataset/BioMedicalDataset/BUSI/benign/image/benign_*.png | head -3 | xargs -I {} cp {} debug_data/BUSI/benign/image/
ls dataset/BioMedicalDataset/BUSI/benign/mask/benign_*_mask.png | head -3 | xargs -I {} cp {} debug_data/BUSI/benign/mask/

# Copy 2 malignant samples  
ls dataset/BioMedicalDataset/BUSI/malignant/image/malignant_*.png | head -2 | xargs -I {} cp {} debug_data/BUSI/malignant/image/
ls dataset/BioMedicalDataset/BUSI/malignant/mask/malignant_*_mask.png | head -2 | xargs -I {} cp {} debug_data/BUSI/malignant/mask/

# Copy the trained model (if it exists)
cp diffusion_model_epoch_50.pth debug_data/checkpoints/ 2>/dev/null || echo "No checkpoint found"

# Create proper CSV files with actual filenames
python -c "
import os
import pandas as pd

rows = []
for class_name in ['benign', 'malignant']:
    image_dir = f'debug_data/BUSI/{class_name}/image'
    mask_dir = f'debug_data/BUSI/{class_name}/mask'
    
    if os.path.exists(image_dir):
        for img_file in sorted(os.listdir(image_dir)):
            if img_file.endswith('.png'):
                mask_file = img_file.replace('.png', '_mask.png')
                if os.path.exists(os.path.join(mask_dir, mask_file)):
                    rows.append({
                        'image_path': f'{class_name}/image/{img_file}',
                        'mask_path': f'{class_name}/mask/{mask_file}',
                        'class': class_name
                    })

df = pd.DataFrame(rows)
df.to_csv('debug_data/BUSI/train_frame.csv', index=False)
df.iloc[:3].to_csv('debug_data/BUSI/test_frame.csv', index=False)
df.iloc[:2].to_csv('debug_data/BUSI/val_frame.csv', index=False)
print(f'Created CSV files with {len(df)} samples')
"
```

## 2. DOWNLOAD THE DEBUG DATA:

```bash
# Compress the debug data
tar -czf debug_data.tar.gz debug_data/

# Download this file to your local machine
# Then extract it in your local project directory
```

## 3. UPDATE .GITIGNORE (on server):

```bash
# Add rules to allow debug_data
cat >> .gitignore << EOF

# Allow debug data for local testing
!debug_data/
debug_data/**
!debug_data/BUSI/**/*.png
!debug_data/BUSI/**/*.csv
!debug_data/checkpoints/*.pth
EOF
```

## 4. COMMIT DEBUG DATA:

```bash
git add debug_data/
git add .gitignore
git commit -m "Add debug dataset for local testing"
git push
```

## 5. LOCALLY:

```bash
git pull
# Now you can test diffusion locally with debug_data/
python test_debug_diffusion.py
```

## What this gives you:

- **Real BUSI images** (5 samples) for testing
- **Real masks** to verify data loading
- **Model checkpoint** for testing generation
- **Proper CSV files** with correct paths
- **Local debugging** capability

This lets you debug the diffusion model locally and identify why it's generating black images!
