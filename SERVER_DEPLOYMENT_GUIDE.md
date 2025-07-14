# Complete Server Deployment Guide
## Privacy-Preserving Style Transfer + Training Pipeline

This guide walks through the complete process from scratch on a server: generating styled data, training, and evaluation.

## 📋 Prerequisites

- Server with GPU access
- Conda/Python environment
- Access to BUSI and BUS-UCLM datasets on server

## 🚀 Step-by-Step Server Deployment

### Step 1: Clone Repository and Setup Environment

```bash
# Clone your repository (without datasets - they're in gitignore)
git clone https://github.com/your_username/your_repo.git
cd your_repo

# Create conda environment
conda create -n bus_segmentation python=3.8 -y
conda activate bus_segmentation

# Install requirements for privacy-preserving style transfer
pip install -r requirements_privacy_preserving.txt

# Install requirements for training (if you have a separate requirements file)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  # Your main requirements file
```

### Step 2: Upload Datasets to Server

```bash
# Upload BUSI dataset to server
scp -r /local/path/to/BUSI/ user@server:~/BUS_Segmentation/dataset/BioMedicalDataset/

# Upload BUS-UCLM dataset to server  
scp -r /local/path/to/BUS-UCLM/ user@server:~/BUS_Segmentation/dataset/BioMedicalDataset/

# Verify dataset structure
ls dataset/BioMedicalDataset/BUSI/
ls dataset/BioMedicalDataset/BUS-UCLM/
```

Expected structure:
```
dataset/BioMedicalDataset/
├── BUSI/
│   ├── train_frame.csv
│   ├── test_frame.csv
│   ├── val_frame.csv
│   ├── benign/
│   └── malignant/
└── BUS-UCLM/
    ├── train_frame.csv
    ├── test_frame.csv
    ├── benign/
    └── malignant/
```

### Step 3: Generate Privacy-Preserving Styled Dataset

```bash
# Run the complete privacy-preserving style transfer pipeline
python run_full_privacy_preserving_pipeline.py
```

This will:
- ✅ Extract BUSI style statistics (485 images)
- ✅ Generate 184 BUSI-styled BUS-UCLM images using gradient_based method
- ✅ Create combined training CSV (669 total samples)
- ✅ Output: `dataset/BioMedicalDataset/BUS-UCLM-Privacy-Styled/`

### Step 4: Train MADGNet on Combined Dataset

```bash
# Train MADGNet using original BUSI + privacy-styled BUS-UCLM
python train_madgnet_with_privacy_styled_data.py \
  --combined_csv dataset/BioMedicalDataset/combined_privacy_preserving_train.csv \
  --test_csv dataset/BioMedicalDataset/BUSI/test_frame.csv \
  --epochs 100 \
  --batch_size 8 \
  --lr 0.001 \
  --output_dir privacy_preserving_results/
```

### Step 5: Evaluate on BUSI Test Set

```bash
# Evaluate trained model on BUSI test set
python evaluate_privacy_preserving_model.py \
  --model_path privacy_preserving_results/best_model.pth \
  --test_csv dataset/BioMedicalDataset/BUSI/test_frame.csv \
  --test_dataset_path dataset/BioMedicalDataset/BUSI \
  --output_dir privacy_preserving_results/evaluation/
```

## 📊 Expected Timeline

| Step | Estimated Time | Resources |
|------|---------------|-----------|
| Environment Setup | 10-15 minutes | CPU |
| Dataset Upload | 20-30 minutes | Network |
| Style Transfer Generation | 2-5 minutes | CPU |
| MADGNet Training | 2-4 hours | GPU |
| Evaluation | 5-10 minutes | GPU |

## 🎯 Expected Results

Based on CCST paper findings:
- **Dice Score**: +9.16% improvement over BUSI-only training
- **IoU**: +9.46% improvement
- **Hausdorff Distance**: -17.28% improvement

## 📈 Resource Requirements

### Minimum:
- **RAM**: 16GB
- **GPU**: 8GB VRAM (RTX 3070 equivalent)
- **Storage**: 20GB free space

### Recommended:
- **RAM**: 32GB+
- **GPU**: 24GB VRAM (RTX 4090/A100)
- **Storage**: 50GB+ free space

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   --batch_size 4
   ```

2. **Dataset Path Issues**:
   ```bash
   # Verify paths
   python -c "import pandas as pd; print(pd.read_csv('dataset/BioMedicalDataset/BUSI/train_frame.csv').head())"
   ```

3. **Style Transfer Fails**:
   ```bash
   # Test single image first
   python test_privacy_preserving_methods.py
   ```

## 🎉 Success Indicators

You'll know it's working when you see:

1. **Style Transfer**: `✅ Generated 184 styled images`
2. **Training**: Dice loss decreasing over epochs
3. **Evaluation**: Metrics improvement over baseline

## 📁 Final Output Structure

After completion:
```
BUS_Segmentation/
├── dataset/BioMedicalDataset/
│   ├── BUSI/                           # Original BUSI
│   ├── BUS-UCLM/                       # Original BUS-UCLM  
│   ├── BUS-UCLM-Privacy-Styled/        # Generated styled data
│   └── combined_privacy_preserving_train.csv
├── privacy_preserving_results/
│   ├── best_model.pth
│   ├── training_logs.txt
│   └── evaluation/
│       ├── dice_scores.csv
│       ├── iou_scores.csv
│       └── predictions/
└── privacy_style_stats/
    └── busi_privacy_stats.json
```

## 🔐 Privacy Benefits Achieved

✅ **No raw BUSI images shared between institutions**  
✅ **Only statistical information exchanged**  
✅ **Federated learning compatible**  
✅ **GDPR/HIPAA compliant approach**  

This approach simulates a real federated learning scenario where Institution A (BUSI) shares only style statistics, and Institution B (BUS-UCLM) generates locally-styled data for improved training. 