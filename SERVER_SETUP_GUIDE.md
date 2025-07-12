# 🚀 CCST Server Setup Guide

Complete guide for running CCST domain adaptation on your server.

## 📁 Files to Upload to Server

Upload these files to your server project directory:

### Core CCST Files:
```
ccst_exact_replication.py           # Main CCST pipeline
dataset/BioMedicalDataset/CCSTDataset.py  # CCST dataset classes
train_with_ccst_data.py              # Training with CCST data
test_ccst_exact_replication.py       # Testing/verification
generate_ccst_data_only.py           # Generate data only
run_ccst_server.job                  # SLURM job script
```

### Documentation (Optional):
```
README_CCST_Exact_Replication.md    # Complete methodology explanation
SERVER_SETUP_GUIDE.md               # This file
```

## 🔧 Server Configuration

### 1. Update Job Script Paths

Edit `run_ccst_server.job` and update these lines:

```bash
# Replace with your actual paths
cd /path/to/your/BUS_Segmentation/        # Your project directory
# source /path/to/your/venv/bin/activate  # Your virtual environment

# Update module versions for your server
module load python/3.8
module load cuda/11.8
module load cudnn/8.6
```

### 2. Update Dataset Paths

If your datasets are in different locations, update the paths in:

**Option A: Edit the default paths in scripts**
```python
# In ccst_exact_replication.py, main() function:
busi_path = "/your/actual/path/to/BUSI"
bus_uclm_path = "/your/actual/path/to/BUS-UCLM"
```

**Option B: Use command line arguments**
```bash
python generate_ccst_data_only.py \
    --busi-path "/your/actual/path/to/BUSI" \
    --bus-uclm-path "/your/actual/path/to/BUS-UCLM"
```

## 🚀 Server Execution Options

### Option 1: Complete Pipeline (Recommended)
```bash
# Submit job that does everything
sbatch run_ccst_server.job
```

### Option 2: Step-by-Step Execution
```bash
# Step 1: Generate CCST data only
python generate_ccst_data_only.py

# Step 2: Inspect generated data
ls -la dataset/BioMedicalDataset/CCST-Results/

# Step 3: Train with generated data
python train_with_ccst_data.py \
    --ccst-augmented-path "dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented" \
    --num-epochs 100 \
    --batch-size 8
```

### Option 3: Interactive Session
```bash
# Start interactive session
srun --partition=gpu --gres=gpu:1 --pty bash

# Load modules
module load python/3.8 cuda/11.8 cudnn/8.6

# Run commands directly
python ccst_exact_replication.py
python train_with_ccst_data.py
```

## 📊 Expected Output Structure

After running CCST pipeline, you'll have:

```
dataset/BioMedicalDataset/CCST-Results/
├── ccst_style_bank.json                    # Federated style bank
├── BUS-UCLM-CCST-augmented/               # 🎯 Primary for domain adaptation
│   ├── benign/
│   │   ├── image/
│   │   │   ├── original_0001_0.png        # Original BUS-UCLM images
│   │   │   ├── styled_0001_0.png          # BUS-UCLM → BUSI style ✅
│   │   │   └── ...
│   │   └── mask/
│   │       ├── original_0001_0_mask.png
│   │       ├── styled_0001_0_mask.png
│   │       └── ...
│   ├── malignant/
│   │   ├── image/
│   │   └── mask/
│   └── ccst_augmented_dataset.csv          # Dataset index
└── BUSI-CCST-augmented/                    # Secondary (optional)
    ├── benign/
    ├── malignant/
    └── ccst_augmented_dataset.csv
```

## 🎯 Training Data Composition

With **K=1** (optimal for 2 domains), your training data will be:

```
Training Dataset:
├── Original BUSI: ~780 images        # Target domain
├── Original BUS-UCLM: ~500 images    # Source domain  
└── BUS-UCLM → BUSI style: ~500 images # Source styled to target ✅

Total: ~1780 images (2.3x increase)
```

## 🔍 Monitoring Job Progress

```bash
# Check job status
squeue -u $USER

# Monitor job output
tail -f logs/ccst_pipeline_[JOB_ID].out

# Check errors
tail -f logs/ccst_pipeline_[JOB_ID].err
```

## 📈 Performance Expectations

### Data Generation Time:
- **Style extraction**: 5-10 minutes
- **Style transfer**: 10-20 minutes  
- **Total**: ~15-30 minutes

### Training Time:
- **100 epochs**: 2-4 hours (depends on GPU)
- **Memory usage**: ~8-16GB GPU memory
- **Disk usage**: ~2-5GB for augmented data

## 🛠️ Troubleshooting

### Common Issues:

1. **Dataset not found**
   ```bash
   # Check paths
   ls -la dataset/BioMedicalDataset/BUSI/
   ls -la dataset/BioMedicalDataset/BUS-UCLM/
   ```

2. **CUDA out of memory**
   ```bash
   # Reduce batch size
   python train_with_ccst_data.py --batch-size 4
   ```

3. **Module not found**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   # Add project directory to path
   export PYTHONPATH=/path/to/your/project:$PYTHONPATH
   ```

4. **Permission denied**
   ```bash
   # Make scripts executable
   chmod +x *.py
   chmod +x run_ccst_server.job
   ```

## 🎯 Key Domain Adaptation Points

### What CCST Does:
1. **Extracts BUSI domain style** (target domain characteristics)
2. **Applies to BUS-UCLM images** (source domain content)
3. **Creates hybrid data**: BUS-UCLM content + BUSI style
4. **Trains on mixed data**: Original BUSI + Styled BUS-UCLM

### Why It Works:
- **Bridges domain gap**: BUS-UCLM images look more like BUSI
- **Preserves anatomical content**: Tumors/structures unchanged
- **Increases training data**: 2x more data in target domain style
- **Improves generalization**: Model sees diverse BUSI-style images

## 📋 Quick Commands Summary

```bash
# 1. Upload files to server
scp ccst_*.py your_server:/path/to/project/

# 2. SSH to server
ssh your_server

# 3. Submit job
sbatch run_ccst_server.job

# 4. Monitor
squeue -u $USER
tail -f logs/ccst_pipeline_*.out
```

## 🎉 Success Indicators

You'll know it's working when you see:
- ✅ Style extraction completed for both domains
- ✅ Style bank created and saved
- ✅ Augmented datasets generated
- ✅ Training loss decreasing
- ✅ Validation performance improving
- ✅ Final model saved as `best_ccst_model.pth`

**Expected final performance**: Better generalization on BUSI test set compared to training without CCST! 