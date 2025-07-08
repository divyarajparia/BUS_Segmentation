# MADGNET Training with Synthetic BUSI Data Pipeline

## 🎯 **Objective**
Train MADGNET on **BUSI training data + 264 synthetic samples** and test on **original BUSI test data only** for fair evaluation.

## 📋 **Pipeline Overview**

1. **Generate Synthetic Data**: Use trained GAN to create 175 benign + 89 malignant samples
2. **Combine Datasets**: Merge BUSI training data with synthetic data
3. **Train MADGNET**: Train on combined dataset (original + synthetic)
4. **Evaluate**: Test on original BUSI test data only (fair assessment)

## 🚀 **Quick Start**

### **Option 1: One-Click Pipeline (Recommended)**
```bash
# Submit complete pipeline to SLURM
sbatch train_madgnet_synthetic.job
```

### **Option 2: Manual Steps**

**Step 1: Generate Synthetic Data**
```bash
python generate_busi_synthetic_final.py \
    --checkpoint checkpoints/busi_gan_final.pth \
    --output_dir synthetic_busi_madgnet \
    --num_benign 175 \
    --num_malignant 89
```

**Step 2: Train MADGNET**
```bash
python train_madgnet_with_synthetic.py \
    --gan_checkpoint checkpoints/busi_gan_final.pth \
    --madgnet_epochs 200
```

**Step 3: Individual Commands (if needed)**
```bash
# Train MADGNET on combined dataset
python IS2D_main.py \
    --train_data_type BUSI-Synthetic-Combined \
    --test_data_type BUSI \
    --synthetic_data_dir synthetic_busi_madgnet \
    --final_epoch 200 \
    --train

# Evaluate on original BUSI test
python IS2D_main.py \
    --train_data_type BUSI-Synthetic-Combined \
    --test_data_type BUSI \
    --synthetic_data_dir synthetic_busi_madgnet \
    --final_epoch 200
```

## 📁 **File Structure**

### **Created Files:**
- `generate_busi_synthetic_final.py` - Generate 264 synthetic samples
- `dataset/BioMedicalDataset/BUSISyntheticCombinedDataset.py` - Combined dataset class
- `train_madgnet_with_synthetic.py` - Complete pipeline script
- `train_madgnet_synthetic.job` - SLURM job script

### **Expected Output:**
```
synthetic_busi_madgnet/
├── benign/
│   ├── image/ (175 synthetic benign images)
│   └── mask/  (175 corresponding masks)
├── malignant/
│   ├── image/ (89 synthetic malignant images)
│   └── mask/  (89 corresponding masks)
└── synthetic_dataset.csv

model_weights/BUSI-Synthetic-Combined/
├── model_weights/
│   └── model_weight(EPOCH 200).pth.tar
└── test_reports/
    └── test_report(EPOCH 200).txt
```

## 🧠 **Dataset Composition**

### **Training Data** (Combined):
- **Original BUSI training samples**: ~400-500 samples
- **Synthetic samples**: 264 samples (175 benign + 89 malignant)
- **Total training data**: ~650-750 samples

### **Test Data** (Original Only):
- **Original BUSI test samples**: ~160 samples
- **No synthetic data**: Ensures fair evaluation

## 📊 **Expected Benefits**

1. **Increased Training Data**: More diverse examples for better learning
2. **Class Balance**: Better balance between benign/malignant samples
3. **Improved Generalization**: More robust feature learning
4. **Fair Evaluation**: Testing only on real data ensures valid results

## 🔧 **Configuration Options**

### **Synthetic Data Generation:**
- `--num_benign 175` - Number of benign samples
- `--num_malignant 89` - Number of malignant samples
- `--output_dir` - Where to save synthetic data

### **MADGNET Training:**
- `--madgnet_epochs 200` - Training epochs
- `--batch_size 4` - Training batch size
- `--gpu 0` - GPU device to use

## 📈 **Monitoring Progress**

### **Check Training Progress:**
```bash
# Check job status
squeue -u $USER

# Monitor training logs
tail -f logs/madgnet_synthetic_*.out
```

### **View Results:**
```bash
# Check test results
cat model_weights/BUSI-Synthetic-Combined/test_reports/test_report*.txt

# Compare with baseline BUSI-only training
cat model_weights/BUSI/test_reports/test_report*.txt
```

## 🔬 **Experimental Validation**

### **Baseline Comparison:**
1. **BUSI Only**: Train on BUSI training, test on BUSI test
2. **BUSI + Synthetic**: Train on BUSI + synthetic, test on BUSI test
3. **Compare metrics**: DSC, IoU, WeightedF-Measure, S-Measure, E-Measure, MAE

### **Expected Metrics:**
- **DSC (Dice)**: Should improve from ~0.82 to ~0.85+
- **IoU**: Should improve from ~0.74 to ~0.77+
- **Other metrics**: Overall improvement expected

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **GAN checkpoint not found:**
   ```bash
   ls checkpoints/  # Check available checkpoints
   # Update path in train_madgnet_synthetic.job
   ```

2. **Out of memory:**
   ```bash
   # Reduce batch size in job script
   BATCH_SIZE=2  # Instead of 4
   ```

3. **Dataset not found:**
   ```bash
   # Check BUSI dataset exists
   ls dataset/BioMedicalDataset/BUSI/
   ```

4. **Synthetic generation fails:**
   ```bash
   # Test GAN separately first
   python test_fixed_gan.py
   ```

## 🎯 **Success Criteria**

✅ **Synthetic Data Generated**: 264 samples (175 benign + 89 malignant)
✅ **Training Completes**: No errors during MADGNET training
✅ **Fair Evaluation**: Test uses only original BUSI data
✅ **Results Available**: Test metrics saved in test_reports/
✅ **Improvement**: Better metrics compared to BUSI-only baseline

## 📞 **Next Steps**

1. **Run the pipeline**: `sbatch train_madgnet_synthetic.job`
2. **Monitor progress**: Check logs and training status
3. **Analyze results**: Compare with baseline BUSI performance
4. **Document findings**: Record improvement in segmentation metrics

## 🔬 **Research Impact**

This pipeline demonstrates:
- **Synthetic data augmentation** for medical image segmentation
- **GAN-based data generation** for ultrasound imaging
- **Fair evaluation methodology** (synthetic in training only)
- **MADGNET performance improvement** through data augmentation

Perfect for publication in medical imaging conferences/journals! 🎓 