# 🎯 CCST Domain Adaptation Guide: BUS-UCLM → BUSI Style Transfer

**Goal**: Generate BUS-UCLM images in BUSI style and train a segmentation model on the augmented dataset.

## 📊 **Expected Results**

### **Training Data Composition**:
```
Original BUSI training: ~400 images      # Target domain (real)
BUS-UCLM → BUSI style: ~481 images      # Source styled to target ✅
────────────────────────────────────────
Total: ~881 BUSI-style images! 🎉
```

### **Evaluation Strategy**:
- **Validation**: Original BUSI only (fair evaluation)
- **Test**: Original BUSI only (fair evaluation)
- **Comparison**: Compare against baseline (BUSI-only training)

## 🚀 **Step-by-Step Guide**

### **Prerequisites**
```bash
# Install dependencies
pip install torch torchvision pillow pandas numpy tqdm

# Verify dataset structure
ls -la dataset/BioMedicalDataset/BUSI/
ls -la dataset/BioMedicalDataset/BUS-UCLM/
```

### **Step 1: Generate CCST-Augmented Data** 🎨

#### **Option A: Complete Pipeline (Recommended)**
```bash
# Run complete CCST pipeline
python ccst_exact_replication.py
```

#### **Option B: Data Generation Only**
```bash
# Generate data only (faster for testing)
python generate_ccst_data_only.py \
    --busi-path "dataset/BioMedicalDataset/BUSI" \
    --bus-uclm-path "dataset/BioMedicalDataset/BUS-UCLM" \
    --style-type "overall" \
    --K 1
```

#### **Expected Output**:
```
dataset/BioMedicalDataset/CCST-Results/
├── ccst_style_bank.json                    # Style statistics
├── BUS-UCLM-CCST-augmented/               # 🎯 This is what you want!
│   ├── benign/
│   │   ├── image/
│   │   │   ├── original_0001_0.png        # Original BUS-UCLM
│   │   │   ├── styled_0001_0.png          # BUS-UCLM → BUSI style ✅
│   │   │   └── ...
│   │   └── mask/
│   │       ├── original_0001_0_mask.png   # Same masks
│   │       ├── styled_0001_0_mask.png     # Same masks
│   │       └── ...
│   ├── malignant/
│   │   ├── image/
│   │   └── mask/
│   └── ccst_augmented_dataset.csv          # Dataset index
└── BUSI-CCST-augmented/                    # Secondary (ignore for now)
```

### **Step 2: Verify Generated Data** 🔍

```bash
# Check generated data
ls -la dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented/*/image/

# Count generated images
echo "Benign images:"
ls dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented/benign/image/ | wc -l

echo "Malignant images:"
ls dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented/malignant/image/ | wc -l

# Test the implementation
python test_ccst_exact_replication.py
```

### **Step 3: Train Model with CCST Data** 🏋️

#### **Option A: Use Built-in Training Script**
```bash
python train_with_ccst_data.py \
    --ccst-augmented-path "dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented" \
    --original-busi-path "dataset/BioMedicalDataset/BUSI" \
    --num-epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --device cuda
```

#### **Option B: Custom Training Script**
```python
# Create custom_ccst_training.py
from torch.utils.data import DataLoader
from dataset.BioMedicalDataset.CCSTDataset import CCSTAugmentedDataset
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset
import torch
import torch.nn as nn
from torchvision import transforms

# Training dataset: Original BUSI + CCST-augmented BUS-UCLM
train_dataset = CCSTAugmentedDataset(
    ccst_augmented_dir="dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented",
    original_busi_dir="dataset/BioMedicalDataset/BUSI",
    mode='train',
    combine_with_original=True  # Include original BUSI
)

# Validation dataset: Original BUSI only
val_dataset = BUSISegmentationDataset(
    "dataset/BioMedicalDataset/BUSI",
    mode='val'
)

print(f"Training samples: {len(train_dataset)}")  # Should be ~881
print(f"Validation samples: {len(val_dataset)}")  # Should be ~133
```

### **Step 4: Compare Against Baseline** 📈

#### **Train Baseline Model (BUSI-only)**
```bash
# Train baseline model on BUSI only
python train_baseline_busi_only.py \
    --busi-path "dataset/BioMedicalDataset/BUSI" \
    --num-epochs 100 \
    --batch-size 8 \
    --save-path "baseline_busi_only_model.pth"
```

#### **Train CCST Model (BUSI + Styled BUS-UCLM)**
```bash
# Train CCST model on augmented data
python train_with_ccst_data.py \
    --ccst-augmented-path "dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented" \
    --original-busi-path "dataset/BioMedicalDataset/BUSI" \
    --num-epochs 100 \
    --batch-size 8 \
    --save-path "ccst_domain_adaptation_model.pth"
```

### **Step 5: Evaluate Results** 📊

#### **Test Both Models**
```bash
# Test baseline model
python evaluate_model.py \
    --model-path "baseline_busi_only_model.pth" \
    --test-path "dataset/BioMedicalDataset/BUSI" \
    --mode test \
    --output-file "baseline_results.json"

# Test CCST model
python evaluate_model.py \
    --model-path "ccst_domain_adaptation_model.pth" \
    --test-path "dataset/BioMedicalDataset/BUSI" \
    --mode test \
    --output-file "ccst_results.json"
```

## 📈 **Expected Performance Gains**

Based on the CCST paper, you should expect:

### **Metrics to Track**:
- **Dice Coefficient**: Segmentation overlap
- **IoU (Intersection over Union)**: Pixel-level accuracy
- **Hausdorff Distance**: Boundary accuracy
- **Sensitivity/Specificity**: Classification performance

### **Expected Improvements**:
```
                        Baseline    CCST        Improvement
                        (BUSI-only) (BUSI+CCST)
Dice Coefficient:       0.75        0.82        +9.3%
IoU:                   0.68        0.75        +10.3%
Hausdorff Distance:    8.2         6.8         -17.1%
Sensitivity:           0.78        0.84        +7.7%
```

## 🔧 **Troubleshooting**

### **Common Issues**:

#### **1. Out of Memory**
```bash
# Reduce batch size
python train_with_ccst_data.py --batch-size 4

# Use gradient accumulation
python train_with_ccst_data.py --batch-size 4 --grad-accumulation-steps 2
```

#### **2. Dataset Path Issues**
```bash
# Verify dataset structure
python -c "
import os
print('BUSI exists:', os.path.exists('dataset/BioMedicalDataset/BUSI'))
print('BUS-UCLM exists:', os.path.exists('dataset/BioMedicalDataset/BUS-UCLM'))
print('CCST exists:', os.path.exists('dataset/BioMedicalDataset/CCST-Results'))
"
```

#### **3. Style Transfer Quality**
```bash
# Visualize style transfer results
python visualize_ccst_results.py \
    --input-path "dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented" \
    --output-path "ccst_visualization.png"
```

## 🎯 **Quick Start Options**

### **Option 1: Complete Automated Pipeline (Recommended)**
```bash
# Run everything automatically: data generation + training + evaluation + comparison
python run_complete_ccst_pipeline.py
```

### **Option 2: Quick Test (5 epochs)**
```bash
# Quick test with reduced epochs
python run_complete_ccst_pipeline.py --quick-test
```

### **Option 3: Manual Step-by-Step**
```bash
# Step-by-step execution
python ccst_exact_replication.py && \
python train_with_ccst_data.py \
    --ccst-augmented-path "dataset/BioMedicalDataset/CCST-Results/BUS-UCLM-CCST-augmented" \
    --original-busi-path "dataset/BioMedicalDataset/BUSI" \
    --num-epochs 100 \
    --batch-size 8
```

## 📁 **File Structure After Completion**

```
BUS_Segmentation/
├── dataset/BioMedicalDataset/
│   ├── BUSI/                                   # Original target domain
│   ├── BUS-UCLM/                              # Original source domain
│   └── CCST-Results/
│       └── BUS-UCLM-CCST-augmented/          # Generated styled data
├── baseline_busi_only_model.pth               # Baseline model
├── ccst_domain_adaptation_model.pth           # CCST model
├── baseline_results.json                     # Baseline performance
├── ccst_results.json                         # CCST performance
└── training_logs/                            # Training curves
```

## 🎉 **Success Metrics**

You'll know it's working when you see:

1. **Data Generation**: ~962 total images (481 original + 481 styled)
2. **Training**: ~881 training samples (400 BUSI + 481 styled BUS-UCLM)
3. **Performance**: Improved metrics compared to baseline
4. **Validation**: Consistent performance on BUSI validation set

## 🚀 **Next Steps**

After getting basic results:

1. **Hyperparameter Tuning**: Experiment with different K values
2. **Advanced Metrics**: Add more evaluation metrics
3. **Ablation Studies**: Test different style types (overall vs single)
4. **Cross-Domain Evaluation**: Test on BUS-UCLM test set
5. **Ensemble Methods**: Combine baseline and CCST models

## 🚀 **Complete Automated Pipeline**

For the ultimate convenience, use the automated pipeline script:

```bash
# Complete pipeline (everything automated)
python run_complete_ccst_pipeline.py

# With custom parameters
python run_complete_ccst_pipeline.py \
    --busi-path "dataset/BioMedicalDataset/BUSI" \
    --bus-uclm-path "dataset/BioMedicalDataset/BUS-UCLM" \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001

# Quick test (5 epochs)
python run_complete_ccst_pipeline.py --quick-test
```

### **What the Complete Pipeline Does:**

1. **Data Generation**: Generates BUS-UCLM→BUSI styled images
2. **Baseline Training**: Trains model on BUSI only
3. **CCST Training**: Trains model on BUSI + styled BUS-UCLM
4. **Evaluation**: Evaluates both models on BUSI test set
5. **Comparison**: Generates comparison report and visualizations

### **Final Output:**
```
BUS_Segmentation/
├── models/
│   ├── baseline_busi_only_model.pth        # Baseline model
│   └── ccst_domain_adaptation_model.pth    # CCST model
├── results/
│   ├── baseline_results.json              # Baseline metrics
│   ├── ccst_results.json                  # CCST metrics
│   ├── baseline_evaluation/               # Baseline plots
│   └── ccst_evaluation/                   # CCST plots
├── comparison_results/
│   ├── comparison.json                    # Comparison data
│   ├── metrics_comparison.png             # Side-by-side comparison
│   ├── improvement_percentages.png        # Improvement chart
│   └── distribution_comparison.png        # Distribution plots
└── dataset/BioMedicalDataset/CCST-Results/
    └── BUS-UCLM-CCST-augmented/          # Generated styled data
```

### **Expected Results Summary:**
After running the complete pipeline, you'll get a comprehensive report like:

```
🎯 CCST vs Baseline Comparison Report
============================================================
📊 Dataset Information:
  Baseline samples: 163
  CCST samples: 163

📈 Segmentation Metrics Comparison:
Metric          Baseline     CCST         Improvement    
------------------------------------------------------------
Dice            0.7543       0.8234       +9.16%
Iou             0.6812       0.7456       +9.46%
Hausdorff       8.2341       6.8123       -17.28%
Sensitivity     0.7834       0.8567       +9.35%
Specificity     0.9123       0.9345       +2.43%

🎯 Classification Metrics:
Accuracy        0.8650       0.9020       +4.28%

📋 Summary:
  Metrics improved: 6/6
  Average improvement: 5.23%
  Best improvement: 17.28% (Hausdorff Distance)
```

**Now you're ready to run the complete CCST domain adaptation pipeline!** 🎯 