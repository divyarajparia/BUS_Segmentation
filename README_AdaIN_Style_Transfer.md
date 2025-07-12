# AdaIN-Based Style Transfer for Medical Image Domain Adaptation

This implementation follows the **CCST (Cross-Client Style Transfer)** methodology from the paper "Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer" but adapted for **single-institution domain adaptation**.

## 🎯 **Goal**

Transform **BUS-UCLM** images to match the **BUSI** style, creating an augmented training dataset for better domain generalization.

## 📋 **Overview**

### **Problem**: 
- You have two ultrasound datasets: BUSI and BUS-UCLM
- Models trained on BUSI work well on BUSI test data
- But they don't generalize well to BUS-UCLM data due to domain gap

### **Solution**:
- Use **AdaIN (Adaptive Instance Normalization)** to transfer BUS-UCLM images to BUSI style
- Train models on: Original BUSI + Style-transferred BUS-UCLM
- Evaluate on: Original BUSI (fair comparison)

## 🏗️ **Architecture**

### **AdaIN Style Transfer**
Following the CCST paper's Equation (2):
```
AdaIN(Fc, Fs) = σ(Fs) * (Fc - μ(Fc)) / σ(Fc) + μ(Fs)
```

Where:
- `Fc`: Content features (from BUS-UCLM images)
- `Fs`: Style features (from BUSI domain)
- `μ(·)`, `σ(·)`: Channel-wise mean and standard deviation

### **Key Components**:

1. **VGG Encoder**: Extracts features up to relu4_1 layer
2. **AdaIN Module**: Performs style transfer in feature space
3. **Decoder**: Reconstructs images from stylized features
4. **Domain Style Extractor**: Computes overall domain style statistics

## 🚀 **Usage**

### **Step 1: Install Dependencies**
```bash
pip install -r requirements_adain.txt
```

### **Step 2: Generate Styled Data**
```bash
python adain_style_transfer.py
```

This will:
- Extract BUSI domain style from all training images
- Apply style transfer to BUS-UCLM images
- Create styled dataset at `dataset/BioMedicalDataset/BUS-UCLM-AdaIN-styled`

### **Step 3: Train Model with Styled Data**
```bash
python train_with_adain_styled_data.py --batch-size 8 --num-epochs 100
```

### **Step 4: Test the Pipeline**
```bash
python test_adain_style_transfer.py
```

## 📁 **File Structure**

```
BUS_Segmentation/
├── adain_style_transfer.py                    # Main AdaIN implementation
├── dataset/BioMedicalDataset/
│   ├── AdaINStyleTransferDataset.py          # Dataset classes
│   ├── BUSI/                                 # Original BUSI dataset
│   ├── BUS-UCLM/                            # Original BUS-UCLM dataset
│   └── BUS-UCLM-AdaIN-styled/               # Generated styled data
├── train_with_adain_styled_data.py          # Training script
├── test_adain_style_transfer.py             # Test script
├── requirements_adain.txt                   # Dependencies
└── README_AdaIN_Style_Transfer.md           # This file
```

## 🔬 **How It Works**

### **1. Domain Style Extraction**
Following CCST paper's Equation (8):
```python
# Extract style statistics from all BUSI training images
busi_domain_style = extractor.extract_domain_style(busi_path)
# Returns: (domain_mean, domain_std) with shape [1, 512, 1, 1]
```

### **2. Style Transfer Process**
```python
# For each BUS-UCLM image:
content_features = vgg_encoder(bus_uclm_image)
stylized_features = adain(content_features, busi_domain_style)
styled_image = decoder(stylized_features)
```

### **3. Dataset Creation**
```
Training Data = Original BUSI + Styled BUS-UCLM
Validation Data = Original BUSI only
Test Data = Original BUSI only
```

## 🎨 **Key Advantages**

### **Over CycleGAN**:
- ✅ **No training required** - Uses pre-trained VGG features
- ✅ **Faster inference** - Single forward pass
- ✅ **Domain-level consistency** - Uses overall domain statistics
- ✅ **Arbitrary style transfer** - Can handle new styles without retraining

### **Over Original CCST**:
- ✅ **Simplified for single institution** - No federated learning complexity
- ✅ **Medical image focused** - Adapted for grayscale ultrasound
- ✅ **Preserved privacy concepts** - Domain-level statistics only

## 📊 **Expected Results**

### **Training Set Composition**:
```
Original BUSI: ~400 samples
Styled BUS-UCLM: ~500 samples
Total Training: ~900 samples (more than doubled!)
```

### **Evaluation**:
- **Fair comparison**: Test only on original BUSI
- **Domain adaptation**: Better generalization to different machines
- **Data augmentation**: Increased training data diversity

## 🧪 **Comparison with Your CycleGAN**

| Aspect | CycleGAN | AdaIN (This Implementation) |
|--------|----------|---------------------------|
| **Training** | Requires training GAN | Uses pre-trained VGG |
| **Speed** | Slower (GAN training) | Faster (direct transfer) |
| **Quality** | Higher (specialized) | Good (general-purpose) |
| **Flexibility** | Fixed style pairs | Can handle new styles |
| **Privacy** | Uses actual images | Uses only statistics |
| **Medical Focus** | Excellent | Good |

## 🔧 **Configuration**

### **Key Parameters**:
- **VGG Layer**: `relu4_1` (layer 21) for feature extraction
- **Image Size**: 224×224 (VGG requirement)
- **Normalization**: ImageNet statistics for VGG
- **Style Statistics**: Channel-wise mean and std

### **Customization**:
```python
# Change VGG layer
self.encoder = nn.Sequential(*list(vgg.children())[:21])  # relu4_1

# Adjust decoder architecture
self.decoder = nn.Sequential(...)  # Mirror of encoder

# Modify domain style computation
domain_mean = torch.mean(all_features, dim=(0, 2, 3), keepdim=True)
```

## 📈 **Performance Tips**

1. **Batch Processing**: Process multiple images together
2. **GPU Memory**: Use appropriate batch sizes
3. **Feature Caching**: Cache domain style statistics
4. **Mixed Precision**: Use FP16 for faster training

## 🐛 **Troubleshooting**

### **Common Issues**:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python train_with_adain_styled_data.py --batch-size 4
   ```

2. **VGG Download Issues**:
   ```bash
   # Pre-download VGG weights
   python -c "import torchvision.models as models; models.vgg19(pretrained=True)"
   ```

3. **Dataset Path Issues**:
   ```bash
   # Check dataset structure
   ls dataset/BioMedicalDataset/BUSI/
   ls dataset/BioMedicalDataset/BUS-UCLM/
   ```

## 🔬 **Research Context**

This implementation bridges:
- **CCST Paper**: Federated domain generalization methodology
- **Your Research**: Single-institution domain adaptation
- **AdaIN**: Real-time arbitrary style transfer
- **Medical Imaging**: Ultrasound tumor segmentation

## 📚 **References**

1. **CCST Paper**: "Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer"
2. **AdaIN Paper**: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
3. **Your Domain**: Medical ultrasound tumor segmentation

## 🎯 **Next Steps**

1. **Run the pipeline** to generate styled data
2. **Compare results** with your CycleGAN approach
3. **Analyze performance** on BUSI test set
4. **Experiment with variations** (different VGG layers, style mixing)

This implementation gives you a **fast, flexible alternative** to CycleGAN while following the **principled CCST methodology** for domain adaptation! 