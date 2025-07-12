# CCST Exact Replication: Federated Domain Generalization for Medical Imaging

This implementation provides an **exact replication** of the CCST (Cross-Client Style Transfer) methodology from the paper:

**"Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer"**

Adapted for **medical ultrasound tumor segmentation** using your **BUSI** and **BUS-UCLM** datasets.

## 🎯 **What This Implements**

### **Exact Paper Methodology:**
- ✅ **Algorithm 1**: Local Cross-Client Style Transfer (line-by-line implementation)
- ✅ **Three-stage workflow**: Style computation → Style bank broadcasting → Style transfer
- ✅ **Two style types**: Single image style vs Overall domain style  
- ✅ **Augmentation level K**: Control diversity of style transfer
- ✅ **Privacy preservation**: Only statistical information shared
- ✅ **AdaIN-based style transfer**: Following Equations (1-8)

### **Medical Domain Adaptation:**
- 🏥 **Client 1**: BUSI dataset (target domain)
- 🏥 **Client 2**: BUS-UCLM dataset (source domain)
- 🎯 **Goal**: Transfer BUS-UCLM to BUSI style for better domain generalization

## 📋 **Quick Start**

### **1. Install Dependencies**
```bash
pip install torch torchvision pillow pandas numpy tqdm
```

### **2. Generate CCST-Augmented Data**
```bash
python ccst_exact_replication.py
```

This runs the complete **CCST pipeline**:
- **Stage 1**: Extract domain styles from both datasets
- **Stage 2**: Create and broadcast style bank
- **Stage 3**: Apply Algorithm 1 for style transfer

### **3. Train with CCST Data**
```bash
python train_with_ccst_data.py --num-epochs 100 --batch-size 8
```

### **4. Test Implementation**
```bash
python test_ccst_exact_replication.py
```

## 🏗️ **Implementation Overview**

### **Core Files:**

```
ccst_exact_replication.py           # Main CCST pipeline implementation
├── CCSTStyleExtractor              # Extract single image & overall domain styles
├── CCSTStyleBank                   # Server-side style bank (federated simulation)
├── CCSTLocalStyleTransfer          # Algorithm 1 implementation
└── run_ccst_pipeline()             # Complete three-stage workflow

dataset/BioMedicalDataset/CCSTDataset.py    # Dataset classes for CCST data
├── CCSTAugmentedDataset            # Load CCST-augmented training data
└── CCSTFederatedDataset            # Simulate federated client datasets

train_with_ccst_data.py             # Training script using CCST methodology
test_ccst_exact_replication.py      # Test suite verifying implementation
```

## 📊 **CCST Methodology Breakdown**

### **Stage 1: Local Style Computation and Sharing**

Following **Section 3.2.1** of the paper:

#### **Overall Domain Style** (Equation 8):
```python
# Extract domain-level statistics from all images
all_features = Stack(F1, F2, ..., FM)  # All VGG features
domain_mean = μ(all_features)          # Channel-wise mean
domain_std = σ(all_features)           # Channel-wise std
```

#### **Single Image Style** (Equation 6):
```python
# Extract style from individual images
single_style = (μ(Fi), σ(Fi))         # Per-image statistics
style_bank = {S1, S2, ..., SJ}        # J random samples
```

### **Stage 2: Server-Side Style Bank Broadcasting**

Following **Section 3.2.2** of the paper:

```python
# Create global style bank
B_overall = {S_BUSI, S_BUS-UCLM}      # Overall domain styles
B_single = {S_BUSI_bank, S_BUS-UCLM_bank}  # Single image style banks

# Broadcast to all clients
broadcast(style_bank)
```

### **Stage 3: Local Style Transfer (Algorithm 1)**

**Exact implementation** of Algorithm 1 from the paper:

```python
def local_cross_client_style_transfer(I_Cn, B, K, T):
    D_Cn = []                          # Line 1: Augmented dataset
    
    for i in range(len(I_Cn)):         # Line 2: For each image
        S = random.choice(B, K)        # Line 3: Select K styles
        
        for S_Cn in S:                 # Line 4: For each selected style
            if Cn is current_client:   # Line 5-6: Keep original
                D_Cn.append(I_i)
            elif T is single_mode:     # Line 7-8: Single style transfer
                D_Cn.append(G(I_i, random.choice(S_Cn, 1)))
            elif T is overall_mode:    # Line 9-10: Overall style transfer
                D_Cn.append(G(I_i, S_Cn))
    
    return D_Cn                        # Line 11: Return augmented data
```

## 🎨 **AdaIN Style Transfer Implementation**

Following **Equations (1-5)** from the paper:

### **Feature Extraction** (Equation 1):
```python
Fc = Φ(Ic)  # Content features (BUS-UCLM)
Fs = Φ(Is)  # Style features (BUSI)
```

### **AdaIN Transformation** (Equation 2):
```python
AdaIN(Fc, Fs) = σ(Fs) * (Fc - μ(Fc)) / σ(Fc) + μ(Fs)
```

### **Image Generation** (Equation 3):
```python
Ic←s = Ψ(AdaIN(Fc, Fs))  # Decoder generates styled image
```

## 📈 **Configuration Options**

### **Style Types:**
- **`overall`**: Domain-level statistics (Equation 8)
- **`single`**: Individual image statistics (Equation 6)

### **Augmentation Levels (K) - Optimized for 2-Domain Setup:**

#### **Overall Domain Style:**
- **K=1**: Each image gets 1 cross-domain style transfer ✅ **OPTIMAL for 2 domains**
- **K=2**: Each image gets 2 transfers (1 useful + 1 duplicate) ⚠️ **Wasteful**
- **K=3**: Each image gets 3 transfers (1 useful + 2 duplicates) ❌ **Very wasteful**

#### **Single Image Style:**
- **K=1**: Each image gets 1 random single image style
- **K=2**: Each image gets 2 different single image styles  
- **K=3**: Each image gets 3 different single image styles ✅ **Good diversity**

### **Recommended Configurations for 2-Domain Setup:**
```bash
# Overall domain style (optimal for 2 domains)
python ccst_exact_replication.py --style-type overall --K 1    # 2x data

# Single image style (good diversity)  
python ccst_exact_replication.py --style-type single --K 1 --J 10   # 2x data
python ccst_exact_replication.py --style-type single --K 2 --J 10   # 3x data
python ccst_exact_replication.py --style-type single --K 3 --J 10   # 4x data
```

### **Why K=3 was confusing:**
- **Paper context**: Multiple hospitals (3+ domains) → K=3 selects 3 different domain styles
- **Our context**: 2 domains only → K=3 forces duplicate style applications
- **Solution**: Use K=1 for overall domain style, or use single image style with K=2/3

## 🔒 **Privacy Preservation**

Following **Section 4.4** of the paper:

### **What Gets Shared:**
```python
# Only statistical information (1024 numbers total)
shared_style = {
    'mean': [512 numbers],     # Channel-wise means
    'std': [512 numbers],      # Channel-wise standard deviations
    'type': 'overall_domain'
}
```

### **Privacy Guarantees:**
- ✅ **No actual images shared** between clients
- ✅ **High compression ratio**: 1000+ images → 1024 numbers  
- ✅ **Statistical aggregation** loses individual image information
- ✅ **VGG feature level** - not raw pixel information
- ✅ **Extremely difficult reconstruction** from style vectors alone

## 📊 **Expected Results**

### **Dataset Augmentation (2-Domain Setup):**

#### **Overall Domain Style (K=1) - Recommended:**
```
Original BUS-UCLM: ~500 images
CCST Augmented (K=1): ~1000 images (2x increase)

Breakdown:
- Original BUS-UCLM: ~500 images  
- BUS-UCLM → BUSI style: ~500 images
```

#### **Single Image Style (K=3) - For More Diversity:**
```
Original BUS-UCLM: ~500 images
CCST Augmented (K=3): ~2000 images (4x increase)

Breakdown:
- Original BUS-UCLM: ~500 images
- BUS-UCLM → BUSI single styles: ~1500 images (3 different single image styles per image)
```

### **Training Setup:**
- **Training**: Original BUSI + CCST-augmented BUS-UCLM (~1900 total)
- **Validation**: Original BUSI only (fair evaluation)  
- **Test**: Original BUSI only (fair evaluation)

### **Performance Gains:**
Following paper's methodology, expect:
- 📈 **Improved domain generalization**
- 📈 **Better performance on unseen domains**
- 📈 **More robust feature learning**
- 📈 **Reduced domain bias**

## 🧪 **Verification**

### **Algorithm 1 Verification:**
```bash
python test_ccst_exact_replication.py
```

**Checks:**
- ✅ Exact Algorithm 1 implementation
- ✅ Proper style extraction (Equations 6 & 8)
- ✅ Correct AdaIN application (Equation 2)
- ✅ Valid augmentation levels K
- ✅ Privacy-preserving style sharing
- ✅ Dataset structure consistency

## 🔬 **Research Context**

### **Paper's Contributions:**
1. **Cross-client style transfer** for federated domain generalization
2. **Two style sharing mechanisms** (single vs overall)
3. **Privacy-preserving** approach using only statistics
4. **Orthogonal to other DG methods** - can be combined
5. **State-of-the-art results** on PACS, Office-Home, Camelyon17

### **Your Adaptation:**
1. **Medical imaging focus** - ultrasound tumor segmentation
2. **Single institution** - simplified federated scenario
3. **Domain adaptation** - BUS-UCLM → BUSI style transfer
4. **Exact methodology** - following paper precisely
5. **Privacy properties preserved** - even though not strictly needed

## 📚 **Comparison with Original AdaIN Implementation**

| Aspect | Your Previous AdaIN | CCST Exact Replication |
|--------|-------------------|----------------------|
| **Methodology** | General AdaIN application | Paper's exact Algorithm 1 |
| **Style Types** | Overall domain only | Both single & overall |
| **Augmentation** | Fixed approach | Configurable K levels |
| **Privacy** | Implicit privacy | Explicit privacy analysis |
| **Federated Simulation** | Single-stage | Three-stage workflow |
| **Style Bank** | Simple statistics | Federated style broadcasting |
| **Paper Compliance** | AdaIN-inspired | CCST exact replication |

## 🎯 **Next Steps**

### **1. Run Main Experiment:**
```bash
python ccst_exact_replication.py
python train_with_ccst_data.py
```

### **2. Compare Results:**
- **Baseline**: Train only on original BUSI
- **CCST**: Train on BUSI + CCST-augmented BUS-UCLM
- **Metric**: Test performance on original BUSI

### **3. Ablation Studies:**
Test different configurations from Table 2:
- Overall vs Single style types
- Different K values (1, 2, 3)
- Different J values for single styles

### **4. Advanced Experiments:**
- **Orthogonality**: Combine with other DG methods
- **Style mixing**: Experiment with multiple style combinations
- **Real federated**: Extend to true multi-hospital scenario

## 🎉 **Summary**

This implementation provides:

✅ **Exact CCST replication** - Algorithm 1 line-by-line  
✅ **Paper methodology** - Three-stage workflow  
✅ **Privacy preservation** - Statistical sharing only  
✅ **Medical adaptation** - BUSI/BUS-UCLM domain gap  
✅ **Complete pipeline** - Data generation to training  
✅ **Verification suite** - Ensure correctness  

**You now have a faithful implementation of the CCST paper's methodology, adapted perfectly for your medical imaging domain adaptation task!** 🚀 