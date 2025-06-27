# Style Transfer for BUS Ultrasound Segmentation

This repository implements a style transfer approach to improve domain generalization between BUSI and BUS-UCLM datasets for breast ultrasound tumor segmentation.

## Overview

The goal is to address the domain generalization issue where:
- Training on BUSI and testing on BUSI works well
- Training on BUS-UCLM and testing on BUSI does not work well

**Solution**: Use CycleGAN to transfer BUS-UCLM images to BUSI style, then train MADGNet on the combined dataset.

## Pipeline Steps

1. **Style Transfer Training**: Train CycleGAN to learn the mapping between BUS-UCLM and BUSI styles
2. **Style Transfer Application**: Apply the trained model to transform BUS-UCLM images
3. **Combined Dataset Creation**: Merge BUSI and style-transferred BUS-UCLM images
4. **MADGNet Training**: Train MADGNet on the combined dataset
5. **Evaluation**: Test performance on both BUSI and BUS-UCLM

## Files Structure

```
├── style_transfer.py              # CycleGAN implementation for style transfer
├── apply_style_transfer.py        # Apply style transfer to BUS-UCLM dataset
├── train_with_style_transfer.py   # Complete pipeline orchestration
├── visualize_style_transfer.py    # Visualize style transfer results
├── dataset/BioMedicalDataset/
│   ├── BUSI/                      # Original BUSI dataset
│   ├── BUS-UCLM/                  # Original BUS-UCLM dataset
│   ├── BUS-UCLM-style-transferred/ # Style-transferred BUS-UCLM images
│   ├── BUSI-Combined/             # Combined dataset (BUSI + style-transferred)
│   └── BUSICombinedDataset.py     # Dataset loader for combined dataset
└── STYLE_TRANSFER_README.md       # This file
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

```bash
# Run the complete pipeline
python train_with_style_transfer.py --gpu 0 --madgnet_epochs 200

# Skip style transfer training if you already have a model
python train_with_style_transfer.py --skip_style_transfer --gpu 0 --madgnet_epochs 200
```

### Option 2: Run Steps Individually

#### Step 1: Train Style Transfer Model
```bash
python style_transfer.py
```

#### Step 2: Apply Style Transfer
```bash
python apply_style_transfer.py
```

#### Step 3: Train MADGNet on Combined Dataset
```bash
python IS2D_main.py --train_data_type BUSI-Combined --test_data_type BUSI --train --final_epoch 200
```

#### Step 4: Evaluate
```bash
# Evaluate on BUSI
python IS2D_main.py --train_data_type BUSI-Combined --test_data_type BUSI --final_epoch 200

# Evaluate on BUS-UCLM
python IS2D_main.py --train_data_type BUSI-Combined --test_data_type BUS-UCLM --final_epoch 200
```

## Configuration

### Style Transfer Parameters
- **Epochs**: 50 (default, adjustable via `--style_transfer_epochs`)
- **Image Size**: 352x352
- **Architecture**: CycleGAN with ResNet generator and PatchGAN discriminator

### MADGNet Parameters
- **Epochs**: 200 (default, adjustable via `--madgnet_epochs`)
- **Backbone**: ResNeSt50
- **Image Size**: 352x352
- **Batch Size**: 4

## Expected Results

### Hypothesis
Training on the combined dataset (BUSI + style-transferred BUS-UCLM) should improve performance on BUSI compared to training only on BUSI, because:

1. **Domain Alignment**: Style transfer aligns BUS-UCLM to BUSI's visual distribution
2. **Increased Diversity**: More training data with consistent style
3. **Better Generalization**: Model learns more robust features

### Evaluation Metrics
- **DSC** (Dice Similarity Coefficient)
- **IoU** (Intersection over Union)
- **Weighted F-Measure**
- **S-Measure**
- **E-Measure**
- **MAE** (Mean Absolute Error)

## Visualization

To visualize style transfer results:

```bash
python visualize_style_transfer.py
```

This will create a visualization showing:
- Original BUS-UCLM images
- Style-transferred images
- Target BUSI images for comparison

## Dataset Requirements

Ensure your datasets are organized as follows:

```
dataset/BioMedicalDataset/
├── BUSI/
│   ├── benign/
│   │   ├── image/
│   │   └── mask/
│   ├── malignant/
│   │   ├── image/
│   │   └── mask/
│   ├── train_frame.csv
│   ├── test_frame.csv
│   └── val_frame.csv
└── BUS-UCLM/
    ├── benign/
    │   ├── images/
    │   └── masks/
    ├── malignant/
    │   ├── images/
    │   └── masks/
    ├── train_frame.csv
    ├── test_frame.csv
    └── val_frame.csv
```

## Dependencies

```bash
pip install torch torchvision pandas numpy pillow tqdm matplotlib
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Style Transfer Not Converging**: Increase training epochs or adjust learning rate
3. **Dataset Not Found**: Check dataset paths and organization
4. **Model Loading Error**: Ensure style transfer model exists before applying

### Performance Tips

1. **GPU Memory**: Use gradient checkpointing for large models
2. **Training Speed**: Use mixed precision training
3. **Data Loading**: Increase `num_workers` for faster data loading
4. **Style Transfer**: Use pre-trained models if available

## Results Analysis

After running the pipeline, check:

1. **Style Transfer Quality**: Visualize transferred images
2. **Training Curves**: Monitor loss during training
3. **Test Results**: Compare metrics between different training approaches
4. **Generalization**: Test on both BUSI and BUS-UCLM

## Expected Improvements

The combined approach should show:
- Better performance on BUSI compared to BUSI-only training
- Improved generalization to BUS-UCLM
- More robust feature learning
- Better boundary detection due to style alignment

## Citation

If you use this style transfer approach, please cite:

```bibtex
@InProceedings{Nam_2024_CVPR,
    author    = {Nam, Ju-Hyeon and Syazwany, Nur Suriza and Kim, Su Jung and Lee, Sang-Chul},
    title     = {Modality-agnostic Domain Generalizable Medical Image Segmentation by Multi-Frequency in Multi-Scale Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11480-11491}
}
```

## Contact

For questions or issues, please refer to the main MADGNet repository or create an issue. 