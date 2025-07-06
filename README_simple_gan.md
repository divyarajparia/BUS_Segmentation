# Simple GAN for Synthetic Medical Image Generation

This implements a **straightforward GAN approach** for generating synthetic breast ultrasound images, providing a fair comparison with your existing CycleGAN/style transfer method.

## Why Simple GAN?

Your current approaches:
- **Complex**: Joint Diffusion Model (DDPM) - 1000 timesteps, complex sampling, numerical stability issues
- **Style Transfer**: CycleGAN - domain adaptation between BUS-UCLM and BUSI
- **Simple**: **This GAN** - direct synthetic generation, easy to understand and debug

## Architecture Overview

### Generator
- **Input**: Random noise (100D) + class label (benign/malignant)
- **Output**: Synthetic image + corresponding mask
- **Architecture**: 
  - Fully connected layer → Convolutional upsampling blocks
  - Separate heads for image and mask generation
  - Image output: Tanh activation ([-1, 1])
  - Mask output: Sigmoid activation ([0, 1])

### Discriminator
- **Input**: Image + class label
- **Output**: Real/fake classification
- **Architecture**: Standard CNN with downsampling blocks

## Key Advantages

1. **Simplicity**: Much simpler than diffusion models
2. **Speed**: Fast training and generation
3. **Interpretability**: Easy to understand and debug
4. **Fair Comparison**: Direct comparison with CycleGAN approach
5. **Joint Generation**: Produces both images and masks simultaneously

## Usage

### Training
```bash
# Train the GAN
python simple_gan_synthetic.py --mode train --data_dir dataset/BUSI --epochs 100

# Monitor training progress
# Checkpoints saved every 10 epochs: simple_gan_epoch_10.pth, simple_gan_epoch_20.pth, etc.
```

### Generation
```bash
# Generate synthetic dataset (175 benign + 89 malignant)
python simple_gan_synthetic.py --mode generate --checkpoint simple_gan_epoch_100.pth --output_dir synthetic_gan_output

# Custom generation numbers
python simple_gan_synthetic.py --mode generate --checkpoint simple_gan_epoch_100.pth --num_benign 200 --num_malignant 100
```

## Comparison Framework

Now you can compare three approaches:

### 1. Style Transfer (CycleGAN)
- **Method**: Domain adaptation BUS-UCLM → BUSI style
- **Output**: 264 style-transferred images
- **Training**: Train on BUSI + style-transferred BUS-UCLM

### 2. Simple GAN (This approach)
- **Method**: Direct synthetic generation
- **Output**: 264 synthetic images + masks
- **Training**: Train on BUSI + synthetic GAN images

### 3. Complex Diffusion (Current)
- **Method**: Joint diffusion model
- **Output**: 264 synthetic images + masks
- **Training**: Train on BUSI + synthetic diffusion images

## Expected Results

The simple GAN should provide:
- **Faster training** than diffusion models
- **More stable generation** (no numerical instability issues)
- **Direct comparison** with style transfer quality
- **Easier debugging** and parameter tuning

## File Structure

```
synthetic_gan_output/
├── benign/
│   ├── images/
│   │   ├── synthetic_benign_001.png
│   │   ├── synthetic_benign_002.png
│   │   └── ...
│   └── masks/
│       ├── synthetic_benign_001_mask.png
│       ├── synthetic_benign_002_mask.png
│       └── ...
└── malignant/
    ├── images/
    │   ├── synthetic_malignant_001.png
    │   ├── synthetic_malignant_002.png
    │   └── ...
    └── masks/
        ├── synthetic_malignant_001_mask.png
        ├── synthetic_malignant_002_mask.png
        └── ...
```

## Training Tips

1. **Batch Size**: Start with 4, increase if you have more GPU memory
2. **Epochs**: 100 epochs should be sufficient for good results
3. **Learning Rate**: 0.0002 works well for most cases
4. **Loss Balance**: L1 loss weight of 10 helps with mask quality

## Integration with Your Workflow

You can easily integrate this into your existing pipeline:

```bash
# 1. Train simple GAN
python simple_gan_synthetic.py --mode train --data_dir dataset/BUSI

# 2. Generate synthetic dataset
python simple_gan_synthetic.py --mode generate --checkpoint simple_gan_epoch_100.pth

# 3. Use generated images in your segmentation training
# (Same as you do with style transfer results)
```

This provides a clean, simple alternative to your complex diffusion approach while maintaining the same output format and integration compatibility. 