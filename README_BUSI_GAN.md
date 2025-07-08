# BUSI Conditional GAN for Synthetic Medical Image Generation

This project implements a conditional Generative Adversarial Network (GAN) specifically designed for generating synthetic breast ultrasound images and their corresponding segmentation masks using the BUSI dataset.

## Features

- **Conditional Generation**: Generate benign or malignant tumor images based on class labels
- **Simultaneous Image-Mask Generation**: Produces both ultrasound images and segmentation masks
- **Customizable Output**: Specify exact numbers of benign/malignant samples to generate
- **Training Monitoring**: Saves sample outputs during training for quality assessment
- **Server-Ready**: Designed to run on GPU servers with SLURM support

## Project Structure

```
├── synthetic_busi_gan.py    # Main GAN implementation
├── run_busi_gan.py         # Easy execution script
├── requirements_gan.txt     # Python dependencies
└── README_BUSI_GAN.md      # This file
```

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_gan.txt
```

2. **Prepare BUSI Dataset**:
Ensure your BUSI dataset follows this structure:
```
dataset/BUSI/
├── benign/
│   ├── image/
│   │   ├── benign (1).png
│   │   ├── benign (2).png
│   │   └── ...
│   └── mask/
│       ├── benign (1)_mask.png
│       ├── benign (2)_mask.png
│       └── ...
└── malignant/
    ├── image/
    │   ├── malignant (1).png
    │   ├── malignant (2).png
    │   └── ...
    └── mask/
        ├── malignant (1)_mask.png
        ├── malignant (2)_mask.png
        └── ...
```

## Usage

### Quick Start (Recommended)

**1. Train the GAN**:
```bash
python run_busi_gan.py train --data_dir dataset/BUSI --epochs 200
```

**2. Generate Synthetic Data**:
```bash
python run_busi_gan.py generate --checkpoint checkpoints/busi_gan_final.pth
```

### Advanced Usage

**Training with Custom Parameters**:
```bash
python synthetic_busi_gan.py --mode train \
    --data_dir dataset/BUSI \
    --epochs 300 \
    --batch_size 16 \
    --lr 0.0001 \
    --checkpoint_dir my_checkpoints
```

**Generate Custom Number of Samples**:
```bash
python synthetic_busi_gan.py --mode generate \
    --checkpoint checkpoints/busi_gan_final.pth \
    --output_dir my_synthetic_data \
    --num_benign 175 \
    --num_malignant 89
```

## Command Line Arguments

### Training Mode
- `--data_dir`: Path to BUSI dataset (default: `dataset/BUSI`)
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Training batch size (default: 8)
- `--lr`: Learning rate (default: 0.0002)
- `--noise_dim`: Noise vector dimension (default: 100)
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)

### Generation Mode
- `--checkpoint`: Path to trained model checkpoint (required)
- `--output_dir`: Output directory for synthetic data (default: `synthetic_busi_dataset`)
- `--num_benign`: Number of benign samples to generate (default: 175)
- `--num_malignant`: Number of malignant samples to generate (default: 89)

## Output Structure

### During Training
- **Checkpoints**: Saved in `checkpoints/` directory every 20 epochs
- **Training Samples**: Preview images saved in `training_samples/` directory

### Synthetic Dataset
Generated synthetic data will be organized as:
```
synthetic_busi_dataset/
├── benign/
│   ├── image/
│   │   ├── synthetic_benign_001.png
│   │   └── ...
│   └── mask/
│       ├── synthetic_benign_001_mask.png
│       └── ...
├── malignant/
│   ├── image/
│   │   ├── synthetic_malignant_001.png
│   │   └── ...
│   └── mask/
│       ├── synthetic_malignant_001_mask.png
│       └── ...
└── synthetic_dataset.csv
```

The CSV file contains metadata for all generated samples:
- `image_path`: Relative path to image
- `mask_path`: Relative path to mask
- `class`: Class name (benign/malignant)
- `class_label`: Numeric class label (0/1)

## Architecture Details

### Generator
- **Input**: Noise vector (100D) + Class embedding
- **Output**: Ultrasound image (256×256, grayscale) + Segmentation mask (256×256, binary)
- **Architecture**: Shared encoder + Dual decoder heads

### Discriminator
- **Input**: Image + Mask + Class embedding
- **Output**: Real/Fake probability
- **Architecture**: Convolutional discriminator with class conditioning

## Training Tips

1. **Monitor Training**: Check `training_samples/` directory for generated samples during training
2. **GPU Usage**: Training is optimized for GPU. Ensure CUDA is available
3. **Batch Size**: Adjust based on GPU memory (reduce if out-of-memory errors occur)
4. **Training Time**: ~200 epochs should take 2-4 hours on a modern GPU
5. **Quality Assessment**: Generated samples improve significantly after epoch 50-100

## Server Deployment

For SLURM-based servers, create a job script:

```bash
#!/bin/bash
#SBATCH --job-name=busi_gan_train
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load python/3.8
module load cuda/11.1

# Install dependencies
pip install -r requirements_gan.txt

# Train the GAN
python run_busi_gan.py train --data_dir /path/to/BUSI --epochs 200

# Generate synthetic data
python run_busi_gan.py generate --checkpoint checkpoints/busi_gan_final.pth
```

## Troubleshooting

### Common Issues

1. **"No samples found"**: 
   - Check dataset directory structure
   - Ensure image and mask files follow naming convention

2. **Out of memory error**: 
   - Reduce batch size (`--batch_size 4`)
   - Use smaller image size (modify code if needed)

3. **Poor quality generation**:
   - Train for more epochs (300-500)
   - Adjust learning rate
   - Check training samples for progressive improvement

4. **CUDA errors**:
   - Ensure PyTorch CUDA version matches system CUDA
   - Use CPU mode if needed (will be slower)

### Quality Assessment

Monitor these during training:
- **Loss curves**: Both G_Loss and D_Loss should stabilize
- **Training samples**: Visual quality should improve over epochs
- **Mask quality**: Masks should be anatomically plausible

## Expected Results

After successful training:
- **Training time**: ~2-4 hours (200 epochs, GPU)
- **Generated images**: Realistic ultrasound appearance
- **Generated masks**: Anatomically plausible tumor shapes
- **Class distinction**: Clear differences between benign/malignant characteristics

## Citation

If you use this code in your research, please consider citing:

```bibtex
@misc{busi_conditional_gan,
  title={Conditional GAN for BUSI Dataset Synthetic Generation},
  author={Your Name},
  year={2024},
  howpublished={GitHub repository}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 