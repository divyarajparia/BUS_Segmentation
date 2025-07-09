#!/bin/bash
#SBATCH --job-name=busi_gan_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/busi_gan_%j.out
#SBATCH --error=logs/busi_gan_%j.err

echo "Starting BUSI GAN training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create logs directory
mkdir -p logs

# Train the GAN with explicit arguments
python synthetic_busi_gan.py \
    --mode train \
    --data_dir dataset/BioMedicalDataset/BUSI \
    --epochs 100 \
    --batch_size 8 \
    --checkpoint_dir checkpoints

echo "Training completed!"