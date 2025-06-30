#!/bin/bash
#SBATCH --job-name=madgnet_infer
#SBATCH --partition=gpu        # from your sinfo output, nodes e22-09 are in “mixed”
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00

# python simple_diffusion_busi.py \
#     --data_dir dataset/BioMedicalDataset/BUSI \
#     --mode train \
#     --num_epochs 50 \
#     --batch_size 8

python fix_busi_combined.py