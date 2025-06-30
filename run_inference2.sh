#!/bin/bash
#SBATCH --job-name=madgnet_infer
#SBATCH --partition=gpu        # from your sinfo output, nodes e22-09 are in “mixed”
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00

python apply_style_transfer.py