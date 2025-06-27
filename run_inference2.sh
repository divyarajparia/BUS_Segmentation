#!/bin/bash
#SBATCH --job-name=madgnet_infer
#SBATCH --partition=gpu        # from your sinfo output, nodes e22-09 are in “mixed”
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00

python3 IS2D_main.py \
  --num_workers 4 \
  --data_path dataset/BioMedicalDataset \
  --save_path model_weights\
  --train_data_type BUSIBUSUCLM \
  --test_data_type BUSUCLM \
  --final_epoch 5 \
  --train