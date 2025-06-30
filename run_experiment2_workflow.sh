#!/bin/bash

# Complete Workflow for Experiment 2: BUSI + Synthetic BUSI
# ========================================================

echo "🎯 EXPERIMENT 2: BUSI + Synthetic BUSI Workflow"
echo "==============================================="

# Step 1: Generate synthetic images
echo "📊 Step 1: Generating synthetic BUSI images..."
echo "Target: 264 samples (175 benign + 89 malignant) to match style-transferred count"

sbatch run_synthetic_generation.job
SYNTHETIC_JOB_ID=$(squeue -u $USER -h -o "%i" | tail -1)
echo "🚀 Submitted synthetic generation job: $SYNTHETIC_JOB_ID"

# Wait for synthetic generation to complete
echo "⏳ Waiting for synthetic generation to complete..."
while squeue -j $SYNTHETIC_JOB_ID &>/dev/null; do
    echo "   Still generating synthetic images... ($(date))"
    sleep 60
done

echo "✅ Synthetic generation completed!"

# Step 2: Create combined dataset
echo "📊 Step 2: Creating BUSI-Synthetic-Combined dataset..."
python create_busi_synthetic_combined.py

if [ $? -eq 0 ]; then
    echo "✅ BUSI-Synthetic-Combined dataset created successfully!"
else
    echo "❌ Failed to create combined dataset!"
    exit 1
fi

# Step 3: Train the model
echo "📊 Step 3: Training MADGNet on BUSI-Synthetic-Combined..."
sbatch train_busi_synthetic.job
TRAIN_JOB_ID=$(squeue -u $USER -h -o "%i" | tail -1)
echo "🚀 Submitted training job: $TRAIN_JOB_ID"

echo ""
echo "🎉 EXPERIMENT 2 WORKFLOW INITIATED!"
echo "=================================="
echo "📊 Synthetic generation: Check logs/synthetic_gen_*.out"
echo "🏋️ Training job: Check logs/busi_synthetic_*.out"
echo "📈 Results will be saved in: model_weights/BUSI-Synthetic-Combined/"
echo ""
echo "To monitor progress:"
echo "  squeue -u $USER"
echo "  tail -f logs/busi_synthetic_*.out"
echo ""
echo "Expected completion: ~6-8 hours for training" 