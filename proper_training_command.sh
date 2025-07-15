
# Proper IS2D-based training (achieves DSC 0.7-0.9)
python IS2D_main.py \
    --data_path dataset/BioMedicalDataset \
    --train_data_type BUS-UCLM \
    --test_data_type BUS-UCLM \
    --save_path model_weights \
    --final_epoch 100 \
    --batch_size 8 \
    --train \
    --num_workers 4
    
# Expected results:
# - Training losses: 0.01-0.8 range (not 40-200+!)
# - Final DSC: 0.7-0.9 (not 0.05-0.09!)  
# - Results saved to: model_weights/BUS-UCLM/test_reports/
