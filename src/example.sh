#!/usr/bin/env bash

### Model names ###
# BERTTrainer: Simply fine-tuned BERT
# BERT1E: BERT+[CLS]
# BERT1F: BERT+SimMatrix
###################

# Fine-tuning
python ./fine-tune_bert.py --out_dir ../model/ --model_type BERT1F --pooling mean --train_epoch 100 --early_stop 5 --margin 1.0 --lr 3e-05 --lossfnc TripletMarginLoss --ft_bert

# Alignment with Constrained tree-edit distance & evaluation
python ./main.py --out_dir ../out/ --model_dir ../model/ --model_name BERT1F_TripletMarginLoss_margin-1.0_lr-3e-05_mean_100_ft-bert-base-uncased.pkl --pooling mean --null_thresh 0.8
# For outputting alignment results, flag "--decode"

# Alignment with naive threshold
python ./baseline_wo_ted.py --out_dir ../out/ --model_dir ../model/ --model_name BERT1F_TripletMarginLoss_margin-1.0_lr-3e-05_mean_100_ft-bert-base-uncased.pkl --pooling mean --null_thresh 0.6

