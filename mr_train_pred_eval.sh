#!/bin/bash

rm csi.txt

# filename
current_dir=$(pwd)
cd ../test; filenames=(*.png); cd $current_dir
echo ${filenames[@]} | tr " " "," > csi.txt

for i in {0..3}
do
    echo "Seed=$i"
    
    echo "Stage1: Training"
    python ../../maskrcnn_train.py --dir "." --pretrained_model coco --n_epochs 500 --batch_size 1 --train_seed $i
    
    echo "Stage2: Prediction and compute AP"
    python ../../maskrcnn_pred.py --dir "test" --min_score 0.3 --mask_threshold 0.5 --pretrained_model $(find models -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1); python ../../pred_processing.py |& tee -a csi.txt
    
done