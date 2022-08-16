#!/bin/bash


# This script takes an integer parameter, 0 to 2, which indicates the cuda_id
# Training will be done using that particular gpu
# Suppose the cuda_id is 0, models will be saved under models0; prediction will use testimages0
# There need to be testimages0, testimages1, testimages2 folders under training
# Results will be appended to csi.txt, together with results from other gpus

echo "Stage1: Training"
python ../../maskrcnn_train.py --dir "." --pretrained_model coco --n_epochs 500 --batch_size 8 --train_seed $1 --patch_size 448 --min_box_size 10 --box_detections_per_img 100 --cuda_id $1

echo "Stage2: Prediction and compute AP"
# extra files mess up evaluation 
rm testimages$1/*masks* 

# predict with newly trained model
python ../../maskrcnn_pred.py --dir "testimages$1" --min_score 0.1 --mask_threshold 0.5 --pretrained_model $(find models$1 -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)

# write csi and bias results to two files
python ../../pred_processing.py $1 AP |& tee -a csi.txt # in pred_processing.py, change '/*_test_img_cp_masks.png' to '/*_test_img_mr_masks.png'
python ../../pred_processing.py $1 bias |& tee -a bias.txt

# extra files mess up evaluation 
rm testimages$1/*masks* 
    
echo "Done with $1"
