#!/bin/bash

# This script takes an integer parameter, 0 to 2, which indicates the cuda_id
# Training will be done using that particular gpu
# Suppose the cuda_id is 0, models will be saved under models0; prediction will use testimages0
# Results will be appended to csi.txt, together with results from other gpus

echo "Stage1: Training"
# --diam_mean is the mean diameter to resize cells to during training 
#       If starting from pretrained models, it cannot be changed from 30.0, but the value is saved to the model and used during prediction
#       In cp2.0, the default for diam_mean is 17 for nuclear, 30 for cyto
#python -m cellpose --train --dir "." --patch_size 56 --pretrained_model cyto --n_epochs 500 --img_filter _img --mask_filter _masks --verbose --use_gpu --train_seed $1 --cuda_id $1

# --pretrained_model $() finds the latest model under models; to train with cyto2, replace $() with cyto2
# --diameter 0 is key. 
#       In cellpose 2.0, if --diam_mean 17 is added during training (this has no impact if training starts from pretrained models), then it is essential to add --diameter 17 during prediction. 
#       This parameter does not impact training according to help
echo "Stage2: Prediction and compute AP"
rm testimages$1/*.npy testimages$1/*masks* # extra files mess up evaluation 
python -m cellpose --dir "testimages$1" --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --pretrained_model $(find models$1 -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)    --save_png --verbose --use_gpu
python ../../pred_processing.py $1 |& tee -a csi.txt
rm testimages$1/*.npy testimages$1/*masks* # extra files mess up evaluation 
    
# predict with cyto   
#python -m cellpose --dir "test" --pretrained_model cyto --chan 1 --chan2 0 --diameter 0.0 --flow_threshold 0.4 --cellprob_threshold 0 --net_avg --save_png --verbose --use_gpu
    
echo "Done with $1"
