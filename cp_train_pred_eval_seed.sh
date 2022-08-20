#!/bin/bash

seed=$1

# This script is to be run in the same folder as the training images
# There should be testimages0, testimages1, testimages2 folders in this directory
# It takes an integer parameter, 0 to 2, which indicates the cuda_id
# Training will be done using that particular gpu
# Suppose the cuda_id is 0, models will be saved under models0; prediction will use testimages0
# Results will be appended to csi.txt, together with results from other gpus


echo "Stage1: Training"
# --diam_mean is the mean diameter to resize cells to during training 
#       If starting from pretrained models, it cannot be changed from 30.0, but the value is saved to the model and used during prediction
#       In cp2.0, the default for diam_mean is 17 for nuclear, 30 for cyto
# --pretrained_model None for no pretrained model
python -m cellpose --train --dir "." --patch_size 448 --no_rotate --pretrained_model cyto --n_epochs 500 --img_filter _img --mask_filter _masks --verbose --use_gpu --train_seed $seed --cuda_id $seed

# --pretrained_model $() finds the latest model under models; to train with cyto2, replace $() with cyto2
# --diameter 0 is key. 
#       In cellpose 2.0, if --diam_mean 17 is added during training (this has no impact if training starts from pretrained models), then it is essential to add --diameter 17 during prediction. 
#       This parameter does not impact training according to help
echo "Stage2: Prediction and compute AP"
# extra files mess up evaluation 
rm testimages$seed/*.npy testimages$seed/*masks* 
# predict with newly trained model
python -m cellpose --dir "testimages$seed" --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --pretrained_model $(find models$seed -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)    --save_png --verbose --use_gpu --no_npy
# predict with pretrained_model   
#python -m cellpose --dir "testimages$seed" --pretrained_model cyto --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --save_png --verbose --use_gpu
# write csi and bias results to two files
python -m syotil --action checkprediction --name testimage$seed --metric csi  |& tee -a csi.txt
python -m syotil --action checkprediction --name testimage$seed --metric bias |& tee -a bias.txt
# extra files mess up evaluation 
rm testimages$seed/*.npy testimages$seed/*masks* 
    
    
echo "Done with $seed"


#python ../../pred_processing.py 0 csi
#python ../../pred_processing.py 0 tpfpfn |& tee -a tpfpfn_cyto_7.txt
#python ../../pred_processing.py 0 coloring


# predicting with cyto_train7
#python -m cellpose --dir "testimages2" --diameter 0  --pretrained_model /fh/fast/fong_y/cellpose_trained_models/cellpose_cyto_train7_seed0 --verbose --use_gpu --no_npy --save_png --chan 2 --chan2 0
#python -m syotil --action checkprediction --name testimage2 --metric csi |& tee -a csi.txt
# predicting with cyto
#python -m cellpose --dir "testimages1" --diameter 0  --pretrained_model cyto --verbose --use_gpu --no_npy --save_png
#python -m syotil --action checkprediction --name testimage1 --metric csi |& tee -a csi.txt

