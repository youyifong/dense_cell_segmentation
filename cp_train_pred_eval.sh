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
    python -m cellpose --train --dir "." --pretrained_model cyto --n_epochs 500 --img_filter _img --mask_filter _masks --verbose --use_gpu --train_seed $i
    
    # find the latest model under models; to train with cyto2, replace $() with cyto2
    # --diam_mean is the mean diameter to resize cells to during training 
    #       If starting from pretrained models, it cannot be changed from 30.0, but the value is saved to the model and used during prediction
    #       In cp2.0, the default for diam_mean is 17 for nuclear, 30 for cyto
    # --diameter 0 is key. 
    #       In cellpose 2.0, if --diam_mean 17 is added during training, then it is essential to add --diameter 17 during prediction. 
    #       This parameter does not impact training according to help
    echo "Stage2: Prediction and compute AP"
    python -m cellpose --dir "test"  --save_png --verbose --use_gpu --diameter 0  --pretrained_model $(find models -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1); python ../../pred_processing.py |& tee -a csi.txt
    
done
