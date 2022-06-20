#!/bin/bash

for i in {0..3}
do
    echo "Seed=$i"
    
    # Training
    echo "Stage1: Training"
    # if --use_gpu, results are not reproducible because torch.manual_seed not set
    python -m cellpose --train --dir "." --pretrained_model cyto2 --n_epochs 500 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --verbose --use_gpu --train_seed $i
    
    # Prediction
    echo "Stage2: Prediction"
    # in cellpose 2.0, essential to add --diameter 17 during prediction. in cellpose 0.7, there is no need to add that
    python -m cellpose --dir "test" --diameter 17 --save_png --verbose --net_avg --use_gpu --pretrained_model $(find . -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1) # grab the latest file
    
        # predicting with cyto2
        #python -m cellpose --use_gpu --dir "test" --diameter 17 --pretrained_model cyto2 --save_png --verbose --use_gpu
    
    # Computing Average Precision
    echo "Stage3: Calculating AP"
    python ../../pred_processing.py
done

# train1: trained with Pos 8 (CD8_traing), 500 epochs
#1-99: 0.6
#0-100: 0.66
# based on this we decide to change 1-99 to 0-100 in transforms.py
#def normalize99(Y, lower=0,upper=100):
# if no_norm is added to both training and prediction, csi is 0
