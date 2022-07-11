#!/bin/bash

rm csi.txt

for i in {0..3}
do
    echo "Seed=$i"
    
    # Training
    echo "Stage1: Training"
    python -m cellpose --train --dir "." --pretrained_model cyto2 --n_epochs 500 --img_filter _img --mask_filter _masks --chan 1 --chan2 0 --verbose --use_gpu --train_seed $i
    
    # Prediction
    echo "Stage2: Prediction"
    # in cellpose 2.0, essential to add --diameter 17 during prediction. in cellpose 0.7, there is no need to add that
	# 17 is diam_mean, a model parameter, for nuclear, 30 is for cyto
    python -m cellpose --dir "test" --diameter 17 --chan 1 --chan2 0 --save_png --verbose --use_gpu --pretrained_model $(find models -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1) # grab the latest file
    
        # predicting with cyto2
        #python -m cellpose --dir "test" --diameter 17 --pretrained_model cyto2 --chan 1 -chan2 0 --net_avg --save_png --verbose --use_gpu
    
    # Computing Average Precision
    echo "Stage3: Calculating AP"
    python ../../pred_processing.py >> csi.txt
done

# train1: trained with Pos 8 (CD8_traing), 500 epochs
#1-99: 0.6
#0-100: 0.66
# based on this we decide to change 1-99 to 0-100 in transforms.py
#def normalize99(Y, lower=0,upper=100):
# if no_norm is added to both training and prediction, csi is 0


# Experiments for tuning hyper-parameters
# 1) Pretrained cyto
# Prediction
    # python -m cellpose --dir "test" --pretrained_model cyto --chan 1 --chan2 0 --diameter 0.0 --flow_threshold 0.4 --cellprob_threshold 0 --net_avg --save_png --verbose --use_gpu
    # python ../../pred_processing.py >> cyto_diam_est.txt

# 2) Trained model on train4
# Training
    # python -m cellpose --train --dir "." --pretrained_model cyto --n_epochs 500 --img_filter _img --mask_filter _masks --chan 1 --chan2 0 --verbose --use_gpu --train_seed 3
# Prediction
    # python -m cellpose --dir "test" --pretrained_model tuning/models/cellpose_residual_on_style_on_concatenation_off_train4_seed3_2022_06_27_09_18_42.784391 --chan 1 --chan2 0 --diameter 0.0 --flow_threshold 0.5 --cellprob_threshold -2 --save_png --verbose --use_gpu
    # python ../../pred_processing.py >> train4_cyto_diam_est_flow05_cp-2.txt

#cellpose_residual_on_style_on_concatenation_off_train4_seed0_2022_06_27_07_35_00.349428
#cellpose_residual_on_style_on_concatenation_off_train4_seed1_2022_06_27_07_52_30.989385
#cellpose_residual_on_style_on_concatenation_off_train4_seed2_2022_06_27_09_05_59.721055
#cellpose_residual_on_style_on_concatenation_off_train4_seed3_2022_06_27_09_18_42.784391
