#!/bin/bash

# Experiments for tuning hyper-parameters
# 1) Pretrained cyto
python -m cellpose --dir "test" --pretrained_model cyto --chan 1 --chan2 0 --diameter 0.0 --flow_threshold 0.4 --cellprob_threshold 0 --net_avg --save_png --verbose --use_gpu
python ../../pred_processing.py >> cyto_diam_est.txt



# 2) Trained model on train4
#python -m cellpose --dir "test" --pretrained_model tuning/models/cellpose_residual_on_style_on_concatenation_off_train4_seed3_2022_06_27_09_18_42.784391 --chan 1 --chan2 0 --diameter 0.0 --flow_threshold 0.5 --cellprob_threshold -2 --save_png --verbose --use_gpu
#python ../../pred_processing.py >> train4_cyto_diam_est_flow05_cp-2.txt

#cellpose_residual_on_style_on_concatenation_off_train4_seed0_2022_06_27_07_35_00.349428
#cellpose_residual_on_style_on_concatenation_off_train4_seed1_2022_06_27_07_52_30.989385
#cellpose_residual_on_style_on_concatenation_off_train4_seed2_2022_06_27_09_05_59.721055
#cellpose_residual_on_style_on_concatenation_off_train4_seed3_2022_06_27_09_18_42.784391
