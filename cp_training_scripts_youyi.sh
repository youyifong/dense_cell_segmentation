#!/bin/bash

# Training   
echo "Stage1: Training"
python -m cellpose --train --dir "." --pretrained_model cyto2 --n_epochs 500 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --verbose --use_gpu

# Prediction
echo "Stage2: Prediction"
python -m cellpose --dir "test" --diameter 17 --save_png --verbose --net_avg --use_gpu --pretrained_model $(find . -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1) # grab the latest file

# Computing Average Precision
echo "Stage3: Calculating AP"
python ../../pred_processing.py