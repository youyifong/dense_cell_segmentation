#!/bin/bash


# This script takes an integer parameter, 0 to 2, which indicates the cuda_id
# Training will be done using that particular gpu
# Suppose the cuda_id is 0, models will be saved under models0; prediction will use testimages0
# There need to be testimages0, testimages1, testimages2 folders under training
# Results will be appended to csi.txt, together with results from other gpus

seed=$1
pretrained=/home/yfong/deeplearning/dense_cell_segmentation/saved_models/pthmaskrcnn_trained_with_Kaggle2018nucleardata.pth
training_epochs=200


echo "Stage1: Training"
python ../../pthmrcnn_train.py --dir "." --pretrained_model $pretrained --n_epochs $training_epochs --gpu_id $seed


echo "Stage2: Prediction and compute AP"
# extra files mess up evaluation 
#rm testimages$1/*masks* 

# predict with newly trained model
python ../../pthmrcnn_eval.py --dir "testimages$seed" --the_model $(find models$seed -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)

    
echo "Done with $seed"
