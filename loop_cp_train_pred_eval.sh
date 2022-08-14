#!/bin/bash

# run this script under cell_segmentation/images: 
# bash ../loop_cp_train_pred_eval.sh 0 # 0 can be 1 or 2
# note that csi.txt and bias.txt thus generated do not have header

echo "Seed=$1"
for i in {1..7}
do
    cd training$i
    echo "Working on training$i"
    bash ../../cp_train_pred_eval_seed.sh $1
    cd ..
    sleep 5 # so that not all processes will try to write to csi.txt at the same time
    
done



## the following also makes sure csi.txt and bias.txt are removed from each folder
## change cyto to cyto2 or as needed
#for i in {1..7}
#do
#    mv training$i/csi.txt training/csi_none_$i.txt    
#    mv training$i/bias.txt training/bias_none_$i.txt    
#done
