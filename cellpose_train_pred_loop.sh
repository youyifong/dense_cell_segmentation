#!/bin/bash

# run this script under images/
# rm nohup0.out; nohup bash ../cellpose_train_pred_loop.sh 0 > nohup0.out &
# rm nohup1.out; nohup bash ../cellpose_train_pred_loop.sh 1 > nohup1.out &
# rm nohup2.out; nohup bash ../cellpose_train_pred_loop.sh 2 > nohup2.out &
# note that csi.txt and bias.txt thus generated do not have header

seed=$1

for i in {0..7}
do
    echo "Working on training$i"
    cd training$i
    bash ../../cellpose_train_pred.sh $seed
    cd ..
    sleep 5 # so that not all processes will try to write to csi.txt at the same time    
done



# the following also makes sure csi.txt and bias.txt are removed from each folder

#pretrained="cyto" # cyto cyto2 tissuenet livecell None
#for i in {0..7}
#do
#    mv training$i/csi_$pretrained.txt ../APresults/csi_cp_$pretrained\_$i.txt    
#    mv training$i/bias_$pretrained.txt ../APresults/bias_cp_$pretrained\_$i.txt    
#    
##    rm training$i/csi_$pretrained.txt
##    rm training$i/bias_$pretrained.txt
#done
