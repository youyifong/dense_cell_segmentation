#!/bin/bash

# run this script under maskrcnn_train/images/training: 
# bash ../../mr_train_pred_eval.sh

rm csi.txt

# filename
current_dir=$(pwd)
cd ../testmasks; filenames=(*.png); cd $current_dir
echo ${filenames[@]} | tr " " "," > csi.txt

for i in {0..2}
do
    echo "Seed=$i"
    bash ../../mr_train_pred_eval_seed.sh $i &
    sleep 5 # so that not all processes will try to write to csi.txt at the same time   
done