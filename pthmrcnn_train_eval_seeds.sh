#!/bin/bash

# run this script under maskrcnn_train/images/training1: 
# bash ../../xxx.sh

rm csi.txt

# filename
current_dir=$(pwd)
cd ../test_gtmasks; filenames=(*.png); cd $current_dir
echo ${filenames[@]} | tr " " "," > csi.txt

for i in {0..2}
do
    echo "Seed=$i"
    sleep $(( 2*i )) # so that not all processes will try to write to csi.txt at the same time   
    bash ../../pthmrcnn_train_eval.sh $i &
done
