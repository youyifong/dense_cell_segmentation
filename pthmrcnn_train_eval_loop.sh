#!/bin/bash

# first, run the following to refresh csi.txt with just filenames
#for i in {1..7}; do
#    cd training$i; rm csi.txt; current_dir=$(pwd); cd ../test_gtmasks; filenames=(*.png); cd $current_dir; echo ${filenames[@]} | tr " " "," > csi.txt; cd ..
#done

# then, run this script under maskrcnn_train/images as follows:
# nohup bash ../pthmrcnn_train_eval_loop.sh 0 > nohup0.out &
# sleep 5
# nohup bash ../pthmrcnn_train_eval_loop.sh 1 > nohup1.out &
# sleep 5
# nohup bash ../pthmrcnn_train_eval_loop.sh 2 > nohup2.out &


seed=$1
for i in {1..7}; do
    cd training$i
    echo "Working on training$i"
    bash ../../pthmrcnn_train_eval.sh $seed 
    cd ..
done


# finally, run the following to rename and mvoe csi.txt files
for i in {1..7}; do
    mv training$i/csi.txt ../APresults/csi_pthmrcnn_$i.txt
done
    


