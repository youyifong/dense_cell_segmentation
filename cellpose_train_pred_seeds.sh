#!/bin/bash

# run this script under dense_cell_segmentation/images/training: 
#bash ../../cellpose_train_pred_seeds.sh

pretrained="cyto"

if [ -f csi_$pretrained.txt ]; then
    rm csi_$pretrained.txt
fi

# filename
current_dir=$(pwd)
cd ../testmasks; filenames=(*.png); cd $current_dir
echo ${filenames[@]} | tr " " "," > csi_$pretrained.txt

for i in {0..2}
do
    echo "Seed=$i"
    bash ../../cellpose_train_pred.sh $i $pretrained &
    sleep 5 # so that not all processes will try to write to csi.txt at the same time
    
done

#mv csi_$pretrained.txt ../../APresults/csi_cp_448_norotate_flow3.txt
