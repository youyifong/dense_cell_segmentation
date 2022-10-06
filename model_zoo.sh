#!/bin/bash

# run this script under cellpose_train_immune/images/training: 
# bash ../../cp_train_pred_eval.sh

rm csi.txt

# filename
current_dir=$(pwd)
cd ../testmasks; filenames=(*.png); cd $current_dir
echo ${filenames[@]} | tr " " "," > csi.txt



seed=0

for i in {1..3}
do
    python -m cellpose --dir "testimages$seed" --pretrained_model TN$i --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --save_png --verbose --use_gpu
    python -m syotil --action checkprediction --name testimages$seed --metric csi  |& tee -a csi.txt
done


for i in {1..4}
do
    python -m cellpose --dir "testimages$seed" --pretrained_model LC$i --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --save_png --verbose --use_gpu
    python -m syotil --action checkprediction --name testimages$seed --metric csi  |& tee -a csi.txt
done
