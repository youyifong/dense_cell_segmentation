#!/bin/bash

# run this script under cellpose_train_immune
# bash cellpose_train_pred_zoo.sh

# available models as listed in models.py: 'cyto','nuclei','tissuenet','livecell', 'cyto2', 'general', 'CP', 'CPx', 'TN1', 'TN2', 'TN3', 'LC1', 'LC2', 'LC3', 'LC4'


# write filename
current_dir=$(pwd)
cd images/testmasks; filenames=(*.png); cd $current_dir # filenames are in the same order as python sort()
echo "model," > APresults/csi_cp_model_zoo.txt # this wipes the previous contents of the file clean
echo ${filenames[@]} | tr " " "," >> APresults/csi_cp_model_zoo.txt # append test file names

for i in cyto cyto2 nuclei tissuenet livecell
do
    echo -n "$i," >> APresults/csi_cp_model_zoo.txt
    python -m cellpose --dir "images/training/testimages" --pretrained_model $i --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --save_png --verbose --use_gpu
    python -m syotil checkprediction --name images/training/testimages --metric csi  |& tee -a APresults/csi_cp_model_zoo.txt
done


# TNx model
for i in {1..3}
do
    echo -n "TN$i," >> APresults/csi_cp_model_zoo.txt
    python -m cellpose --dir "images/training/testimages" --pretrained_model TN$i --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --save_png --verbose --use_gpu
    python -m syotil checkprediction --name images/training/testimages --metric csi  |& tee -a APresults/csi_cp_model_zoo.txt
done

# LCx models
for i in {1..4}
do
    echo -n "LC$i," >> APresults/csi_cp_model_zoo.txt
    python -m cellpose --dir "images/training/testimages" --pretrained_model LC$i --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0  --save_png --verbose --use_gpu
    python -m syotil checkprediction --name images/training/testimages --metric csi  |& tee -a APresults/csi_cp_model_zoo.txt
done
