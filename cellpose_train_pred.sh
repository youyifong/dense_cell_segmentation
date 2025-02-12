#!/bin/bash

# Run this script in the same folder as the training images, e.g. images/training1
# There should be testimages0, testimages1, testimages2 folders in this directory
# It takes an integer parameter, 0 to 2, which indicates the gpu_device and training will be done using that particular gpu
# Suppose the gpu_device is 0, models will be saved under models0; prediction will use testimages0
# Results will be appended to csi_$pretrained.txt, together with results from other gpus

seed=$1
#actualseed=$(($seed + 0)) # change 0 to 3 when need more than 3 replicates. Maybe in the future we should make new folders for new seeds

pretrained=$2 # cyto cyto2 tissuenet livecell None
training_epochs=500


###############################################################################
echo "Stage1: Training"

if [ `ls -1 *.png 2>/dev/null | wc -l ` -gt 0 ];
then
    # --diam_mean is the mean diameter to resize cells to during training 
    #       If starting from pretrained models, it cannot be changed from 30.0 (see models.py), but the value is saved to the model and used during prediction
    #       In cp2.0, the default for diam_mean is 17 for nuclear, 30 for cyto
    # --pretrained_model None for no pretrained model
    
    # default. note that patch_size default is not 224 in the forked cellpose repo, thus needs to be specified
    python -m cellpose --train --dir "." --patch_size 224 --pretrained_model $pretrained --n_epochs $training_epochs --img_filter _img --mask_filter _masks --verbose --use_gpu --train_seed $seed --gpu_device $seed
    # optimzied: --patch_size 448 --no_rotate 
    #python -m cellpose --train --dir "." --patch_size 448 --no_rotate --pretrained_model $pretrained --n_epochs $training_epochs --img_filter _img --mask_filter _masks --verbose --use_gpu --train_seed $seed --gpu_device $seed
    echo "do nothing"
else
    echo "no png files, skip training"
fi


###############################################################################
echo "Stage2: Prediction and compute AP"

# rm extra files that can mess up evaluation 
if [ `ls -1 testimages$seed/*masks* 2>/dev/null | wc -l ` -gt 0 ];
then
    rm testimages$seed/*masks* 
fi
if [ `ls -1 testimages$seed/*outlines* 2>/dev/null | wc -l ` -gt 0 ];
then
    rm testimages$seed/*outlines* 
fi


#### prediction
if [ `ls -1 *.png 2>/dev/null | wc -l ` -gt 0 ];
then
    # predict with newly trained model
    # --pretrained_model $() finds the latest model under models; to train with cyto2, replace $() with cyto2, tissuenet, livecell, or none
    # --diameter 0 is key. 
    #       In cellpose 2.0, if --diam_mean 17 is added during training (this has no impact if training starts from pretrained models), then it is essential to add --diameter 17 during prediction. 
    #       This parameter does not impact training according to help
    python -m cellpose --dir "testimages$seed" --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0 --pretrained_model $(find models$seed/cellpose* -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2- | tail -n 1)  --verbose --use_gpu --save_png --no_npy --savedir "testimages$seed/tmp"
else
    # predict with pretrained_model
    python -m cellpose --dir "testimages$seed" --flow_threshold 0.4 --cellprob_threshold 0 --diameter 0 --pretrained_model $pretrained  --verbose --use_gpu --save_png  --no_npy --savedir tmp
fi

python -m tsp checkprediction --metric csi  --predfolder testimages$seed/tmp --gtfolder ../test_gtmasks |& tee -a csi_$pretrained.txt
python -m tsp checkprediction --metric ari  --predfolder testimages$seed/tmp --gtfolder ../test_gtmasks |& tee -a ari_$pretrained.txt
python -m tsp checkprediction --metric dice --predfolder testimages$seed/tmp --gtfolder ../test_gtmasks |& tee -a dice_$pretrained.txt


## extra files mess up evaluation 
#if [ `ls -1 testimages$seed/*masks* 2>/dev/null | wc -l ` -gt 0 ];
#then
#    rm testimages$seed/*masks* 
#fi
    
echo "Done with seed $seed"


#python ../../pred_processing.py 0 csi
#python ../../pred_processing.py 0 tpfpfn |& tee -a tpfpfn_cyto_7.txt
#python ../../pred_processing.py 0 coloring


# predicting with cyto_train7
#python -m cellpose --dir "testimages2" --diameter 0  --pretrained_model /fh/fast/fong_y/cellpose_trained_models/cellpose_cyto_train7_seed0 --verbose --use_gpu --no_npy --save_png --chan 2 --chan2 0
#python -m syotil --action checkprediction --name testimage2 --metric csi |& tee -a csi.txt
# predicting with cyto
#python -m cellpose --dir "testimages1" --diameter 0  --pretrained_model cyto --verbose --use_gpu --no_npy --save_png
#python -m syotil --action checkprediction --name testimage1 --metric csi |& tee -a csi.txt
