# Summary

- The AP results are saved in csv files under the APresults folder. Tabular and graphical summary of the results are made with summary.R
- images/training_resized is 2x, images/training_resized is 1.54x, both are 2-dimensional H x W. images/training_resized_3chan is H x W x 3. 
- See installation notes.txt for more info on how to run code.


# Cellpose training 

Cellpose training and prediction are implemented in shell scripts, run on Linux. All training is done for 500 epochs. Cellpose training is repeated three times with seeds 0,1,2. 

- Results for pretrained models are obtained with cellpose_pred_model_zoo.sh. 
- Results for fine-tuned models and optimizing Cellpose are obtained with cellpose_train_pred_loop.sh and cellpose_train_pred_seeds.sh. 


# DeepCell training 

DeepCell training and prediction are implemented in jupyter notebooks, run on Linux. DeepCell_tn_nuclear_Kxx.ipynb all train with K' training images, starting from a model trained with Tissuenet 1.0 nuclear data. All training is done for 200 epochs. 

- DeepCell_tn_nuclear_K1a.ipynb trains with images that are cut into 7x4 non-overlapping 512x512 patches. 
    Best mAP 0.37 at mpp=1.
- DeepCell_tn_nuclear_K1b.ipynb trains with images that are cut into 7x25 non-overlapping 512x512 patches. 
    Best mAP 0.37 at mpp=1.
- * DeepCell_tn_nuclear_K2a.ipynb trains with images that are first resized by a factor of 2 and then cut into 175 overlapping 512x512 patches. 
    Best mAP 0.48 at mpp=1.3.
- DeepCell_tn_nuclear_K2b.ipynb trains with images that are resized by 1.54 (1/0.65) and then cut into 112 overlapping 512x512 patches. 
    Best mAP 0.44 at mpp=1.3.
- DeepCell_tn_nuclear_K2c.ipynb trains with images that are resized by 1.54 (1/0.65) and then cut into 112 overlapping 512x512 patches with edge cases removed. 
    Best mAP 0.48 at mpp=1.3.
- DeepCell_tn_nuclear_K3.ipynb trains with CroppingDataGenerator using images resized by 2. Each epoch only trains with 7 images. 
    Best mAP 0.42 at mpp=1.3.

It makes sense that the best mAP for models trained with enlarged images are obtained at a higher mpp than for models with trained with un-enlarged images.

When making predictions with the pretrained nuclear model, mpp 1.2 has the best performance. But since mpp=1.3 corresponds to an enlargement factor of 1.3/0.65=2, we choose that for the notebook K2a. As a sensitivity analysis, we also try mpp=1, which leads to the second resize factor and that is the notebook 2b. 

DeepCell_tn_nuclear_K2a.ipynb performs the best. 

DeepCell_tn_nuclear_K2a_series.ipynb is the one trains at different number of training images.

DeepCell_tn_nuclear_K2a_train7_ari.ipynb is the one trains at different number of training images.


# CellSeg

CellSeg prediction is implemented in jupyter noteook, run on Windows.

We installed CellSeg on a Windows 10 machine following the instructions on https://
michaellee1.github.io/CellSegSite/windows-install.html.
At the completion of the installation, CellSeg failed to run. We rolled back the changes specified in the last section of the instructions on GPU acceleration, reverting Keras and tensorflow to the versions indicated in https://github.com/michaellee1/CellSeg/blob/master/requirements.txt (Keras 2.2.4, tensorflow 1.14.0). CellSeg ran after that and used CPU. We then installed tensorflow-gpu 1.14.0, but CellSeg failed to run again, so we rolled back that change. 

conda activate cellsegsegmenter3


# Pytorch MR-CNN

Pytorch MR-CNN training and prediction are implemented in in shell scripts (pthmrcnn_train_eval.sh, not the jacs repo), run on Linux.

ml Python/3.9.6-GCCcore-11.2.0
ml cuDNN/8.2.2.26-CUDA-11.4.1
ml IPython/7.26.0-GCCcore-11.2.0
env tv013


# Mask_R_CNN-TF2 training 

Mask_R_CNN-TF2 training is implemented in pythons scripts.

By "the original repo", we mean our fork of the alsombra TF 2.4 port of the Matterport repo at https://github.com/youyifong/Mask_RCNN-TF2

The Stringer training script is at https://github.com/MouseLand/cellpose/blob/main/paper/1.0/train_maskrcnn.py

The configuration classes
- mrcnntf2_config_CellSeg.py: settings from the CellSeg paper (Lee et al.)
- mrcnntf2_config_Stringer.py: settings from the Stringer training script

The dataset classes
- mrcnntf2_dataset_Kaggle2018.py: code from the original repo that processes the Kaggle 2018 Data Science Bowl dataset
- mrcnntf2_dataset_Stringer.py: code from the Stringer training script that reads images from a training folder of intensities and masks images

The training scripts
- mrcnntf2_train_Kaggle.py: training with the Kaggle 2018 Data Science Bowl dataset
- mrcnntf2_train_cellpose.py: training with the Cellpose training dataset
- mrcnntf2_train_K.py: training with K's training data

The evaluation script for all cases.
- mrcnntf2_eval.py: 





