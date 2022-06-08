### Training ###
"""
Note
- To train Cellpose model, TissueNet images and ground-truth masks should be saved as .tiff files.
- "nohup" works on the server, and the linux command "ps xw" or "top -u username" shows current working jobs.

To train the cellpose model (on Linux):
python -m cellpose --train --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/train/" --test_dir "/fh/fast/fong_y/tissuenet_1.0/images/val/" --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 2 --n_epochs 100

-- chan 3: segmenting whole cell with blue
-- chan2 2: optional channel for nuclear with green

To run the newly trained cellpose model (on Linux):
python -m cellpose --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/test/" --pretrained_model "/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_21_13_47_58.317948" --chan 3 --chan2 2 --save_tif

To run the pretrained version of the CellPose model:
python -m cellpose --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/test/" --pretrained_model cyto2 --chan 3 --chan2 2 --save_tif
"""



### Evaluation ###
# Library
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from cellpose import utils, models, io

from utils import * # this file should be in the current working directory at this point
os.chdir("../K's training data")
#exec(open("/home/shan/utils.py").read()) # alternative to "import utils"


# Check whether gpu is available
if torch.cuda.is_available() :
    gpu = True
else :
    gpu = False

    
# Set working directory
if(os.name!="nt") : os.chdir('/fh/fast/fong_y/tissuenet_1.0/images/test')
start = os.getcwd()


# Summary results
pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
for t in thresholds:
    threshold = t
    pred_vec = []
    for i in range(1249): # total number of test images is 1249 (index: 0-1248)
        print(i)
        masks_true_path = os.path.join(start, 'test'+str(i)+'_masks.tif')
        masks_true = io.imread(masks_true_path)
        masks_pred_path = os.path.join(start, 'res_cellpose-TN2','test'+str(i)+'_img_seg.npy')
        masks_pred_res = np.load(masks_pred_path, allow_pickle=True).item()
        masks_pred = masks_pred_res['masks']
        #pred_vec.append(csi(masks_true, masks_pred, threshold))
        pred_vec.append(csi_old([masks_true],[masks_pred], threshold=t, verbose=0))
    pred_mat.append(pred_vec)


# Save results
pred_mat = pd.DataFrame(pred_mat).T
colnames = []
for i in thresholds: colnames.append("iou_threshold_" + str(i))
pred_mat.columns = colnames
rownames = []
for i in range(1249): rownames.append("test" + str(i+1))
pred_mat.index = rownames
pred_mat.to_csv(os.path.join("/fh/fast/fong_y/tissuenet_1.0/results", "cellpose_iou_threshold.txt"), header=True, index=True, sep=',')





### Appendix ###
# Utility
# IoU
def iou_map(masks_ture, masks_pred):
    """IoU: Intersection over Union between true masks and predicted masks
    This function is modified based on "_label_overlap()" and "_intersection_over_union" functions in cellpose github (https://github.com/MouseLand/cellpose/blob/main/cellpose/metrics.py)
    
    Inputs:
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    
    Outputs:
    iou: ND-array, float
        IoU map
    """
    x = masks_true.ravel() # flatten matrix to vector
    y = masks_pred.ravel() # flatten matrix to vector
    true_objects = masks_true.max()+1
    pred_objects = masks_pred.max()+1
    intersection = np.zeros((true_objects,pred_objects), dtype=np.uint)
    for i in range(len(x)):
        intersection[x[i], y[i]] += 1
    n_pixels_true = np.sum(intersection, axis=1, keepdims=True)
    n_pixels_pred = np.sum(intersection, axis=0, keepdims=True)
    iou = intersection / (n_pixels_true + n_pixels_pred - intersection)
    iou[np.isnan(iou)] = 0.0
    return iou

# TP, FP, FN
def tp_fp_fn(threshold, iou):
    """Computes true positive (TP), false positive (FP), and false negative (FN) at a given threshold
    
    Inputs:
    iou: ND-array, float
        IoU map
    threshold: float
        threshold on IoU for positive label
    
    Outputs:
    TP, FP, FN
    """
    matches = iou >= threshold
    true_positives = np.sum(matches, axis=1) >= 1 # predicted masks are matched to true masks
    false_positives = np.sum(matches, axis=0) == 0 # predicted masks are matched to false masks (number of predicted masks - TP)
    false_negatives = np.sum(matches, axis=1) == 0 # true masks are not matched to predicted masks (number of true masks - TP)
    tp, fp, fn = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives))
    return tp, fp, fn

# CSI
def csi(masks_true, masks_pred, threshold=0.5):
    """
    Compute CSI (= TP/(TP+FP+FN)) at a given threshold
    """
    iou = iou_map(masks_true, masks_pred)[1:, 1:] # ingnore background (masks=0)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    csi = tp / (tp + fp + fn)
    return csi
