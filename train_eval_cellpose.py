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

# Check whether gpu is available
if torch.cuda.is_available() :
    gpu = True
else :
    gpu = False

# Set working directory
if(os.name!="nt") : os.chdir('/fh/fast/fong_y/tissuenet_1.0/images/test')
start = os.getcwd()


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
        pred_vec.append(csi(masks_true, masks_pred, threshold))
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
"""
# Utility
# IoU between 
def compute_iou(labels, y_pred):
    '''
    Compute the IoU for ground-truth mask (labels) and the predicted mask (y_pred).
    '''
    true_objects = (np.unique(labels))
    pred_objects = (np.unique(y_pred))
    
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0] # compute the 2D histogram of two data samples; it returns frequency in each bin
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1) # makes true_objects * 1
    area_pred = np.expand_dims(area_pred, 0) # makes 1 * pred_objects
    
    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union
    return iou[1:, 1:] # exclude background; remove frequency for bin [0,1)

# Precision
def precision_at(threshold, iou):
    '''
    Computes the precision at a given threshold
    '''
    matches = iou >= threshold
    true_positives = np.sum(matches, axis=1) >= 1 # correct objects
    false_positives = np.sum(matches, axis=1) == 0 # missed objects
    false_negatives = np.sum(matches, axis=0) == 0 # extra objects
    tp, fp, fn = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives))
    return tp, fp, fn

# IoU
def csi(truths, preds, threshold=0.5, verbose=0):
    '''
    Computes IoU at a given threshold
    '''
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]    
    #tps, fps, fns = 0, 0, 0
    ps=[]
    for iou in ious:
        tp, fp, fn = precision_at(threshold, iou)
        #tps += tp
        #fps += fp
        #fns += fn
        p = tp / (tp + fp + fn) 
        ps.append(p)
    p=np.mean(ps)
    #if verbose:
    #    print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(threshold, tps, fps, fns, p))        
    return p
"""
