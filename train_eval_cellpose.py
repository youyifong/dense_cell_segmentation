### Training ###
"""
Note
- To train Cellpose model, TissueNet images and ground-truth masks should be saved as .tiff files.
- "nohup" works on the server, and the linux command "ps xw" shows current working jobs.

To train the cellpose model (on Linux):
python -m cellpose --train --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/train/" --test_dir "/fh/fast/fong_y/tissuenet_1.0/images/val/" --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 2 --n_epochs 100

-- chan 3: segmenting whole cell with blue
-- chan2 2: optional channel for nuclear with green

To run the newly trained cellpose model (on Linux):
python -m cellpose --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/test/" --pretrained_model "/fh/fast/fong_y/tissuenet_1.0/images/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_21_13_47_58.317948" --chan 3 --chan2 2 --save_tif

#To run the pretrained version of the CellPose model:
#python -m cellpose --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/test/" --pretrained_model cyto2 --chan 3 --chan2 2 --save_tif
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
# IoU between 
def compute_iou(labels, y_pred):
    '''
    Compute the IoU for ground-truth mask (labels) and the predicted mask (y_pred).
    '''
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    
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
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1 # correct objects
    false_positives = np.sum(matches, axis=1) == 0 # missed objects
    false_negatives = np.sum(matches, axis=0) == 0 # extra objects
    tp, fp, fn = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives))
    return tp, fp, fn

# IoU
def iou_at(truths, preds, threshold=0.5, verbose=0):
    '''
    Computes IoU at a given threshold
    '''
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]
    
    tps, fps, fns = 0, 0, 0
    for iou in ious:
        tp, fp, fn = precision_at(threshold, iou)
        tps += tp
        fps += fp
        fns += fn
    p = tps / (tps + fps + fns)
    
    if verbose:
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(threshold, tps, fps, fns, p))
        
    return p


# Summary results
pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
for t in thresholds:
    threshold = t
    pred_vec = []
    for i in range(1249): # total number of test images is 1249 (index: 0-1248)
        print(i)
        label_path = os.path.join(start, 'test'+str(i)+'_masks.tif')
        labels = io.imread(label_path)
        res_path = os.path.join(start, 'test'+str(i)+'_img_seg.npy')
        res = np.load(res_path, allow_pickle=True).item()
        masks = res['masks']
        pred_vec.append(iou_at([labels], [masks], threshold=threshold, verbose=0))
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


# Displaying img, groud-truth masks, predicted masks # 
img = io.imread("/Users/shan/Desktop/test0_img.tif") # image
masks = io.imread("/Users/shan/Desktop/test0_masks.tif") # groun-truth masks
masks = io.imread("/Users/shan/Desktop/test0_img_cp_masks.tif") # predicted masks

my_dpi = 96
outlines = utils.masks_to_outlines(masks)
outX, outY = np.nonzero(outlines)
imgout= img.copy()
imgout[outX, outY] = np.array([255,75,75])
fig=plt.figure(figsize=(1392/my_dpi, 1040/my_dpi), dpi=my_dpi); plt.gca().set_axis_off(); plt.imshow(imgout)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator()); plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.savefig("/Users/shan/Desktop/test0_img.png", bbox_inches = 'tight', pad_inches = 0); plt.close('all')





### Appendix ###
"""
def compute_iou(labels, y_pred):
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1) # makes true_objects * 1
    area_pred = np.expand_dims(area_pred, 0) # makes 1 * pred_objects
    union = area_true + area_pred - intersection
    iou = intersection / union
    return iou[1:, 1:] # exclude background; remove frequency for bin [0,1)

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1 # correct objects
    false_positives = np.sum(matches, axis=1) == 0 # missed objects
    false_negatives = np.sum(matches, axis=0) == 0 # extra objects
    tp, fp, fn = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives))
    return tp, fp, fn

def iou_at(truths, preds, threshold=0.5, verbose=0):
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]
    tps, fps, fns = 0, 0, 0
    for iou in ious:
        tp, fp, fn = precision_at(threshold, iou)
        tps += tp
        fps += fp
        fns += fn
    p = tps / (tps + fps + fns)
    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(p))
    return p
"""