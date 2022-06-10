import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from read_roi import read_roi_file # pip install read-roi
from PIL import Image, ImageDraw
from cellpose import utils, io


# IoU
def compute_iou(mask_true, mask_pred):
    '''
    Compute the IoU for ground-truth mask (mask_true) and predicted mask (mask_pred).
    '''
    true_objects = (np.unique(mask_true))
    pred_objects = (np.unique(mask_pred))
    
    # Compute intersection between all objects
    # compute the 2D histogram of two data samples; it returns frequency in each bin
    # important to append n.inf otherwise the number of bins will be 1 less than the number of unique masks
    intersection = np.histogram2d(mask_true.flatten(), mask_pred.flatten(), bins=(np.append(true_objects, np.inf),np.append(pred_objects, np.inf)))[0] 
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(mask_true, bins=np.append(true_objects, np.inf))[0]
    area_pred = np.histogram(mask_pred, bins=np.append(pred_objects, np.inf))[0]
    area_true = np.expand_dims(area_true, -1) # makes true_objects * 1
    area_pred = np.expand_dims(area_pred, 0) # makes 1 * pred_objects
    
    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union
    return iou[1:, 1:] # exclude background; remove frequency for bin [0,1)

# TP, FP, FN
def tp_fp_fn(threshold, iou):
    '''
    Computes true positive (TP), false positive (FP), and false negative (FN) at a given threshold
    '''
    matches = iou >= threshold
    true_positives = np.sum(matches, axis=1) >= 1 # predicted masks are matched to true masks
    false_positives = np.sum(matches, axis=1) == 0 # predicted masks are matched to false masks (number of predicted masks - TP)
    false_negatives = np.sum(matches, axis=0) == 0 # true masks are not matched to predicted masks (number of true masks - TP)
    tp, fp, fn = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives))
    return tp, fp, fn

# CSI
def csi(mask_true, mask_pred, threshold=0.5):
    '''
    Compute CSI (= TP/(TP+FP+FN)) at a given threshold
    '''
    iou = compute_iou(mask_true, mask_pred)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    csi = tp / (tp + fp + fn)
    return csi

# Precision
def precision(mask_true, mask_pred, threshold=0.5):
    '''
    Compute precision (= TP/(TP+FP) at a given threshold
    '''
    iou = compute_iou(mask_true, mask_pred)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    precision = tp / (tp + fp)
    return precision

# Recall
def recall(mask_true, mask_pred, threshold=0.5):
    '''
    Compute Recall (= TP/(TP+FN)) at a given threshold
    '''
    iou = compute_iou(mask_true, mask_pred)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    recall = tp / (tp + fn)
    return recall


# From .roi files to masks file
def roifiles2mask(roi_files, width, height):
    files = glob.glob(roi_files) 
    masks = Image.new('I', (width, height), 0)
    for idx in range(len(files)):
        print(idx)
        mask_temp = read_roi_file(files[idx])
        filename = files[idx].split('\\')[-1][:-4]
        x = mask_temp[filename]['x']
        y = mask_temp[filename]['y']
            
        polygon = []
        for i in range(len(x)):
            polygon.append((x[i], y[i]))
        
        ImageDraw.Draw(masks).polygon(polygon, outline=idx+1, fill=idx+1)
        
    masks = np.array(masks, dtype=np.uint16) # resulting masks
    plt.imshow(masks, cmap='gray') # display ground-truth masks
    plt.show()
    io.imsave(os.path.split(roi_files)[0]+'_masks.png', masks)
    
    outlines = utils.masks_to_outlines(masks)
    plt.imsave(os.path.split(roi_files)[0] + "_masks_outline.png", outlines, cmap='gray')


def maskfile2outline(mask_file):
    masks = io.imread(mask_file)
    outlines = utils.masks_to_outlines(masks)
    plt.imsave(os.path.splitext(mask_file)[0] + "_outline.png", outlines, cmap='gray')

    
    

    
### Appendix ###
#    The following function is modified based on "_label_overlap()" and "_intersection_over_union" functions in cellpose github (https://github.com/MouseLand/cellpose/blob/main/cellpose/metrics.py).
#    For "intersection" below, the original functions seem not to deal with empty masks between background (value 0) and mask with maximum number (maximum value). It makes a difference between iou_map() and compute_iou() functions.
#    We modifed it so as to remove empty masks in the "intersection". After the modification, iou_map() and compute_iou() functions generates the same results.
'''
def iou_map(masks_ture, masks_pred):
    """IoU: Intersection over Union between true masks and predicted masks
       
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
    
    # modification #
    empty_mask_idx = []
    for i in range(intersection.shape[0]):
        if(sum(intersection[i,:]) == 0): empty_mask_idx.append(i)
    intersection = np.delete(intersection, empty_mask_idx, 0)
    
    n_pixels_true = np.sum(intersection, axis=1, keepdims=True)
    n_pixels_pred = np.sum(intersection, axis=0, keepdims=True)
    iou = intersection / (n_pixels_true + n_pixels_pred - intersection)
    iou[np.isnan(iou)] = 0.0
    return iou
'''
