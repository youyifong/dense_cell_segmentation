import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from read_roi import read_roi_file # pip install read-roi
from PIL import Image, ImageDraw
from cellpose import utils, io


def compute_iou(labels, y_pred):
    '''
    Compute the IoU for ground-truth mask (labels) and the predicted mask (y_pred).
    '''
    true_objects = (np.unique(labels))
    pred_objects = (np.unique(y_pred))
    
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(np.append(true_objects, np.inf),np.append(pred_objects, np.inf)))[0] # compute the 2D histogram of two data samples; it returns frequency in each bin
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=np.append(true_objects, np.inf))[0]
    area_pred = np.histogram(y_pred, bins=np.append(pred_objects, np.inf))[0]
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
def csi_old(truths, preds, threshold=0.5, verbose=0):
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

