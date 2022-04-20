### Training ###
"""
Save TissueNet as individulal TIFF files.

These files are then fed into cellpose for training.

To train the cellpose model:

python -m cellpose --train --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/split/train/" --test_dir "/fh/fast/fong_y/tissuenet_1.0/images/split/val/" --pretrained_model None --img_filter _img --mask_filter _masks --chan 2 --chan2 1 --n_epochs 100


To run the newly trained cellpose model:

python -m cellpose --use_gpu --dir "/fh/fast/fong_y/tissuenet_1.0/images/split/test/" --pretrained_model "/fh/fast/fong_y/tissuenet_1.0/images/split/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_04_18_15_14_22.493804" --chan 2 --chan2 1 --save_tif

To run the pretrained version of the CellPose model:

python -m cellpose
    --dir /deepcell_data/users/willgraf/cellpose/test_split_1_channels_first
    --pretrained_model cyto
    --chan 0 --chan2 1
    --diameter 0.
    --save_tif --use_gpu
"""


# Library
import os
import numpy as np
import tifffile

SEED = 1

MODEL_NAME = 'cellpose'
ROOT_DIR = '/fh/fast/fong_y/tissuenet_1.0'
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = ROOT_DIR
TIFF_PATH = os.path.join(DATA_DIR, 'images', 'seed{}'.format(SEED))

TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'tissuenet_v1.0_train.npz')
VAL_DATA_FILE = os.path.join(DATA_DIR, 'tissuenet_v1.0_val.npz')
TEST_DATA_FILE = os.path.join(DATA_DIR, 'tissuenet_v1.0_test.npz')
TEST_PRED_DATA_FILE = os.path.join(DATA_DIR, 'cellpose_test_pred.npz')

def save_as_tiffs(npz_path, tiff_dir):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    assert X.shape[0] == y.shape[0], 'X and y should have the same number of images.'
    for i in range(X.shape[0]):
        img_filename = '{:04d}_img.tif'.format(i)
        mask_filename = '{:04d}_masks.tif'.format(i)
        tifffile.imsave(os.path.join(tiff_dir, img_filename), X[i])
        #tifffile.imsave(os.path.join(tiff_dir, mask_filename), y[i]) # original; it did not work due to image shape issue
        tifffile.imsave(os.path.join(tiff_dir, mask_filename), y[i,:,:,0]) # modified; it works (maybe only for nuclear ground-truth)
    print('saved %s files to %s' % (len(X), tiff_dir))

if __name__ == '__main__':
    data_files = [
        ('train', TRAIN_DATA_FILE),
        ('val', VAL_DATA_FILE),
        ('test', TEST_DATA_FILE),
    ]
    for prefix, data_file in data_files:
        f = os.path.join(DATA_DIR, data_file)
        subdir = os.path.join(TIFF_PATH, prefix)
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        save_as_tiffs(f, subdir)

    X_train = train_data['X']
    y_train = train_data['y']

    X_val = val_data['X']
    y_val = val_data['y']

    X_test = test_data['X']
    y_test = test_data['Y']





### Evaluation ###
# Library
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from cellpose import utils, models, io

# Check whether gpu is available #
if torch.cuda.is_available() :
    gpu = True
else :
    gpu = False

# Change working directory if needed
if(os.name!="nt") : os.chdir('/fh/fast/fong_y/tissuenet_1.0/images/split/test')
start = os.getcwd()

# Utility
# IoU
def compute_iou(labels, y_pred):
    '''
    Compute the IoU for instance labels and predictions.
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

# Overall metric
# IoU
def iou_map(truths, preds, verbose=0):
    '''
    Compute the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated, and 0 is the background
    '''
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]
    print(ious[0].shape)
    
    if verbose:
        print("\Thresh\tTP\tFP\tFN\tPrecision")
    
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn
        p = tps / (tps + fps + fns)
        prec.append(p)
        
        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))
        
    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        
    return np.mean(prec)



# IoU
pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
for t in thresholds:
    threshold = t
    pred_vec = []
    for i in range(1249): # total number of test images is 1249 (index: 0-1248)
        print(i)
        label_path = os.path.join(start, '{:04d}_masks.tif'.format(i))
        labels = io.imread(label_path)
        res_path = os.path.join(start, '{:04d}_img_seg.npy'.format(i))
        res = np.load(res_path, allow_pickle=True).item()
        masks = res['masks']
        pred_vec.append(iou_map([labels], [masks], threshold=threshold, verbose=0))
    pred_mat.append(pred_vec)

# Save result as .txt file
pred_mat = pd.DataFrame(pred_mat).T
colnames = []
for i in thresholds: colnames.append("iou_threshold_" + str(i))
pred_mat.columns = colnames
rownames = []
for i in range(1249): rownames.append("test" + str(i+1))
pred_mat.index = rownames
pred_mat.to_csv(os.path.join("/fh/fast/fong_y/tissuenet_1.0/results", "cellpose_iou_threshold.txt"), header=True, index=True, sep=',')



# Displaying img, groud-truth masks, predicted masks # 
img = io.imread("/Users/shan/Desktop/0000_img.tif")
img = img.reshape(1,256,256,2)
rgb_images = create_rgb_image(img, channel_colors=['green', 'blue'])
img = rgb_images[0]

masks = io.imread("/Users/shan/Desktop/0000_masks.tif")
masks = masks[:,:,1]

masks = io.imread("/Users/shan/Desktop/0000_img_cp_masks.tif")
masks = masks[:,:,1]

my_dpi = 96
outlines = utils.masks_to_outlines(masks)
outX, outY = np.nonzero(outlines)
imgout= img.copy()
imgout[outX, outY] = np.array([255,75,75])
fig=plt.figure(figsize=(1392/my_dpi, 1040/my_dpi), dpi=my_dpi); plt.gca().set_axis_off(); plt.imshow(imgout)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator()); plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.savefig("/Users/shan/Desktop/0000_img_cp_masks_rgb.png", bbox_inches = 'tight', pad_inches = 0); plt.close('all')





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

def iou_map(truths, preds, threshold=0.5, verbose=0):
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]
    if verbose:
        print("\Thresh\tTP\tFP\tFN\tPrecision")
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