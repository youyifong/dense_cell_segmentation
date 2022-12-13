"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Heavily modified by Carsen Stringer for general datasets (12/2019)
------------------------------------------------------------

"""

# Import mrcnn libraries from the following
mrcnn_path='../Mask_RCNN-TF2'
import sys, os
assert os.path.exists(mrcnn_path), 'mrcnn_path does not exist: '+mrcnn_path
sys.path.insert(0, "../CellSeg/src") # to import CellSeg
sys.path.insert(0, mrcnn_path) 


import os, sys
import numpy as np
import skimage.io
from skimage import img_as_ubyte, img_as_uint
import syotil

from mrcnn import model as modellib # mrcnn tf2 repo
from cvmodelconfig import CVSegmentationConfig # CellSeg repo
from mrcnntf2_dataset_Stringer import StringerDataset
from mrcnntf2_config_Stringer import StringerEvalConfig
from mrcnntf2_config_CellSeg import CellSegInferenceConfig


basedir = './' 

MODELS_DIR = os.path.join(basedir, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def mask_3dto2d(mask, scores):
    "transform a mask array that is [H, W, count] to [H, W]"
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return np.zeros(mask.shape[:3])
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    return mask


def remove_overlaps_Stringer(masks, cellpix, medians):
    """ replace overlapping mask pixels with mask id of closest mask
        masks = Nmasks x Ly x Lx
    """
    overlaps = np.array(np.nonzero(cellpix>1.5)).T
    dists = ((overlaps[:,:,np.newaxis] - medians.T)**2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:,0], overlaps[:,1]] = 0
    masks[tocell, overlaps[:,0], overlaps[:,1]] = 1

    # labels should be 1 to mask.shape[0]
    masks = masks.astype(int) * np.arange(1,masks.shape[0]+1,1,int)[:,np.newaxis,np.newaxis]
    masks = masks.sum(axis=0)
    return masks


###############################################################################
# main
###############################################################################


import argparse
parser = argparse.ArgumentParser(description='Mask R-CNN for cell counting and segmentation')
parser.add_argument('--gpu_id', default = 1, type=int, help='which gpu to run on')
parser.add_argument('--dataset', default="images/test_images", metavar="/path/to/dataset/", help='Root directory of the dataset')
parser.add_argument('--batch_size', default = 2, type=int, help='batch_size')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id) # set which gpu to use


##############################################################
# config and weight

# test image is size 233 x 1040 

## CellSeg pretrained model

"""
CellSeg config
smallest_side=128: 0.35
smallest_side=186: 0.40
smallest_side=256: 0.39
smallest_side=300: 0.35
smallest_side=512: a lot worse
"""
# weights_path = "../CellSeg/src/modelFiles/final_weights.h5"
# config = CVSegmentationConfig(smallest_side=186) # scaled by 256x2.5/233 = 2.7
# config.NAME = "CellSeg"
# config.PRE_NMS_LIMIT = 6000
# config.DETECTION_MIN_CONFIDENCE       = 0.8 # if set to 0.5, mAP=0.30; if set to 0.7, mAP=0.37; if set to 0.8 (default), mAP=0.39


"""
Stringer config to match performance of CellSeg config
mAP 0.41
"""
# weights_path = "../CellSeg/src/modelFiles/final_weights.h5"
# config = StringerEvalConfig() # scaled by 2
# changing BACKBONE and MEAN_PIXEL has a large impact on mAP
# config.BACKBONE                       = "resnet101" # changed from resnet50
# config.MEAN_PIXEL                     = [123.7, 116.8, 103.9] # changed from [43.53 39.56 48.22]
# config.DETECTION_MIN_CONFIDENCE       = 0.7 # if set to 0.5, mAP=0.39; if set to 0.7, mAP=0.41. Default 0.5


"""
model trained with cellpose data
epochs 20+180
epoch 060, the mAP is 0.02. 
epoch 200, the mAP is 0.33. 
"""
# weights_path = "models/cellpose20221129T2150/mask_rcnn_cellpose_0200.h5"
# config = StringerEvalConfig() # scale up by 2


"""
models trained with Kaggle data using a reverse engineered CellSeg model

models trained with DETECTION_MIN_CONFIDENCE set to 0.75:
    
kaggle20221208T1050
head epochs 20 lr 0.001, all epochs lr 0.001
epoch 020 0.27
epoch 030 0.31
epoch 040 0.30
epoch 060 0.31
epoch 080 0.33
epoch 120 0.34
epoch 160 0.33

kaggle20221208T1052
head epochs 150 lr 0.001, all epochs lr 0.001
epoch 020 0.30
epoch 100 0.27
epoch 150 0.31
epoch 180 0.32



models trained with DETECTION_MIN_CONFIDENCE set to 0.8:
    
kaggle20221207T1115
head epochs 150 lr 0.001, all epochs 50  lr 0.001
epoch 020 0.26
epoch 150 0.23
epoch 160 0.24
epoch 170 0.26
epoch 180 0.26
epoch 190 0.36
epoch 200 0.35



models trained with DETECTION_MIN_CONFIDENCE set to 0.5:

cellseg20221204T2219
head epochs 20 lr 0.001, all epochs 180  lr 0.001
epoch 030 0.40 
epoch 050 0.34 
epoch 075 0.32

cellseg20221205T1851
head epochs 150  lr 0.001, all epochs 25  lr 0.0005
epoch 050 .29
epoch 100 .32
epoch 150 .33
epoch 175 .36

cellseg20221206T1457
head epochs 150 lr 0.001, all epochs 50 lr 0.001
epoch 150 .32
epoch 175 .35
epoch 200 .29
"""
# weights_path = "models/cellseg20221204T2219/mask_rcnn_cellseg_0030.h5"
# weights_path = "models/cellseg20221205T1851/mask_rcnn_cellseg_0050.h5"
# weights_path = "models/cellseg20221206T1457/mask_rcnn_cellseg_0180.h5"
# weights_path = "models/cellsegconfig20221207T1108/mask_rcnn_cellseg_0200.h5"
# weights_path = "models/kaggle20221207T1115/mask_rcnn_kaggle_0050.h5"
# weights_path = "models/kaggle20221208T1050/mask_rcnn_kaggle_0120.h5"
weights_path = "models/kaggle20221208T1052/mask_rcnn_kaggle_0180.h5"
config = CellSegInferenceConfig() 
config.IMAGE_MIN_DIM=0; config.IMAGE_MIN_SCALE=2 # to scale by 2


##############################################################
# reload model in inference mode

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODELS_DIR)    
model.load_weights(weights_path, by_name=True)

# Read dataset
dataset = StringerDataset()
dataset.load_data(args.dataset, '')
dataset.prepare()
print("Running on {}".format(args.dataset))

# if need to save mask images
# results_dir = "testmasks_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
# print("Mask pngs saved to {}".format(args.results_dir))

# masks with overlap removed work better
remove_overlap=True
AP_arr=[]
for image_id in dataset.image_ids:
    image = dataset.load_image(image_id)
    r = model.detect([image], verbose=0)[0]
    mask = r["masks"]
    
    if remove_overlap:
        medians = []
        for m in range(mask.shape[-1]):
            ypix, xpix = np.nonzero(mask[:,:,m])
            medians.append(np.array([ypix.mean(), xpix.mean()]))
        mask = np.int32(remove_overlaps_Stringer(np.transpose(mask, (2,0,1)), mask.sum(axis=-1), np.array(medians)))             
    else:
        # save masks as 2D image
        mask = mask_3dto2d(mask, r["scores"])

    truth=skimage.io.imread("images/test_gtmasks/"+dataset.image_info[image_id]["id"].replace("img","masks")+".png")
    AP_arr.append(syotil.csi(mask, truth))# masks may lost one pixel
    
    # skimage.io.imsave("{}/{}.png".format(results_dir, dataset.image_info[image_id]["id"].replace("_img","_masks")), img_as_uint(mask), check_contrast=False)
    
#print(AP_arr)
print(np.mean(AP_arr))