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


import os, sys, datetime
import numpy as np
import skimage.io
from skimage import img_as_ubyte, img_as_uint
import syotil

from mrcnn import model as modellib

from tfmrcnn_CellsegDataset import CellsegDataset
from tfmrcnn_config_stringer import StringerEvalConfig
from tfmrcnn_config_nucleus import NucleusInferenceConfig

# CellSeg
from cvmodelconfig import CVSegmentationConfig

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


def remove_overlaps(masks, cellpix, medians):
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


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Mask R-CNN for cell counting and segmentation')
    parser.add_argument('--gpu_id', default = 1, type=int, help='which gpu to run on')
    parser.add_argument('--dataset', default="images/test_images", metavar="/path/to/dataset/", help='Root directory of the dataset')
    parser.add_argument('--batch_size', default = 2, type=int, help='batch_size')

    # parser.add_argument('--weights_path', default="../CellSeg/src/modelFiles/final_weights.h5", help="Path to weights .h5 file")
    # parser.add_argument('--weights_path', default="models/cellpose20221129T2150/mask_rcnn_cellpose_0200.h5", help="Path to weights .h5 file")
    parser.add_argument('--weights_path', default="models/cellseg20221204T2219/mask_rcnn_cellseg_0030.h5", help="Path to weights .h5 file")

    parser.add_argument('--results_dir', required=False, default = "images/test_tfmrcnn_cellseg2", help='place to save masks.png')

    args = parser.parse_args()

    # set which gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)    
    
    # config = StringerEvalConfig()
    # config.NAME = "cellpose200"        
    # config.BACKBONE                       = "resnet101" # changed from resnet50
    # config.MEAN_PIXEL                     = [123.7, 116.8, 103.9] # changed from [43.53 39.56 48.22]
    # config.DETECTION_MIN_CONFIDENCE       = 0.7 # changed from 0

    config = NucleusInferenceConfig()

    # config = CVSegmentationConfig(smallest_side=256)
    # config.NAME = "CellSeg"
    # config.PRE_NMS_LIMIT = 6000
    
    # reload model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODELS_DIR)      
    
    model.load_weights(args.weights_path, by_name=True)
    
    print("Running on {}".format(args.dataset))
    
    if args.results_dir: 
        results_dir=args.results_dir
    else:
        results_dir = "testmasks_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print("Mask pngs saved to {}".format(args.results_dir))

    # Read dataset
    dataset = CellsegDataset()
    dataset.load_data(args.dataset, '')
    dataset.prepare()

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
            mask = np.int32(remove_overlaps(np.transpose(mask, (2,0,1)), mask.sum(axis=-1), np.array(medians)))             
        else:
            # save masks as 2D image
            mask = mask_3dto2d(mask, r["scores"])

        truth=skimage.io.imread("images/test_gtmasks/"+dataset.image_info[image_id]["id"].replace("img","masks")+".png")
        AP_arr.append(syotil.csi(mask, truth))# masks may lost one pixel
        
        # skimage.io.imsave("{}/{}.png".format(results_dir, dataset.image_info[image_id]["id"].replace("_img","_masks")), img_as_uint(mask), check_contrast=False)
        
    print(AP_arr)
    print(np.mean(AP_arr))


"""
cellpose20221129T2150
mask_rcnn_cellpose_0200, the mAP is 0.33. 
These two config changes decrease mAP substantially.
    config.BACKBONE                       = "resnet101" # changed from resnet50
    config.MEAN_PIXEL                     = [123.7, 116.8, 103.9] # changed from [43.53 39.56 48.22]
This config change does not affect mAP.
    config.DETECTION_MIN_CONFIDENCE       = 0.7 # changed from 0

cellseg20221204T2219
epochs 20-200
for mask_rcnn_cellseg_0075, the mAP is 0.32. 
for mask_rcnn_cellseg_0050, the mAP is 0.34. 
for mask_rcnn_cellseg_0030, the mAP is 0.40. 


"""
