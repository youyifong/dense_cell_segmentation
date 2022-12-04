"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Heavily modified by Carsen Stringer for general datasets (12/2019)
------------------------------------------------------------

python ../tfmrcnn_Stringer_eval.py --dataset=. --weights_path=models/cellpose20221129T2150/mask_rcnn_cellpose_0130.h5

"""

# Import mrcnn libraries from the following
mrcnn_path='../Mask_RCNN-TF2'
import sys, os
assert os.path.exists(mrcnn_path), 'mrcnn_path does not exist: '+mrcnn_path
sys.path.insert(0, "../CellSeg/src") # to import CellSeg
sys.path.insert(0, mrcnn_path) 


if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os, sys, datetime, glob,pdb
import numpy as np
#np.random.bit_generator = np.random._bit_generator
import skimage.io
from imgaug import augmenters as iaa
from skimage import img_as_ubyte, img_as_uint
import syotil

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import matplotlib.pyplot as plt

from cvmodelconfig import CVSegmentationConfig

#from stardist import matching

from tfmrcnn_CellsegDataset import *
from tfmrcnn_StringerConfig import *


basedir = './' 

MODELS_DIR = os.path.join(basedir, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


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


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


def compute_batch_ap(dataset, image_ids, verbose=1):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id)#, use_mini_mask=False)
        # Run object detection
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
            print("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs


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






############################################################
#  Command Line
############################################################

if __name__ == '__main__':

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Mask R-CNN for cell counting and segmentation')
    parser.add_argument('--dataset', default="images/test_images", metavar="/path/to/dataset/", help='Root directory of the dataset')
    parser.add_argument('--batch_size', default = 2, type=int, help='batch_size')
    parser.add_argument('--gpu_id', default = 1, type=int, help='which gpu to run on')
    # parser.add_argument('--weights_path', default="models/cellpose20221129T2150/mask_rcnn_cellpose_0200.h5", help="Path to weights .h5 file")
    parser.add_argument('--weights_path', default="../CellSeg/src/modelFiles/final_weights.h5", help="Path to weights .h5 file")
    parser.add_argument('--results_dir', required=False, default = "images/test_tfmrcnn_cellseg2", help='mask files will be saved under a timestamped folder under the results_dir')
    args = parser.parse_args()

    # set which gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)    
    
    # config = StringerEvalConfig(); config.NAME = "cellpose"
    config = CVSegmentationConfig(smallest_side=256); config.NAME = "CellSeg"; config.PRE_NMS_LIMIT = 6000
    
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
    print("Results saved to {}".format(args.results_dir))

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
        
        skimage.io.imsave("{}/{}.png".format(results_dir, dataset.image_info[image_id]["id"].replace("_img","_masks")), 
                      img_as_uint(mask), check_contrast=False)
        
        # Encode image to RLE. Returns a string of multiple lines
        #source_id = dataset.image_info[image_id]["id"]
        #rle = mask_to_rle(source_id, r["masks"], r["scores"])

    print("Saved to ", results_dir)
    print(AP_arr)
