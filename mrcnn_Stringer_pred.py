"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Heavily modified by Carsen Stringer for general datasets (12/2019)
------------------------------------------------------------

python ../Stringer_maskrcnn_pred.py --dataset=. --weights_path=./models/.20220827T0921/mask_rcnn_._0500.h5


"""
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

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import matplotlib.pyplot as plt

from stardist import matching

from maskrcnn_Stringer_NucleusConfig import *


basedir = './' # where to save outputs
MODELS_DIR = os.path.join(basedir, "models")
# Save submission files here
RESULTS_DIR = os.path.join(basedir, "maskrcnn/")


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


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


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
#  Detection
############################################################

def detect(model, dataset_dir, RESULTS_DIR=RESULTS_DIR):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))
    #config.BATCH_SIZE = 1
    #config.IMAGES_PER_GPU = 1
    #config.GPU_COUNT = 1
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, '')
    dataset.prepare()
    # Load over images
    submission = []
    masks = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        #source_id = dataset.image_info[image_id]["id"]
        #rle = mask_to_rle(source_id, r["masks"], r["scores"])
        masks.append(r["masks"])
        #submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to npy file
    file_path = os.path.join(submit_dir, "overlapping_masks.npy")
    np.save(file_path, {'masks': masks})

    print("Saved to ", submit_dir)

    return masks



############################################################
#  Command Line
############################################################

if __name__ == '__main__':

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Mask R-CNN for cell counting and segmentation')
    parser.add_argument('--dataset', required=False, default=".", metavar="/path/to/dataset/", help='Root directory of the dataset')
    parser.add_argument('--weights_path', required=True, help="Path to weights .h5 file")
    parser.add_argument('--batch_size', default = 2, type=int, help='batch_size')
    args = parser.parse_args()
    
    dataset = os.path.basename(os.path.normpath(args.dataset))
    dataset_test = os.path.join(args.dataset, 'test/')
    print("Dataset: ", dataset)
    
    config = NucleusInferenceConfig()
    config.NAME = dataset
    
    # reload model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODELS_DIR)    
    model.load_weights(args.weights_path, by_name=True)
    
#    # score output
#    ndataset = NucleusDataset()
#    ndataset.load_nucleus(dataset_test, '')
#    ndataset.prepare()
#    APs = compute_batch_ap(ndataset, ndataset.image_ids)
#    print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))

    overlapping_masks = detect(model, dataset_test)
    
    masks = []
    for i in range(len(overlapping_masks)):
        mask = overlapping_masks[i]
        medians = []
        for m in range(mask.shape[-1]):
            ypix, xpix = np.nonzero(mask[:,:,m])
            medians.append(np.array([ypix.mean(), xpix.mean()]))
        masks.append(np.int32(remove_overlaps(np.transpose(mask, (2,0,1)), 
                                                           mask.sum(axis=-1), np.array(medians))))
    mlist = glob.glob(os.path.join(dataset_test, '*_masks.png'))
    Y_test = [skimage.io.imread(fimg)+1 for fimg in mlist]
    rez = matching.matching_dataset(Y_test, masks, thresh=[0.5,0.75,.9], by_image=True)
    print(rez)
