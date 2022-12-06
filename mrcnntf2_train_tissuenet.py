# Adapted from the Nucleus example of the Matterport implementation of Mask R-CNN (Alsombra port to Tensorflow 2.4.1)
# with reference to the Stringer modification 

import datetime
t1=datetime.datetime.now()

from mrcnntf2_cellsegdataset import *




train_dir="/fh/fast/fong_y/tissuenet_v1.0/images/train_nuclear"
val_dir="/fh/fast/fong_y/tissuenet_v1.0/images/val_nuclear"
test_dir="/fh/fast/fong_y/tissuenet_v1.0/images/test"

import matplotlib.pyplot as plt
import os
import sys
import glob
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa


ROOT_DIR="."

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "../mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs_maskrcnn_matterport_alsombra")
if not os.path.isdir(DEFAULT_LOGS_DIR):
    os.mkdir(DEFAULT_LOGS_DIR)

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results_maskrcnn_matterport_alsombra/")
if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


# Training dataset
dataset_train = TissueNetNucleusDataset()
dataset_train.load_TissueNetNucleus(train_dir)
dataset_train.prepare()

# Validation dataset
dataset_val = TissueNetNucleusDataset()
dataset_val.load_TissueNetNucleus(val_dir)
dataset_val.prepare()

# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])



config = TissueNetNucleusConfig()
# config.display() 
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# Load weights
weights="coco"
if weights.lower() == "coco":
    weights_path = COCO_WEIGHTS_PATH
elif weights.lower() == "last":
    # Find last trained weights
    weights_path = model.find_last()
elif weights.lower() == "imagenet":
    # Start from ImageNet trained weights
    weights_path = model.get_imagenet_weights()

print("Loading weights ", weights_path)
if weights.lower() == "coco":
    # Exclude the last layers because they require a matching of classes
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)



# *** This training schedule is an example. Update to your needs ***

# If starting from imagenet, train heads only for a bit
# since they have random weights
print("Train network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            augmentation=augmentation,
            layers='heads')





print("Train all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            augmentation=augmentation,
            layers='all')    



import datetime
t2=datetime.datetime.now()
print("time passed: "+str(t2-t1))


# # In[ ]:


# def rle_encode(mask):
#     """Encodes a mask in Run Length Encoding (RLE).
#     Returns a string of space-separated values.
#     """
#     assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
#     # Flatten it column wise
#     m = mask.T.flatten()
#     # Compute gradient. Equals 1 or -1 at transition points
#     g = np.diff(np.concatenate([[0], m, [0]]), n=1)
#     # 1-based indicies of transition points (where gradient != 0)
#     rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
#     # Convert second index in each pair to lenth
#     rle[:, 1] = rle[:, 1] - rle[:, 0]
#     return " ".join(map(str, rle.flatten()))


# def rle_decode(rle, shape):
#     """Decodes an RLE encoded list of space separated
#     numbers and returns a binary mask."""
#     rle = list(map(int, rle.split()))
#     rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
#     rle[:, 1] += rle[:, 0]
#     rle -= 1
#     mask = np.zeros([shape[0] * shape[1]], bool)
#     for s, e in rle:
#         assert 0 <= s < mask.shape[0]
#         assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
#         mask[s:e] = 1
#     # Reshape and transpose
#     mask = mask.reshape([shape[1], shape[0]]).T
#     return mask


# def mask_to_rle(image_id, mask, scores):
#     "Encodes instance masks to submission format."
#     assert mask.ndim == 3, "Mask must be [H, W, count]"
#     # If mask is empty, return line with image ID only
#     if mask.shape[-1] == 0:
#         return "{},".format(image_id)
#     # Remove mask overlaps
#     # Multiply each instance mask by its score order
#     # then take the maximum across the last dimension
#     order = np.argsort(scores)[::-1] + 1  # 1-based descending
#     mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
#     # Loop over instance masks
#     lines = []
#     for o in order:
#         m = np.where(mask == o, 1, 0)
#         # Skip if empty
#         if m.sum() == 0.0:
#             continue
#         rle = rle_encode(m)
#         lines.append("{}, {}".format(image_id, rle))
#     return "\n".join(lines)



# # In[ ]:


# class TissueNetNucleusInferenceConfig(TissueNetNucleusConfig):
#     # Set batch size to 1 to run one image at a time
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     # Don't resize imager for inferencing
#     IMAGE_RESIZE_MODE = "pad64"
#     # Non-max suppression threshold to filter RPN proposals.
#     # You can increase this during training to generate more propsals.
#     RPN_NMS_THRESHOLD = 0.7

    
# # eval
# config_i = TissueNetNucleusInferenceConfig()
# # config_i.display()
# model_i = modellib.MaskRCNN(mode="inference", config=config_i, model_dir=DEFAULT_LOGS_DIR)

# dataset_dir=val_dir

# print("Running on {}".format(dataset_dir))

# # Read dataset
# dataset = TissueNetNucleusDataset()
# dataset.load_TissueNetNucleus(dataset_dir)
# dataset.prepare()

# for image_id in dataset.image_ids:
#     # Load image and run detection
#     image = dataset.load_image(image_id)
#     # Detect objects
#     r = model_i.detect([image], verbose=0)[0]
#     print(r)
# #     #Encode image to RLE. Returns a string of multiple lines
# #     source_id = dataset.image_info[image_id]["id"]
# #     rle = mask_to_rle(source_id, r["masks"], r["scores"])
# #     # Save image with masks
# #     visualize.display_instances(
# #         image, r['rois'], r['masks'], r['class_ids'],
# #         dataset.class_names, r['scores'],
# #         show_bbox=False, show_mask=False,
# #         title="Predictions")
# #     plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

