import datetime
t1=datetime.datetime.now()


train_dir="/fh/fast/fong_y/tissuenet_v1.0/images/train_nuclear"
val_dir="/fh/fast/fong_y/tissuenet_v1.0/images/val_nuclear"
test_dir="/home/yfong/deeplearning/dense_cell_segmentation/images/test_images"
#jacs20221127T1918/mask_rcnn_jacs_0001.h5

import matplotlib.pyplot as plt
import os
import sys
import glob
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa



# Root directory of the alsombra project
# that version leads to a warning about multiprocessing and the process hung
# sys.path.insert(0, os.path.abspath("../Mask_RCNN_alsombra/"))  # To find local version of the library

# Import Mask RCNN from module
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize




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

submit_dir = os.path.join(RESULTS_DIR, "submit/")
if not os.path.isdir(submit_dir):
    os.mkdir(submit_dir)


class JACSDataset(utils.Dataset):

    def load_JACS(self, dataset_dir):
        """Load a subset of the dataset.

        dataset_dir: Root directory of the dataset
        subset_dir: Subset directory
        """
        # Add classes. We have one class.
        # Naming the dataset JACS, and the class JACS
        self.add_class("JACS", 1, "JACS")

        image_ids = sorted(glob.glob(os.path.join(dataset_dir, '*_img.*')))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "JACS",
                image_id=image_id, #os.path.splitext(os.path.basename(image_id))
                path=image_id)

            
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        info = self.image_info[image_id]
            
        m = skimage.io.imread(info['path'].replace("img","masks"))
        obj_ids = np.unique(m) # get list of gt masks, e.g. [0,1,2,3,...]
        obj_ids = obj_ids[1:] # remove background 0
        mask = m == obj_ids[:, None, None] # masks contain multiple binary mask map
        mask = mask.transpose(tuple(np.array([1,2,0])))

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "JACS":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


class JACSConfig(Config):
    """Configuration for training on cell segmentation datasets."""
    # Give the configuration a recognizable name
    NAME = "JACS"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + JACS

    GPU_COUNT=1 # setting this to 2 leads to a RuntimeError: It looks like you are subclassing `Model` and you forgot to call `super(YourClass, self).__init__()`. Always start with this line.
    
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6 # this is about 10% faster than 1. 10 has not been timed, but seems slower than 6

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH  = 4162//6
    VALIDATION_STEPS = 1040//6
    
    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between JACS and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


# #####################################################################################
# # Training dataset.
# dataset_train = JACSDataset()
# dataset_train.load_JACS(train_dir)
# dataset_train.prepare()

# # Validation dataset
# dataset_val = JACSDataset()
# dataset_val.load_JACS(val_dir)
# dataset_val.prepare()

# # Image augmentation
# # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
# augmentation = iaa.SomeOf((0, 2), [
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
#     iaa.OneOf([iaa.Affine(rotate=90),
#                iaa.Affine(rotate=180),
#                iaa.Affine(rotate=270)]),
#     iaa.Multiply((0.8, 1.5)),
#     iaa.GaussianBlur(sigma=(0.0, 5.0))
# ])



# config = JACSConfig()
# # config.display() 
# model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# # Load weights
# weights="coco"
# if weights.lower() == "coco":
#     weights_path = COCO_WEIGHTS_PATH
# elif weights.lower() == "last":
#     # Find last trained weights
#     weights_path = model.find_last()
# elif weights.lower() == "imagenet":
#     # Start from ImageNet trained weights
#     weights_path = model.get_imagenet_weights()

# print("Loading weights ", weights_path)
# if weights.lower() == "coco":
#     # Exclude the last layers because they require a matching of classes
#     model.load_weights(weights_path, by_name=True, exclude=[
#         "mrcnn_class_logits", "mrcnn_bbox_fc",
#         "mrcnn_bbox", "mrcnn_mask"])
# else:
#     model.load_weights(weights_path, by_name=True)



# # *** This training schedule is an example. Update to your needs ***

# # If starting from imagenet, train heads only for a bit
# # since they have random weights
# print("Train network heads")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=1,
#             augmentation=augmentation,
#             layers='heads')



# print("Train all layers")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=1,
#             augmentation=augmentation,
#             layers='all')    



# import datetime
# t2=datetime.datetime.now()
# print("time passed: "+str(t2-t1))


#####################################################################################

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


class JACSInferenceConfig(JACSConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    
# eval
config_i = JACSInferenceConfig()
# config_i.display()
model_i = modellib.MaskRCNN(mode="inference", config=config_i, model_dir=DEFAULT_LOGS_DIR)


print("Running on {}".format(test_dir))

# Read dataset
dataset = JACSDataset()
dataset.load_JACS(test_dir)
dataset.prepare()

for image_id in dataset.image_ids:
    # Load image and run detection
    image = dataset.load_image(image_id)
    # Detect objects
    r = model_i.detect([image], verbose=0)[0]
    #Encode image to RLE. Returns a string of multiple lines
    source_id = dataset.image_info[image_id]["id"]
    rle = mask_to_rle(source_id, r["masks"], r["scores"])
    # Save image with masks
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'],
        dataset.class_names, r['scores'],
        show_bbox=False, show_mask=False,
        title="Predictions")
    plt.savefig("{}/{}.png".format(submit_dir, image_id))

