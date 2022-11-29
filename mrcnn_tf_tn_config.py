import os, sys, datetime, glob, pdb
import numpy as np
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config
import skimage.io


############################################################
#  Configurations
############################################################

class TissueNetNucleusConfig(Config):
    """Configuration for training on cell segmentation datasets."""

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + TissueNetNucleus

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between TissueNetNucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing. Random crops of size 512x512
    # for tissuenet, since the images are 512x512, no random cropping happens
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

    # How many anchors per image to use for RPN training. Updated according to Stringer
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1500

    # Image mean (RGB). Stringer kept this unchanged from nucleus, not sure why it
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images. Updated according to Stringer
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold. Updated according to Stringer
    TRAIN_ROIS_PER_IMAGE = 300

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    # setting this to 2 leads to a RuntimeError: 
    # It looks like you are subclassing `Model` and you forgot to call 
    # `super(YourClass, self).__init__()`. 
    GPU_COUNT=1 
    
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6 # this is about 10% faster than 1. 
    # 10 has not been timed, but seems slower than 6

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH  = 4162//6
    VALIDATION_STEPS = 1040//6
    
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    # tissuenet nuclear images are grayscale, 
    # but it is a bit of work to make all the necessary code changes
    IMAGE_CHANNEL_COUNT = 3
    

class TissueNetNucleusDataset(utils.Dataset):

    def load_Nucleus(self, dataset_dir):
        """add image file names.
        Modified so that training and validation are from separate folders

        dataset_dir: Root directory of the dataset
        """
        # Add classes. We have one class.
        # Naming the dataset TissueNetNucleus, and the class TissueNetNucleus
        self.add_class("TissueNetNucleus", 1, "TissueNetNucleus")
        
        image_ids = sorted(glob.glob(os.path.join(dataset_dir, '*_img.*')))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "TissueNetNucleus",
                # note that this image_id is not the internal image id used by utils.Datatset, the latter is an integer
                # for example, load_mask(image_id) uses integer image_id
                image_id=image_id, #original: os.path.splitext(os.path.basename(image_id))
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
        if info["source"] == "TissueNetNucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


