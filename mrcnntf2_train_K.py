"""
Using Mask R-CNN for Cell Segmentation

Licensed under the MIT License (see LICENSE for details)

Written by Waleed Abdulla
https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py

Modified by Carsen Stringer for general cell segmentation datasets (12/2019)
https://github.com/MouseLand/cellpose/blob/main/paper/1.0/train_maskrcnn.py

Modified by Youyi Fong (12/2022)
"""

# without this, it hangs when workers>0. see https://pythonspeed.com/articles/python-multiprocessing/
# when using this, there may be an attribute error https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
from multiprocessing import set_start_method
set_start_method("spawn")


# Import mrcnn libraries from the following
mrcnn_path='../Mask_RCNN-TF2'
import sys, os
assert os.path.exists(mrcnn_path), 'mrcnn_path does not exist: '+mrcnn_path
sys.path.insert(0, mrcnn_path) 

import datetime
t1=datetime.datetime.now()


import sys, datetime, glob
from imgaug import augmenters as iaa
#from stardist import matching


from mrcnntf2_CellsegDataset import CellsegDataset
from mrcnntf2_config_stringer import StringerConfig

from mrcnn import utils
from mrcnn import model as modellib


# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

basedir = './' 

DEFAULT_LOGS_DIR = os.path.join(basedir, "logs")
if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)

MODELS_DIR = os.path.join(basedir, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

COCO_WEIGHTS_PATH = os.path.join(".", "mask_rcnn_coco.h5")


############################################################
#  Training
############################################################

def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = CellsegDataset()
    dataset_train.load_data(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellsegDataset()
    dataset_val.load_data(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.5, 1.5)),
        #iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    
    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.HEAD_EPOCHS,
                augmentation=augmentation,
                layers='heads',
                )

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.TRAIN_EPOCHS,
                augmentation=augmentation,
                layers='all',
                )



############################################################
#  Command Line
############################################################

if __name__ == '__main__':

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Mask R-CNN for cell counting and segmentation')
    
    # train with cellpose images
    # parser.add_argument('--dataset', required=False, default="/fh/fast/fong_y/cellpose_images/train", metavar="/path/to/dataset/", help='Root directory of the dataset')
    # parser.add_argument('--weights', required=False, default="imagenet", metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
    
    # train with K's images
    parser.add_argument('--dataset', required=False, default="images/training_resized", metavar="/path/to/dataset/", help='Root directory of the dataset')
    parser.add_argument('--weights', required=False, default="../CellSeg/src/modelFiles/final_weights.h5", metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
    # CellSeg weights cannot be used as the starting model weight

    parser.add_argument('--LR', default=0.001, type=float, required=False, metavar="learning rate", help="initial learning rate")
    parser.add_argument('--nepochs', default = 10, type=int, help='number of epochs')
    parser.add_argument('--nepochs_head', default = 0, type=int, help='number of head epochs')
    parser.add_argument('--batch_size', default = 1, type=int, help='batch_size')
    
    parser.add_argument('--gpu_id', default = 0, type=int, help='which gpu to run on')
    parser.add_argument('--num_cpus', default = 0, type=int, help='number of additional cpus to use. In model.py, the variable is workers')
    args = parser.parse_args()

    # set which gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
        
    batch_size = args.batch_size
    learning_rate = args.LR

    # Validate arguments
    assert args.dataset, "Argument --dataset is required"
    
    fs = glob.glob(os.path.join(args.dataset, '*_img.png'))
    ntrain = len(fs)
    print('ntrain %d'%(ntrain))
    
    # Configurations
    config = StringerConfig()
    # Name needs to be a single word because it will be used to create sub-directories under MODELS_DIR
    config.NAME = "Ktrain" 
    config.CPU_COUNT = args.num_cpus
    config.BATCH_SIZE = batch_size
    config.IMAGES_PER_GPU = batch_size
    config.LEARNING_RATE = learning_rate
    config.HEAD_EPOCHS = args.nepochs_head
    config.TRAIN_EPOCHS = args.nepochs
    config.STEPS_PER_EPOCH = ntrain // config.IMAGES_PER_GPU
    config.VALIDATION_STEPS = 1

    config.IMAGE_SHAPE = [512,512,3]

    config.BACKBONE                       = "resnet101" # changed from resnet50
    config.MEAN_PIXEL                     = [123.7, 116.8, 103.9] # changed from [43.53 39.56 48.22]
    config.DETECTION_MIN_CONFIDENCE       = 0.7 # changed from 0
    
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODELS_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
#        # Download weights file
#        if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)


    # train model
    train(model, args.dataset)
    weights_path = model.checkpoint_path.format(epoch=model.epoch)
    print(weights_path)
    

    # import datetime
    t2=datetime.datetime.now()
    print("time passed: "+str(t2-t1))

    