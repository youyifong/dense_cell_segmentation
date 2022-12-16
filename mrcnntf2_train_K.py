"""
Using Mask R-CNN for Cell Segmentation

Licensed under the MIT License (see LICENSE for details)

Written by Waleed Abdulla
https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py

Modified by Carsen Stringer for general cell segmentation datasets (12/2019)
https://github.com/MouseLand/cellpose/blob/main/paper/1.0/train_maskrcnn.py

Modified by Youyi Fong (12/2022)
"""


import os
os.environ['PYTHONHASHSEED']='1' # this does not work for python 3.7 or 3.9, has to be set in the terminal
os.environ['TF_DETERMINISTIC_OPS']='1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_CUDNN_USE_AUTOTUNE']='0' # this should not be set to 0

import numpy as np
np.random.seed(0)
import random
random.seed(0)

import tensorflow as tf

# this patch does not work because no patch is available for tf 2.4 https://pypi.org/project/tensorflow-determinism/
# from tfdeterminism import patch
# patch()

#mrcnntf2 model.py has this line. we repeat it here so that we can set_seed after it. 
#setting seeds before it does not work
tf.compat.v1.disable_eager_execution() 
tf.random.set_seed(0)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.compat.v2.random.set_seed(0)

# tf.debugging.set_log_device_placement(True)

import datetime
t1=datetime.datetime.now()

# Import mrcnn libraries from the following
mrcnn_path='../Mask_RCNN-TF2'
import sys, os, glob
assert os.path.exists(mrcnn_path), 'mrcnn_path does not exist: '+mrcnn_path
sys.path.insert(0, mrcnn_path) 

from imgaug import augmenters as iaa

from mrcnn import utils
from mrcnn import model as modellib

from mrcnntf2_dataset_Stringer import StringerDataset
from mrcnntf2_config_CellSeg import CellSegConfig


basedir = './' 

DEFAULT_LOGS_DIR = os.path.join(basedir, "logs")
if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)

MODELS_DIR = os.path.join(basedir, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

COCO_WEIGHTS_PATH = "../mask_rcnn_coco.h5"


if __name__ == '__main__':

    # This part has to be udner __main__
    # without this, it hangs when workers>0. see https://pythonspeed.com/articles/python-multiprocessing/
    # when using this, there may be an attribute error https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
    from multiprocessing import set_start_method
    set_start_method("spawn")

    import argparse
    parser = argparse.ArgumentParser(description='Mask R-CNN for cell counting and segmentation')
    
    # train with cellpose images
    # parser.add_argument('--dataset', required=False, default="/fh/fast/fong_y/cellpose_images/train", metavar="/path/to/dataset/", help='Root directory of the dataset')
    # parser.add_argument('--weights', required=False, default="imagenet", metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
    
    # train with K's images
    # parser.add_argument('--dataset', required=False, default="/fh/fast/fong_y/cellpose_images/tmp", metavar="/path/to/dataset/", help='Root directory of the dataset')
    parser.add_argument('--dataset', required=False, default="images/training_resized_3chan1", metavar="/path/to/dataset/", help='Root directory of the dataset')
    
    parser.add_argument('--weights', required=False, default="models/cellseg20221204T2219/mask_rcnn_cellseg_0030.h5", metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--weights', required=False, default="coco", metavar="/path/to/weights.h5")

    parser.add_argument('--LR', default=0.001, type=float, required=False, metavar="learning rate", help="initial learning rate")
    parser.add_argument('--nepochs', default = 100, type=int, help='number of epochs')    
    parser.add_argument('--gpu_id', default = 0, type=int, help='which gpu to run on')
    args = parser.parse_args()

    # set which gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id) # this works
        
    learning_rate = args.LR

    # Validate arguments
    assert args.dataset, "Argument --dataset is required"
    
    fs = glob.glob(os.path.join(args.dataset, '*_img.png'))
    ntrain = len(fs)
    print('ntrain %d'%(ntrain))
    
    # Configurations
    config = CellSegConfig()
    # Name needs to be a single word because it will be used to create sub-directories under MODELS_DIR
    config.NAME = "Ktrain" 
    config.CPU_COUNT = 10
    # config.BATCH_SIZE =1
    # config.IMAGES_PER_GPU = 1 # needs to be 1
    config.IMAGE_MIN_DIM=128
    config.LEARNING_RATE = learning_rate
    config.STEPS_PER_EPOCH = ntrain // config.IMAGES_PER_GPU
    config.VALIDATION_STEPS = 1
    
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODELS_DIR)
    
    # load weights
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)


    # Training dataset.
    dataset_train = StringerDataset()
    dataset_train.load_data(args.dataset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = StringerDataset()
    dataset_val.load_data(args.dataset)
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
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***
    
    # # skip head training because we are not starting from imagenet    

    print("Train all layers")
    model.train(dataset_train, None,
                learning_rate=config.LEARNING_RATE,
                epochs=args.nepochs+30, #starts at 30
                augmentation=None,
                layers='all',
                )

    # weights_path = model.checkpoint_path.format(epoch=model.epoch)
    

    # import datetime
    t2=datetime.datetime.now()
    print("time passed: "+str(t2-t1))

    