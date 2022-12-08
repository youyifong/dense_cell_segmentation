"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla    


Train mrcnn-tf2 with Kaggle 2018 Data Science Bowl data

Lee et al. (CellSeg) starts with coco.
"""

import datetime
t1=datetime.datetime.now()

# Import mrcnn libraries from the following
mrcnn_path='../Mask_RCNN-TF2'
import sys, os
assert os.path.exists(mrcnn_path), 'mrcnn_path does not exist: '+mrcnn_path
sys.path.insert(0, mrcnn_path) 

from imgaug import augmenters as iaa

from mrcnn import utils
from mrcnn import model as modellib

from mrcnntf2_dataset_Kaggle2018 import Kaggle2018Dataset
from mrcnntf2_config_CellSeg import CellSegConfig

COCO_WEIGHTS_PATH = "../mask_rcnn_coco.h5"
DEFAULT_MODELS_DIR = "models"


if __name__ == '__main__':
    
    # This part has to be udner __main__
    # without this, it hangs when workers>0. see https://pythonspeed.com/articles/python-multiprocessing/
    # when using this, there may be an attribute error https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
    from multiprocessing import set_start_method
    set_start_method("spawn")
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument('--gpu_id', default = 2, type=int, help='which gpu to run on')
    parser.add_argument('--dataset', required=False, default="/fh/fast/fong_y/Kaggle_2018_Data_Science_Bowl_Stage1/", metavar="/path/to/dataset/")
    parser.add_argument('--weights', required=False, default="coco", metavar="/path/to/weights.h5")
    args = parser.parse_args()
    # set which gpu to use
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    print("Run on GPU: ", args.gpu_id)
    print("Dataset: ", args.dataset)
    print("Weights: ", args.weights)


    # Config and model
    config = CellSegConfig()
    config.NAME = "Kaggle" # used in naming model directory
    config.CPU_COUNT = 10    
    # Number of training and validation steps per epoch
    # hard code these numbers for Kaggle dataset
    config.STEPS_PER_EPOCH = (670 - 25) // config.IMAGES_PER_GPU
    config.VALIDATION_STEPS = max(1, 25 // config.IMAGES_PER_GPU)

    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_MODELS_DIR)

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

    
    # Prepare dataset
    dataset_train = Kaggle2018Dataset()
    dataset_train.load_nucleus(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = Kaggle2018Dataset()
    dataset_val.load_nucleus(args.dataset, "val")
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


    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                augmentation=augmentation,
                layers='all')

    t2=datetime.datetime.now()
    print("time passed: "+str(t2-t1))
