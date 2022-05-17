"""
This python script, comes from (https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/Mesmer_training_notebook.ipynb), is to train Mesmer on TissueNet with two-channels input. 

Note
- First run "ml Anaconda3; ml CUDA" on Linux
- "nohup" can be used, but seems to be stopped after log-off Volta (the linux command "ps xw" shows current working jobs)


To train mesmer (on Linux):
nohup python training_mesmer.py
"""


### Library
import os
import errno
import numpy as np 
import deepcell
from deepcell_toolbox.processing import phase_preprocess
from deepcell.applications import MultiplexSegmentation


### Set directory
experiment_folder = "mesmer_tissuenet"
MODEL_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", experiment_folder)
NPZ_DIR = "/fh/fast/fong_y/tissuenet_1.0/"
LOG_DIR = '/fh/fast/fong_y/tissuenet_1.0/mesmer/logs/'

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)


### Load train and validation images
from deepcell.utils.data_utils import get_data
from skimage.segmentation import relabel_sequential

npz_name = "tissuenet_v1.0_"
train_dict = np.load(NPZ_DIR + npz_name + "train.npz")
val_dict = np.load(NPZ_DIR + npz_name + "val.npz")
X_train, y_train = train_dict['X'], train_dict['y'] # for X, 1st channel is nuclear, 2nd channel is cyto; for y, 1st is cell, 2nd is nuclear
X_val, y_val = val_dict['X'], val_dict['y']


### Display an example of images
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
import copy
def make_color_overlay(input_data):
    """Create a color overlay from 2 channel image data
    Args:
        input_data: stack of input images
    Returns:
        numpy.array: color-adjusted stack of overlays in RGB mode
    """
    RGB_data = np.zeros(input_data.shape[:3] + (3, ), dtype='float32')
    # rescale channels to aid plotting
    for img in range(input_data.shape[0]):
        for channel in range(input_data.shape[-1]):
            # get histogram for non-zero pixels
            percentiles = np.percentile(input_data[img, :, :, channel][input_data[img, :, :, channel] > 0],
                                            [5, 95])
            rescaled_intensity = rescale_intensity(input_data[img, :, :, channel],
                                                       in_range=(percentiles[0], percentiles[1]),
                                                       out_range='float32')
            RGB_data[img, :, :, channel + 1] = rescaled_intensity        
    # create a blank array for red channel
    return RGB_data

def make_outline_overlay(RGB_data, predictions):
    boundaries = np.zeros_like(predictions)
    overlay_data = copy.copy(RGB_data)
    for img in range(predictions.shape[0]):
        boundary = find_boundaries(predictions[img, :, :], connectivity=1, mode='inner')
        boundaries[img, boundary > 0] = 1    
    overlay_data[boundaries > 0, :] = 1    
    return overlay_data

rgb_data = make_color_overlay(X_train[2000:2030])
cell_overlay = make_outline_overlay(rgb_data, y_train[2000:2030, :, :, 0]) # for y, 1st channel is cell, 2nd channel is nuclear
nuc_overlay = make_outline_overlay(rgb_data, y_train[2000:2030, :, :, 1]) # for y, 1st channel is cell, 2nd channel is nuclear

import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
cmap.set_bad('black')
index = 27

#fig, axes = plt.subplots(1,2,figsize=(30,20))
#axes = axes.flatten()
#axes[0].imshow(nuc_overlay[index, ...], cmap=cmap)
#axes[0].set_title('Nuclear Overlay', fontsize=24)
#axes[1].imshow(cell_overlay[index, ...], cmap=cmap)
#axes[1].set_title('Cell Overlay', fontsize=24)
## axes[5].set_title('Ground Truth Mask', fontsize=24)
#for ax in axes.flatten():
#    ax.set_axis_off()
#plt.show()
#plt.savefig('/home/shan/image_example.png')


### Train model
from deepcell.model_zoo.panopticnet import PanopticNet

new_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 2),
    norm_method=None,
    num_semantic_heads=4,
    num_semantic_classes=[1, 3, 1, 3], # inner distance, pixelwise, inner distance, pixelwise
    location=True,  # should always be true
    include_top=True)

from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler

model_name = npz_name + 'deep_watershed'
n_epoch = 2
optimizer = Adam(lr=1e-4, clipnorm=0.001)
lr_sched = rate_scheduler(lr=1e-4, decay=0.99)
batch_size = 8
min_objects = 0  # throw out images with fewer than this many objects
seed=0
model_name


# Define data generators
from deepcell import image_generators
from deepcell.utils import train_utils

datagen = image_generators.CroppingDataGenerator(
    rotation_range=180,
    shear_range=0,
    zoom_range=(0.7, 1/0.7),
    horizontal_flip=True,
    vertical_flip=True,
    crop_size=(256, 256))

datagen_val = image_generators.SemanticDataGenerator(
    rotation_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=0,
    vertical_flip=0)
    
train_data = datagen.flow(
    {'X': X_train, 'y': y_train},
    seed=seed,
    transforms=['inner-distance','pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1}, 
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'}},
    min_objects=min_objects,
    batch_size=batch_size)

val_data = datagen_val.flow(
    {'X': X_val, 'y': y_val},
    seed=seed,
    transforms=['inner-distance', 'pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1},
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'}},
    min_objects=min_objects,
    batch_size=batch_size)


# Display images, inner distance, pixel-wise transform
from matplotlib import pyplot as plt

inputs, outputs = train_data.next()
img = inputs[0]
inner_distance = outputs[0] # inner distance for cell
pixelwise = outputs[1] # for pixel-wise transform, 1st ch: cell boundary, 2nd ch: cell interior, 3rd ch: background for cell
inner_distance_nuc = outputs[2] # inner distance for nuclear
pixelwise_nuc = outputs[3] # cell interior, cell boundary, and background for nuclear

## For cell
#fig, axes = plt.subplots(1, 4, figsize=(30, 20))
#axes = axes.flatten()
#axes[0].imshow(img[:, :, 0]) # the first channel of img is nuclear
#axes[0].set_title('DNA (nuclear)')
#axes[1].imshow(img[:, :, 1]) # the second channel of img is cytoplasm
#axes[1].set_title('Membrane (cytoplasm)')
#axes[2].imshow(inner_distance[0, ..., 0]) # inner distance for cell
#axes[2].set_title('Inner Distance for cell')
#axes[3].imshow(pixelwise[0, ..., 1]) # cell interior
#axes[3].set_title('Cell interior')
#plt.show()
##plt.savefig('/home/shan/cell_example.png')
#
## For nuclear
#fig, axes = plt.subplots(1, 4, figsize=(30, 20))
#axes = axes.flatten()
#axes[0].imshow(img[:, :, 0])
#axes[0].set_title('DNA')
#axes[1].imshow(img[:, :, 1])
#axes[1].set_title('Membrane')
#axes[2].imshow(inner_distance_nuc[0, ..., 0]) # inner distance for nuclear
#axes[2].set_title('Inner Distance for nuclear')
#axes[3].imshow(pixelwise_nuc[0, ..., 1]) # nuclear interior
#axes[3].set_title('Nuclear interior')
#plt.show()
##plt.savefig('/home/shan/nuclear_example.png')
#
#
### Define loss (create a dictionary of losses for each semantic head)
from tensorflow.python.keras.losses import MSE
from deepcell import losses

def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * losses.weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes)
        return MSE(y_pred, y_true)
    return _semantic_loss

loss = {}

# Give losses for all of the semantic heads
for layer in new_model.layers:
    if layer.name.startswith('semantic_'):
        n_classes = layer.output_shape[-1]
        loss[layer.name] = semantic_loss(n_classes)

new_model.compile(loss=loss, optimizer=optimizer)


### Interate model training
from timeit import default_timer
from deepcell.utils.train_utils import get_callbacks
from deepcell.utils.train_utils import count_gpus

model_path = os.path.join(MODEL_DIR, '{}.h5'.format(model_name))
loss_path = os.path.join(MODEL_DIR, '{}.npz'.format(model_name))

num_gpus = count_gpus()
print('Training on', num_gpus, 'GPUs.')

train_callbacks = get_callbacks(
    model_path,
    lr_sched=lr_sched,
    tensorboard_log_dir=LOG_DIR,
    save_weights_only=num_gpus >= 2,
    monitor='val_loss',
    verbose=1)

start = default_timer()
loss_history = new_model.fit_generator(
    train_data,
    steps_per_epoch=train_data.y.shape[0] // batch_size,
    epochs=n_epoch,
    validation_data=val_data,
    validation_steps=val_data.y.shape[0] // batch_size,
    callbacks=train_callbacks)
training_time = default_timer() - start
print('Training time: ', training_time, 'seconds.')
