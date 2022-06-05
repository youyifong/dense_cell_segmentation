"""
Modified from https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/Mesmer_training_notebook.ipynb
To train Mesmer on TissueNet with single-channes input. 

Note
- First run "ml Anaconda3; ml CUDA; ml cuDNN" on Linux
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

from deepcell import image_generators
from deepcell.utils import train_utils

from deepcell.utils.data_utils import get_data
from skimage.segmentation import relabel_sequential

from deepcell.model_zoo.panopticnet import PanopticNet

from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler

from tensorflow.python.keras.losses import MSE

from deepcell import losses
from timeit import default_timer
from deepcell.utils.train_utils import get_callbacks
from deepcell.utils.train_utils import count_gpus

import tensorflow as tf



### Set directory
#experiment_folder = "mesmer_tissuenet"
#MODEL_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", experiment_folder)
#NPZ_DIR = "/fh/fast/fong_y/tissuenet_1.0/"
#LOG_DIR = '/fh/fast/fong_y/tissuenet_1.0/mesmer/logs/'
username = os.getlogin()
MODEL_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", username)
NPZ_DIR = "/fh/fast/fong_y/tissuenet_1.0/"
LOG_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", username, 'logs')

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)


### Load train and validation images

npz_name = "tissuenet_v1.0_"
train_dict = np.load(NPZ_DIR + npz_name + "train.npz")
val_dict = np.load(NPZ_DIR + npz_name + "val.npz")
X_train, y_train = train_dict['X'], train_dict['y'] # for X, 1st channel is nuclear, 2nd channel is cyto; for y, 1st is cell, 2nd is nuclear
X_val, y_val = val_dict['X'], val_dict['y']

# remove nuclear
# expand_dims(-1) add a fourth dimension
X_train=np.expand_dims(X_train[:,:,:,1], -1) 
y_train=np.expand_dims(y_train[:,:,:,0], -1) 
X_val  =np.expand_dims(X_val  [:,:,:,1], -1) 
y_val  =np.expand_dims(y_val  [:,:,:,0], -1) 


### Train model

new_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 1),
    norm_method=None,
    num_semantic_heads=2,
    num_semantic_classes=[1, 3], # inner distance, pixelwise
    location=True,  # should always be true
    include_top=True)


model_name = npz_name + 'deep_watershed'
n_epoch =500
optimizer = Adam(lr=1e-4, clipnorm=0.001)
lr_sched = rate_scheduler(lr=1e-4, decay=0.99)
batch_size = 8
min_objects = 0  # throw out images with fewer than this many objects
seed=0
model_name


# Define data generators

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


### Define loss (create a dictionary of losses for each semantic head)

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


tf.keras.models.save_model (new_model,
    filepath="deepcell_cyto",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
    save_traces=True)

