### 1. Training deepcell on TissueNet ###
"""
This python script, which comes from (https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/Mesmer_training_notebook.ipynb), is to train Mesmer on TissueNet with two-channels input.

Note
- First run "ml Anaconda3; ml CUDA; ml cuDNN" on Linux.
- "nohup" can be used, but seems to be stopped after log-off Volta (the linux command "ps xw" shows current working jobs)

To train mesmer (on Linux):
nohup python training_mesmer.py

grabnode --constraint=gizmoj
ml DeepCell/0.11.1-foss-2021b-CUDA-11.4.1
GPU works
"""


# Library
import os
import errno
import numpy as np 
import deepcell
from deepcell_toolbox.processing import phase_preprocess
from deepcell.applications import MultiplexSegmentation
import getpass


# Set directory
username = os.getlogin()
MODEL_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", username)
NPZ_DIR = "/fh/fast/fong_y/tissuenet_1.0/"
LOG_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", username, 'logs')

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# Load train and validation images
from deepcell.utils.data_utils import get_data
from skimage.segmentation import relabel_sequential

npz_name = "tissuenet_v1.0_"
train_dict = np.load(NPZ_DIR + npz_name + "train.npz")
val_dict = np.load(NPZ_DIR + npz_name + "val.npz")
X_train, y_train = train_dict['X'], train_dict['y'] # for X, 1st channel is nuclear, 2nd channel is cyto; for y, 1st is cell, 2nd is nuclear
X_val, y_val = val_dict['X'], val_dict['y']


# Train model
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
n_epoch = 100
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


# Define loss (create a dictionary of losses for each semantic head)
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


# Iterate model training 
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





#####





### 2. Training deepcell on K's images ###
"""
The following codes basically follow the reference notebook, https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/Mesmer_training_notebook.ipynb. We do not use PanopticNet() as model but directly load the pre-trained deepcell model by using tf.keras.models.load_model(), like as Mesmer() application.

Note
- First run "ml Anaconda3; ml CUDA; ml cuDNN" on Linux.
- "nohup" can be used, but seems to be stopped after log-off Volta (the linux command "ps xw" shows current working jobs)

To train mesmer (on Linux):
nohup python training_mesmer.py

grabnode --constraint=gizmoj
ml DeepCell/0.11.1-foss-2021b-CUDA-11.4.1
GPU works
"""

# Library
import os
import errno
import numpy as np 
import deepcell
import getpass
import cv2
import skimage.io as io


# Set directory
username = getpass.getuser()
MODEL_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", username)
LOG_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", username, 'logs')

 
# Import images
# CD3
cd3_img = io.imread('/home/shan/kdata/M872956_Position8_CD3_img.png')
X_train = cd3_img[:,:,2] # blue channel only
X_train = X_train.reshape((1, X_train.shape[0], X_train.shape[1], 1))
y_train = cv2.imread('/home/shan/kdata/M872956_Position8_CD3-BUV395_no_inputs_GTmasks_1908_masks.png', cv2.IMREAD_UNCHANGED)
y_train = y_train.reshape((1, y_train.shape[0], y_train.shape[1], 1))

# CD3+DAPI
cd3_img = io.imread('/home/shan/kdata/M872956_Position8_CD3_img.png')
dapi_img = io.imread('/home/shan/kdata/M872956_Position8_DAPI_img.png')
X_train = np.stack((dapi_img[:,:,2], cd3_img[:,:,2]), axis=2) # image with two channels: DAPI and CD3
X_train = X_train.reshape((1, X_train.shape[0], X_train.shape[1], 2))
y_train = cv2.imread('/home/shan/kdata/M872956_Position8_CD3-BUV395_no_inputs_GTmasks_1908_masks.png', cv2.IMREAD_UNCHANGED)
y_train = y_train.reshape((1, y_train.shape[0], y_train.shape[1], 1)) # two ground-truth masks are needed (one for CD3 and the other for DAPI)


# Training model
# One-channel (CD3)
import tensorflow as tf
MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/CytoplasmSegmentation-3.tar.gz')
MODEL_HASH = '6a244f561b4d37169cb1a58b6029910f'
archive_path = tf.keras.utils.get_file(
                'CytoplasmSegmentation.tgz', MODEL_PATH,
                file_hash=MODEL_HASH,
                extract=True, cache_subdir='models')
model_path = os.path.splitext(archive_path)[0]
new_model = tf.keras.models.load_model(model_path)
#new_model.save_weights('/home/shan/cyto_pretrained_weights.h5')

# Two-channel (CD3+DAPI)
import tensorflow as tf
#MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-9.tar.gz')
#MODEL_HASH = 'a1dfbce2594f927b9112f23a0a1739e0'
MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/MultiplexSegmentation-7.tar.gz')
MODEL_HASH = 'e7360e8e87c3ab71ded00a577a61c689'
archive_path = tf.keras.utils.get_file(
                'MultiplexSegmentation.tgz', MODEL_PATH,
                file_hash=MODEL_HASH,
                extract=True, cache_subdir='models')
model_path = os.path.splitext(archive_path)[0]
new_model = tf.keras.models.load_model(model_path)
#new_model.save_weights('/home/shan/mesmer_pretrained_weights.h5')


from tensorflow.keras.optimizers import SGD, Adam
from deepcell.utils.train_utils import rate_scheduler

model_name = 'cd3_June152022'
n_epoch = 100
optimizer = Adam(learning_rate=1e-4, clipnorm=0.001)
lr_sched = rate_scheduler(lr=1e-4, decay=0.99)
batch_size = 1 # 8
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
    #crop_size=(256, 256)) # generate error
    crop_size=(512, 512))

#datagen_val = image_generators.SemanticDataGenerator(
#    rotation_range=0,
#    shear_range=0,
#    zoom_range=0,
#    horizontal_flip=0,
#    vertical_flip=0)
    
train_data = datagen.flow(
    {'X': X_train, 'y': y_train},
    seed=seed,
    transforms=['inner-distance','pixelwise'],
    transforms_kwargs={'pixelwise':{'dilation_radius': 1}, 
                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'}},
    min_objects=min_objects,
    batch_size=batch_size)

#val_data = datagen_val.flow(
#    {'X': X_val, 'y': y_val},
#    seed=seed,
#    transforms=['inner-distance', 'pixelwise'],
#    transforms_kwargs={'pixelwise':{'dilation_radius': 1},
#                      'inner-distance': {'erosion_width': 1, 'alpha': 'auto'}},
#    min_objects=min_objects,
#    batch_size=batch_size)


# Define loss (create a dictionary of losses for each semantic head)
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


# Iterate model training 
from timeit import default_timer
from deepcell.utils.train_utils import get_callbacks
from deepcell.utils.train_utils import count_gpus

model_path = os.path.join(MODEL_DIR, '{}.h5'.format(model_name))
loss_path = os.path.join(MODEL_DIR, '{}.npz'.format(model_name))

#num_gpus = count_gpus()
#print('Training on', num_gpus, 'GPUs.')

train_callbacks = get_callbacks(
    model_path,
    lr_sched=lr_sched,
    tensorboard_log_dir=LOG_DIR,
    save_weights_only=True,
    #save_weights_only=num_gpus >= 2,
    #monitor='val_loss',
    monitor='loss', # training loss
    verbose=1)

start = default_timer()
loss_history = new_model.fit_generator(
    train_data,
    steps_per_epoch=train_data.y.shape[0] // batch_size,
    epochs=n_epoch,
    #validation_data=val_data,
    #validation_steps=val_data.y.shape[0] // batch_size,
    callbacks=train_callbacks)
training_time = default_timer() - start
print('Training time: ', training_time, 'seconds.')

# The following is the other way to save the trained model, but this way does not work well in loading the trained model (all predicted masks are 0)
#tf.keras.models.save_model (new_model,
#    filepath="/home/shan/deepcell_cd3",
#    overwrite=True,
#    include_optimizer=True,
#    save_format=None,
#    signatures=None,
#    options=None,
#    save_traces=True)
