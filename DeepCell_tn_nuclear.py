#!/usr/bin/env python
# coding: utf-8

# **Traing DeepCell Model and Make Predictions**<br>
# It trains a model with three heads: inner distance, outer distance, and fgbg. It works on tensorflow 2.7.1.<br>
# Tissuenet V1.0 dataset is used, which has images of size 512x512 in the "Intro to DeepCell" are 256x256. In contrast, the pretrained model in NuclearApplication was trained on images of 512x512.<br>
# Training can also be done via deepcell.training.train_model_sample, which allows arbitrary size images and uses window_size to control patch size.

# In[ ]:


import syotil

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from timeit import default_timer

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MSE

import deepcell
from deepcell.utils.train_utils import rate_scheduler, get_callbacks, count_gpus
from deepcell.losses import weighted_categorical_crossentropy
from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell import image_generators

print(tf.__version__)


# In[2]:


train_dict = np.load('/fh/fast/fong_y/tissuenet_v1.0/tissuenet_v1.0_train.npz')
train_X, train_y = train_dict['X'], train_dict['y']

# # val is 256x256, thus not used
# val_dict = np.load('/fh/fast/fong_y/tissuenet_v1.0/tissuenet_v1.0_val.npz')
# X_val, y_val = val_dict['X'], val_dict['y']


# In[119]:


model_name = 'tn1.0_nuclear'; idx_X=0; idx_y=1
model_path = '{}.h5'.format(model_name)
#model_name = 'tn1.0_cyto'; idx.X=1; idx.y=0

seed = 0 
n=train_X.shape[0]

min_objects = 2

val_size = 0.2 # fraction of data saved as validation

import random
tmp = random.sample(range(n), int(val_size*n))

X_val = np.expand_dims(train_X[tmp,:,:,idx_X], axis=-1)
y_val = np.expand_dims(train_y[tmp,:,:,idx_y], axis=-1)

tmp1 = list(set(range(n)).difference (set(tmp)))
X_train = np.expand_dims(train_X[tmp1,:,:,idx_X], axis=-1)
y_train = np.expand_dims(train_y[tmp1,:,:,idx_y], axis=-1)

print('X_train.shape: {}\nX_val.shape: {}'.format(
    X_train.shape, X_val.shape))
transforms = ['inner-distance', 'outer-distance', 'fgbg']
transforms_kwargs = {'outer-distance': {'erosion_width': 0}}


# use augmentation for training but not validation
datagen = image_generators.SemanticDataGenerator(
    rotation_range=180,
    fill_mode='reflect',
    zoom_range=(0.75, 1.25),
    horizontal_flip=True,
    vertical_flip=True)

datagen_val = image_generators.SemanticDataGenerator()

batch_size = 4 # 8 causes memory outage

train_data = datagen.flow(
    {'X': X_train, 'y': y_train},
    seed=seed,
    transforms=transforms,
    transforms_kwargs=transforms_kwargs,
    min_objects=min_objects,
    batch_size=batch_size)

val_data = datagen_val.flow(
    {'X': X_val, 'y': y_val},
    seed=seed,
    transforms=transforms,
    transforms_kwargs=transforms_kwargs,
    min_objects=min_objects,
    batch_size=batch_size)


# In[113]:


i=5
plt.subplot(1, 2, 1) # row 1, col 2 index 1
io.imshow(X_train[i,:,:,0])
plt.subplot(1, 2, 2) # row 1, col 2 index 1
# tmp = syotil.masks_to_outlines(y_train[i,:,:,0]); io.imshow(tmp)
io.imshow(y_train[i,:,:,0])
plt.show()


# **The two cells below train the model and can be skipped if trained model will be loaded.**

# In[114]:


semantic_classes = [1, 1, 2] # inner distance, outer distance, fgbg

model = PanopticNet(
    backbone='resnet50',
    input_shape=X_train.shape[1:],
    norm_method='whole_image',
    num_semantic_classes=semantic_classes)

lr = 1e-4
optimizer = Adam(lr=lr, clipnorm=0.001)
lr_sched = rate_scheduler(lr=lr, decay=0.99)

# Create a dictionary of losses for each semantic head

def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes)
        return MSE(y_pred, y_true)
    return _semantic_loss

loss = {}

# Give losses for all of the semantic heads
for layer in model.layers:
    if layer.name.startswith('semantic_'):
        n_classes = layer.output_shape[-1]
        loss[layer.name] = semantic_loss(n_classes)
        
model.compile(loss=loss, optimizer=optimizer)

[(layer.name, layer.output_shape) for layer in filter(lambda x: x.name.startswith('semantic_'), model.layers)]


# In[120]:


# fit the model
print('Training on', count_gpus(), 'GPUs.')

train_callbacks = get_callbacks(
    model_path,
    lr_sched=lr_sched,
    monitor='val_loss',
    verbose=1)

loss_history = model.fit(
    train_data,
    steps_per_epoch=train_data.y.shape[0] // batch_size,
    epochs=60, # about 6 epochs/hr
    validation_data=val_data,
    validation_steps=val_data.y.shape[0] // batch_size,
    callbacks=train_callbacks)


# <B>Make predictions on Nuclear test dataset.</B> 

# In[121]:


# create prediction model

prediction_model = PanopticNet(
    backbone='resnet50',
    norm_method='whole_image',
    num_semantic_classes=[1, 1], # inner distance, outer distance
    input_shape=X_val.shape[1:]
)

prediction_model.load_weights(model_path, by_name=True)


# In[122]:


# make predictions on validation data

start = default_timer()
test_images = prediction_model.predict(X_val)
watershed_time = default_timer() - start

print('Watershed segmentation of shape', test_images[0].shape,
      'in', watershed_time, 'seconds.')

masks = deep_watershed(
    test_images,
    min_distance=10,
    detection_threshold=0.1,
    distance_threshold=0.01,
    exclude_border=False,
    small_objects_threshold=0)


# In[123]:


io.imshow(X_val[1,:,:,0])
plt.show()
io.imshow(y_val[1,:,:,0])
plt.show()
print(X_val.shape)
print(y_val.shape)


# In[124]:


APs = [syotil.csi(y_val[i,:,:,0], masks[i,:,:,0]) for i in range(y_val.shape[0])]
print(np.nanmean(APs))


# **Make prediction on K's data.**<br>
# Using NuclearSegmentation allows setting image_mpp, which has a substantial influence on performance.

# In[125]:


from deepcell.applications import NuclearSegmentation

app = NuclearSegmentation(prediction_model)
[(layer.name, layer.output_shape) for layer in filter(lambda x: x.name.startswith('semantic_'), app.model.layers)]


# In[126]:


# a rectangular patch that requires tiling
im0 = io.imread('images/test/M872956_JML_Position8_CD3_test_img.png')
print(im0.shape)
im=im0
io.imshow(im)
plt.show()
im = np.expand_dims(im, axis=-1)
im = np.expand_dims(im, axis=0)

mask_true=io.imread("images/test/M872956_JML_Position8_CD3_test_masks.png")
print(mask_true.shape)
io.imshow(mask_true)
plt.show()


# In[127]:


app.model_image_shape


# In[128]:


x=im
y, tile_info = app._tile_input(im)
print(x.shape)
print(y.shape)
print(tile_info)
pred = app.predict(y, image_mpp=2)
prd = app._untile_output(pred, tile_info)
io.imshow(prd[0,:,:,0])
plt.show()
syotil.csi(mask_true, prd[0,:,:,0])
# image_mpp=1: .2
# image_mpp=2: .3
# image_mpp=3 .1


# **Additional testing with a square patch.**

# In[ ]:


# read CD3 data

img=io.imread("images/square_patches/M872956_JML_Position8_CD3_img_patch256x256.png")
print(img.shape)
X_cd3=img[:,:,2]
#io.imshow(im)
#plt.show()
X_cd3=np.expand_dims(X_cd3, -1) 
X_cd3=np.expand_dims(X_cd3, 0) 

y_cd3=io.imread("images/square_patches/M872956_JML_Position8_CD3_masks_patch256x256.png")
print(y_cd3.shape)
#io.imshow(mask_true)
#plt.show()
y_cd3=np.expand_dims(y_cd3, -1) 
y_cd3=np.expand_dims(y_cd3, 0) 

batch_size_cd3 = 1

print(X_cd3.shape)
print(y_cd3.shape)

train_cd3 = datagen.flow(
    {'X': X_cd3, 'y': y_cd3},
    seed=seed,
    transforms=transforms,
    transforms_kwargs=transforms_kwargs,
    min_objects=min_objects,
    batch_size=batch_size)

val_cd3 = datagen_val.flow(
    {'X': X_cd3, 'y': y_cd3},
    seed=seed,
    transforms=transforms,
    transforms_kwargs=transforms_kwargs,
    min_objects=min_objects,
    batch_size=batch_size)


# In[ ]:


y, tile_info = app._tile_input(X_cd3)
print(y.shape)
#print(tile_info)
pred = app.predict(y, image_mpp=1)  #, image_mpp=1 0.299, with default image_mpp 0.65, it is 0.09
prd = app._untile_output(pred, tile_info)
io.imshow(prd[0,:,:,0])
plt.show()
syotil.csi(y_cd3[0,:,:,0], prd[0,:,:,0]) # 


# **Predictions can also be made without Application.**<br>
# Don't set min_distance to 0.

# In[ ]:


# make predictions on CD3 data
start = default_timer()
test_images = prediction_model.predict(X_cd3)
watershed_time = default_timer() - start

pred_cd3 = deep_watershed(
    test_images,
    min_distance=10, # this should not be set to 0, https://github.com/vanvalenlab/deepcell-toolbox/blob/master/deepcell_toolbox/deep_watershed.py
    detection_threshold=0.001,
    distance_threshold=0.0,
    exclude_border=False,
    small_objects_threshold=0)

#print(y_cd3.shape, pred_cd3.shape)
# io.imshow(y_cd3[0,:,:,0])
# plt.show()
io.imshow(pred_cd3[0,:,:,0])
plt.show()

syotil.csi(y_cd3[0,:,:,0], pred_cd3[0,:,:,0])

