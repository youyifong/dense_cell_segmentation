# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:00:47 2022

@author: Youyi
"""

# to run this script, first load the module:
# ml DeepCell/0.11.1-foss-2021b-CUDA-11.4.1


# modified from the example in the comments of deepcell/applications/nuclear_segmentation.py

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
%matplotlib inline
# needed to get download model files to work; otherwise will get ssl error
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context
import syotil

#from deepcell.applications import CytoplasmSegmentation
# a new clone does not work on 10/6/22, has to make sure not in a dir that contains deepcell/applications
from deepcell.applications import NuclearSegmentation



# Load the image
im0 = io.imread('M872956_JML_Position8_CD3_img_patch256x256.png')
im=im0[:,:,2]
io.imshow(im)
plt.show()

# Expand image dimensions to rank 4
im = np.expand_dims(im, axis=-1)
im = np.expand_dims(im, axis=0)

# Create the application, CytoplasmSegmentation does not work well
app = NuclearSegmentation()

# create the lab
y = app.predict(im)

io.imshow(y)
plt.show()

np.unique(y)

mask_true=io.imread("M872956_JML_Position8_CD3_masks_patch256x256.png")

syotil.csi(mask_true, y[0,:,:,0])

#io.imsave('CD8patch1mask.png', y[0,:,:,0])
