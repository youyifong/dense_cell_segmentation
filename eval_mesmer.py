### 1. Evaluation images from K ###
'''
In this script, prediction using the pre-trained deepcell is done with Mesmer() and CytoplasmSegmentation().
Refer to https://github.com/vanvalenlab/deepcell-tf/blob/master/notebooks/applications/Mesmer-Application.ipynb.
'''

# Library
import os
import numpy as np
import deepcell
import cv2


# Import images
os.getcwd()
dapi_img = cv2.imread('/home/shan/kdata/M872956_Position8_DAPI_img_patch.png') # cv2.imread imports an image as BGR
cd3_img = cv2.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch.png')
X_test = np.stack((dapi_img[:,:,0], cd3_img[:,:,0]), axis=2) # image with two channels: DAPI and CD3
X_test = X_test.reshape((1,256,256,2))
#X_test = cd3_img[:,:,0] # image with one channel: CD3
#X_test = X_test.reshape((1,256,256,1))


# Prediction
# DAPI + CD3
from deepcell.applications import Mesmer
app = Mesmer()
masks = app.predict(X_test, image_mpp=2)
cv2.imwrite('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_2ch_mpp2.png', masks[0])

# CD3 
from deepcell.applications import CytoplasmSegmentation
app = CytoplasmSegmentation()
masks = app.predict(X_test, image_mpp=5)
cv2.imwrite('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_1ch_mpp5.png', masks[0])


# Plotting
import numpy as np
from cellpose import utils, io
import matplotlib.pyplot as plt

cd3_img = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch.png')
masks = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_2ch_mpp2.png')

my_dpi = 96
outlines = utils.masks_to_outlines(masks); outX, outY = np.nonzero(outlines)
imgout = cd3_img.copy(); imgout[outX, outY] = np.array([255,255,255])
fig=plt.figure(figsize=(imgout.shape[0]/my_dpi, imgout.shape[1]/my_dpi), dpi=my_dpi); plt.gca().set_axis_off(); plt.imshow(imgout)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator()); plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.savefig("/home/shan/kdata/M872956_Position8_CD3_img_patch_outlines_2ch_mpp2.png", bbox_inches = 'tight', pad_inches = 0); plt.close('all')


# Calculating CSI
import numpy as np
#from cellpose import utils, io
from utils import * # this file should be in the current working directory at this point

masks_true = io.imread('/home/shan/kdata/M872956_Position8_CD3_masks_patch.png')
#masks_pred = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_1ch_mpp05.png') # csi=0.0
masks_pred = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_1ch_mpp1.png') # csi=0.01
#masks_pred = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_1ch_mpp2.png') # csi=0.11
#masks_pred = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_2ch_mpp025.png') # csi=0.03
#masks_pred = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_2ch_mpp05.png') # csi=0.12
#masks_pred = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_2ch_mpp1.png') # csi=0.10
#masks_pred = io.imread('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_2ch_mpp2.png') # csi=0.02
csi([masks_true],[masks_pred], threshold=0.5, verbose=0)
