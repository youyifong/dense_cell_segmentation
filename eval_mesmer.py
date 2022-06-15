### 1. Evaluation images from K ###
'''
In this script, prediction using the pre-trained deepcell is done with Mesmer() and CytoplasmSegmentation().
Refer to https://github.com/vanvalenlab/deepcell-tf/blob/master/notebooks/applications/Mesmer-Application.ipynb.
'''

# Library
import os
import numpy as np
import deepcell
import skimage.io as io


# Import images
os.getcwd()
dapi_img = io.imread('/home/shan/kdata/256x256/M872956_Position8_DAPI_img_patch256x256.png')
cd3_img = io.imread('/home/shan/kdata/256x256/M872956_Position8_CD3_img_patch256x256.png')
X_test = np.stack((dapi_img[:,:,2], cd3_img[:,:,2]), axis=2) # image with two channels: DAPI and CD3
#X_test = cd3_img[:,:,0] # image with one channel: CD3

height = X_test.shape[0]
width = X_test.shape[1]
channels = X_test.shape[2]
X_test = X_test.reshape((1,height,width,channels))


# Prediction
# DAPI + CD3
from deepcell.applications import Mesmer
app = Mesmer()
masks = app.predict(X_test, image_mpp=0.5)
np.save('/home/shan/kdata/256x256/M872956_Position8_CD3_img_patch256x256_masks_2ch_mpp05', masks[0]) # save masks as .npy file

# CD3 
from deepcell.applications import CytoplasmSegmentation
app = CytoplasmSegmentation()
masks = app.predict(X_test, image_mpp=5)
io.imwrite('/home/shan/kdata/M872956_Position8_CD3_img_patch_masks_1ch_mpp5.png', masks[0,])


# Plotting
import numpy as np
from cellpose import utils, io
import matplotlib.pyplot as plt

cd3_img = io.imread('/home/shan/kdata/512x512/M872956_Position8_CD3_img_patch512x512.png')
masks = np.load('/home/shan/kdata/512x512/M872956_Position8_CD3_img_patch512x512_masks_2ch_mpp05.npy')
masks = masks[:,:,0]

my_dpi = 96
outlines = utils.masks_to_outlines(masks); outX, outY = np.nonzero(outlines)
imgout = cd3_img.copy(); imgout[outX, outY] = np.array([255,255,255])
fig=plt.figure(figsize=(imgout.shape[0]/my_dpi, imgout.shape[1]/my_dpi), dpi=my_dpi); plt.gca().set_axis_off(); plt.imshow(imgout)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator()); plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.savefig("/home/shan/kdata/512x512/M872956_Position8_CD3_img_patch512x512_outlines_2ch_mpp05.png", bbox_inches = 'tight', pad_inches = 0)
plt.close('all')


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
csi(masks_true, masks_pred, threshold=0.5)




### Create a patch of CD3, DAPI, and gt masks ###
import numpy as np
import skimage.io as io

# CD3 image
cd3_img = io.imread('/Users/shan/Desktop/Paper/YFong/8.New/Result/ground_truth/mesmer/M872956_Position8_CD3_img.png')
#cd3_img = cd3_img[450:482, 450:482, :] # 32x32
#cd3_img = cd3_img[400:656, 400:650, :] # 256x250
#cd3_img = cd3_img[400:656, 400:656, :] # 256x256
#cd3_img = cd3_img[250:750, 250:750, :] # 500x500
cd3_img = cd3_img[250:762, 250:762, :] # 512x512
io.imsave('/Users/shan/Desktop/M872956_Position8_CD3_img_patch512x512.png', cd3_img)

# CD3 gt masks
cd3_masks = io.imread('/Users/shan/Desktop/Paper/YFong/8.New/Result/ground_truth/single/train_test/CD3/M872956_Position8_CD3-BUV395_no_inputs_GTmasks_1908_masks.png')
#cd3_masks = cd3_masks[450:482, 450:482] # 32x32
#cd3_masks = cd3_masks[400:656, 400:650] # 256x250
#cd3_masks = cd3_masks[400:656, 400:656] # 256x256
#cd3_masks = cd3_masks[250:750, 250:750] # 500x500
cd3_masks = cd3_masks[250:762, 250:762] # 512x512
io.imsave('/Users/shan/Desktop/M872956_Position8_CD3_masks_patch512x512.png', cd3_masks)

# DAPI image
dapi_img = io.imread('/Users/shan/Desktop/Paper/YFong/8.New/Result/ground_truth/mesmer/M872956_Position8_DAPI_img.png')
dapi_img = dapi_img[:,:,0:3] # dapi_img has four-channel; not sure why but the values in the last channel are all 255
#dapi_img = dapi_img[450:482, 450:482, :] # 32x32
#dapi_img = dapi_img[400:656, 400:650, :] # 256x250
#dapi_img = dapi_img[400:656, 400:656, :] # 256x256
#dapi_img = dapi_img[250:750, 250:750, :] # 500x500
dapi_img = dapi_img[250:762, 250:762, :] # 512x512
io.imsave('/Users/shan/Desktop/M872956_Position8_DAPI_img_patch512x512.png', dapi_img)





#####




### 2. Evaluation TissueNet ###
# Library
import os
import numpy as np
import deepcell
import skimage.io as io


# Prediction
# nuclear + cytoplasm (images with two channels)
from deepcell.applications import Mesmer
app = Mesmer()

os.getcwd()
img_path = "/fh/fast/fong_y/tissuenet_1.0/images/test"

for i in range(1249): # total number of test images is 1249 (index: 0-1248)
    print(i)
    img = io.imread(os.path.join(img_path, "test" + str(i) +"_img.tif"))
    X_test = np.stack((img[:,:,1], img[:,:,2]), axis=2) # two channels: nuclear (green) and cytoplasm (blue)
    X_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    masks = app.predict(X_test, image_mpp=0.5) # image_mpp=0.5 like as the reference notebook, but default is None
    io.imsave(os.path.join(img_path, "res_mesmer", "test" + str(i) + "_img_ms_masks.tif"), masks[0,:,:,0])


# Calculating CSI
import os
import numpy as np
from cellpose import utils, io
from utils import * # this file should be in the current working directory at this point

os.getcwd()
img_path = "/fh/fast/fong_y/tissuenet_1.0/images/test"

pred = []
for i in range(1249): # total number of test images is 1249 (index: 0-1248)
    print(i)
    masks_true = io.imread(os.path.join(img_path, "test" + str(i) +"_masks.tif"))
    masks_pred = io.imread(os.path.join(img_path, "res_mesmer", "test" + str(i) + "_img_ms_masks.tif"))
    pred.append(csi(masks_true, masks_pred, threshold=0.5))
    #pred.append(precision(masks_true, masks_pred, threshold=0.5))
    #pred.append(recall(masks_true, masks_pred, threshold=0.5))
pred
np.mean(pred) # csi=0.71; precision=0.85; recall=0.80





#####




### Appendix. Prediction with PanopticNet() ###
'''
Reference notebook: https://github.com/vanvalenlab/deepcell-tf/blob/master/notebooks/training/panopticnets/Nuclear%20Segmentation%20-%20DeepWatershed.ipynb
'''

# Library
import os
import numpy as np
import deepcell
import skimage.io as io


# Import images
# CD3
os.getcwd()
cd3_img = io.imread('/home/shan/kdata/256x256/M872956_Position8_CD3_img_patch256x256.png')
X_test = cd3_img[:,:,2] # image with one channel: CD3
height = X_test.shape[0]
width = X_test.shape[1]
channels = 1
X_test = X_test.reshape((1,height,width,channels))


# Prediction
from deepcell.model_zoo.panopticnet import PanopticNet

# for one-channel input
prediction_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 1),
    norm_method=None,
    num_semantic_heads=2,
    #num_semantic_classes=[1, 3], # inner distance, pixelwise; makes error
    num_semantic_classes=[1, 1], # seems to inner distance and outdistance (not pixelwise)
    location=True,  # should always be true
    include_top=True)
prediction_model.load_weights("/fh/fast/fong_y/tissuenet_1.0/mesmer/shan/cd3_June152022.h5", by_name=True) # load updated weights


import skimage.io as io
from timeit import default_timer
from deepcell_toolbox.deep_watershed import deep_watershed

start = default_timer()
test_images = prediction_model.predict(X_test)
watershed_time = default_timer() - start
print('Watershed segmentation of shape', test_images[0].shape, 'in', watershed_time, 'seconds.')

masks = deep_watershed( # follow taken from cytoplasm_segmentation.py 
    test_images,
    min_distance=10,
    detection_threshold=0.1,
    distance_threshold=0.01,
    exclude_border=False,
    small_objects_threshold=0)

np.unique(masks[0])
np.save('/home/shan/kdata/256x256/M872956_Position8_CD3_img_patch256x256_masks_cd3_trained', masks[0])
