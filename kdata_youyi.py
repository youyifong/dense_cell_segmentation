### Training ###
"""
Note
- To train Cellpose model, TissueNet images and ground-truth masks should be saved as .tiff files.
- "nohup" works on the server, and the linux command "ps xw" shows current working jobs.

working dir: ~/deeplearning/kdata/



##Training:

# no validation data    
python -m cellpose --train --use_gpu --dir "train3" --n_epochs 500 --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --verbose 

# with validation data
python -m cellpose --train --use_gpu --dir "train5" --n_epochs 2500 --test_dir "train5/val" --save_each --save_every 1 --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --verbose 

python -m cellpose --train --use_gpu --dir "train6" --n_epochs 500  --test_dir "train6/val" --save_each --save_every 1 --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --verbose 

# add pos 9
python -m cellpose --train --use_gpu --dir "train20220616" --n_epochs 500 --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --verbose 

##Prediction:

# pretrained cyto2 model
    python -m cellpose --use_gpu --dir "test" --pretrained_model cyto2  --save_png  --verbose 
    
# model trained with cd8 part, 500 epochs
    python -m cellpose --verbose --use_gpu --dir "test" --pretrained_model "train1/models/cellpose_residual_on_style_on_concatenation_off_train1_2022_05_31_20_10_07.089239"  --save_png

# train2: train with cd3 + cd8 part, 500 epochs (2500 epochs performance is similar)

# cellpose 0.7
python -m cellpose --train --use_gpu --dir "kdata/train2" --n_epochs 100 --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 
python -m cellpose --dir "kdata/test" --pretrained_model "kdata/train2/models/cellpose_residual_on_style_on_concatenation_off_train2_2022_06_08_20_57_22.012551"  --save_png --use_gpu

# cellpose 2.0
python -m cellpose --train --dir "train2" --no_resample --n_epochs 500 --pretrained_model cyto2 --diam_mean=17 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --verbose --use_gpu
python -m cellpose --dir "test" --diameter 17 --pretrained_model "train2/models/cellpose_residual_on_style_on_concatenation_off_train2_2022_06_09_09_38_55.039903"   --save_png --verbose --use_gpu 





python -m cellpose --use_gpu --dir "test" --diam--pretrained_model "train2/models/cellpose_residual_on_style_on_concatenation_off_train2_2022_06_07_15_57_40.287329"  --save_png


# model trained with cd3, 500 epochs
    python -m cellpose --verbose --use_gpu --dir "test" --pretrained_model "train3/models/cellpose_residual_on_style_on_concatenation_off_train3_2022_05_31_20_12_03.723997"  --save_png
    python -m cellpose --verbose --use_gpu --dir "test" --pretrained_model "train3/models/cellpose_residual_on_style_on_concatenation_off_train3_2022_06_07_12_31_40.639360"  --save_png

# model trained with cd8 part + cd3 + cd4, 2500 epochs 
    python -m cellpose --verbose --use_gpu --dir "test" --pretrained_model "train4/models/cellpose_residual_on_style_on_concatenation_off_train4_2022_06_07_09_41_46.839430"  --save_png

# model trained with cd3 + cd4, with val cd8 part, best epoch by val loss and train loss 
    python -m cellpose --verbose --use_gpu --dir "test" --pretrained_model "train5/models/cellpose_residual_on_style_on_concatenation_off_train5_2022_06_07_11_52_54.052792_epoch_202"  --save_png

    python -m cellpose --verbose --use_gpu --dir "test" --pretrained_model "train5/models/cellpose_residual_on_style_on_concatenation_off_train5_2022_06_07_11_52_54.052792_epoch_499"  --save_png

# model trained with cd3, with val cd8 part, best epoch by val loss and train loss 
    python -m cellpose --verbose --use_gpu --dir "test" --pretrained_model "train6/models/cellpose_residual_on_style_on_concatenation_off_train6_2022_06_07_12_25_42.902173_epoch_466"  --save_png




"""

import os
from utils import * # this file should be in the current working directory at this point
os.chdir("../K's training data")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose import utils, io
import glob
from read_roi import read_roi_file # pip install read-roi
from PIL import Image, ImageDraw

from importlib_metadata import version

# get image width and height
img = io.imread('JM_Les_Pos9_CD3-gray_CD4-green_CD8-red_CD4CD8-aligned.tif') # image
height = img.shape[1]
width = img.shape[2]
roifiles2mask("JM_Les_Pos9_CD3_RoiSet_1986/*", width, height)


maskfile2outline('CD8patch1_mask.png')


    
# cellpose==0.6.5.dev4+g4b1eaa3

#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_pretrained.png') #0.45
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train1.png') #0.71
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train3.png') #0.73
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train4.png') #0.75
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train5.png') #0.46
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train6.png') #0.53
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_pretrainednew.png') # 0.51
pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
labels = io.imread('M872956_Position8_CD8_test_masks.png')
y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks.png') #0.77
for t in thresholds:
    pred_vec = csi(labels, y_pred, threshold=t, verbose=0) 
    pred_mat.append(pred_vec)
pred_mat


# cellpose==2.0, diam 17
pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
labels = io.imread('M872956_Position8_CD8_test_masks.png')
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_pretrained.png') #0.45
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train1.png') #0.71
y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train2new.png') #0.56
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train3.png') #0.73
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train4.png') #0.75
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train5.png') #0.46
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train6.png') #0.53
#y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_pretrainednew.png') # 0.51
for t in thresholds:
    #pred_vec = csi([labels], [y_pred], threshold=t, verbose=0) 
    pred_vec = csi(labels, y_pred, threshold=t) 
    pred_mat.append(pred_vec)
pred_mat




# Split CD3 and CD4 image/masks files into training (4/5) and test (1/5)
# For image
img = io.imread('JM_Les_Pos8_img_CD3-gray_CD4-green_CD8-red_aligned.tif') # for image
height = img.shape[1]
width =  img.shape[2]
training = img[:, (int(height/5)+1):, :] # training
test =     img[:, :(int(height/5)+1), :] # test
plt.imshow(training[0,:,:]) # display training; it can be changed to test
plt.show()

# train files need to be 3-D
data=np.zeros((training.shape[1],width,3)); data[:,:,0]=training[0,:,:]
io.imsave('train/M872956_Position8_CD8_train_img.png', data) 
data=np.zeros((training.shape[1],width,3)); data[:,:,0]=training[1,:,:]
io.imsave('train/M872956_Position8_CD4_train_img.png', data) 
data=np.zeros((training.shape[1],width,3)); data[:,:,0]=training[2,:,:]
io.imsave('train/M872956_Position8_CD3_train_img.png', data) 
# test files do not need to be 3-D
io.imsave('test/M872956_Position8_CD8_test_img.png', test[0,:,:])
io.imsave('test/M872956_Position8_CD4_test_img.png', test[1,:,:])
io.imsave('test/M872956_Position8_CD3_test_img.png', test[2,:,:])


# For splitting CD3 ground-truth masks
masks = io.imread('JM_Les_Pos8_CD3_RoiSet_1908_masks.tif') # for masks
height = masks.shape[0]
width = masks.shape[1]
training = masks[(int(height/5)+1):, :] # training
test = masks[:(int(height/5)+1), :] # test
plt.imshow(training, cmap='gray') # display training; it can be changed to test
plt.show()
io.imsave('train/M872956_Position8_CD3_train_masks.png', training) 
io.imsave('test/M872956_Position8_CD3_test_masks.png', test) 

# For splitting CD4 ground-truth masks
masks = io.imread('JM_Les_Pos8_CD4_with_CD3_input_RoiSet_1354_masks.png') # for masks
height = masks.shape[0]
width = masks.shape[1]
training = masks[(int(height/5)+1):, :] # training
test = masks[:(int(height/5)+1), :] # test
plt.imshow(training, cmap='gray') # display training; it can be changed to test
plt.show()
io.imsave('train/M872956_Position8_CD4_train_masks.png', training) # it can be changed to test
io.imsave('test/M872956_Position8_CD4_test_masks.png', test) # it can be changed to test


# add empty channels or permutation channel position
img = io.imread('JM_Les_Pos9_CD3-gray_CD4-green_CD8-red_CD4CD8-aligned.tif') # for image
height = img.shape[1]
width =  img.shape[2]
data=np.zeros((height,width,3)); data[:,:,0]=img[2,:,:] # 0 seems to correspond to blue channel or channel 3
io.imsave('train/M872956_Position9_CD4_img.png', data) 

img = io.imread('train/M872956_Position8_CD8_train_img.png'); print(sum(sum(img[:,:,0]))); print(sum(sum(img[:,:,1]))); print(sum(sum(img[:,:,2])))
img = io.imread('train/M872956_Position8_CD4_img.png'); print(sum(sum(img[:,:,0]))); print(sum(sum(img[:,:,1]))); print(sum(sum(img[:,:,2])))
img = io.imread('train/M872956_Position8_CD3_img.png'); print(sum(sum(img[:,:,0]))); print(sum(sum(img[:,:,1]))); print(sum(sum(img[:,:,2])))







import os
import numpy as np
import deepcell
import skimage.io as io

X = io.imread('test/CD8patch1.jpg')
X_test=np.expand_dims(X[:,:,0], -1) 
X_test=np.expand_dims(X_test, 0)

from deepcell.applications import CytoplasmSegmentation
app = CytoplasmSegmentation()

masks = app.predict(X_test, image_mpp=1)
io.imsave('test/CD8patch1_mask.png', masks[0,:,:,0])

maskfile2outline('CD8patch1_mask.png')
