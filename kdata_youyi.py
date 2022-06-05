### Training ###
"""
Note
- To train Cellpose model, TissueNet images and ground-truth masks should be saved as .tiff files.
- "nohup" works on the server, and the linux command "ps xw" shows current working jobs.

working dir: ~/deeplearning/kdata/



To train the cellpose model (on Linux):
python -m cellpose --train --use_gpu --dir "train4"  --pretrained_model cyto2 --img_filter _img --mask_filter _masks --chan 3 --chan2 0 --n_epochs 2500

# use val
python -m cellpose --train --use_gpu --dir "train5" --test_dir "train5/val" --pretrained_model cyto2 --img_filter _img --mask_filter _masks --n_epochs 500



To run cellpose models (on Linux):

# pretrained
    python -m cellpose --use_gpu --dir "pretrained" --pretrained_model cyto2  --save_png
    
# trained with cd8 part, 500 epochs
    python -m cellpose --use_gpu --dir "test" --pretrained_model "train1/models/cellpose_residual_on_style_on_concatenation_off_train1_2022_05_31_20_10_07.089239"  --save_png

# trained with cd8 part + cd3, 500 epochs (2500 epochs performance is similar)
    python -m cellpose --use_gpu --dir "test" --pretrained_model "train2/models/cellpose_residual_on_style_on_concatenation_off_train2_2022_05_31_20_10_59.860657"  --save_png

# trained with cd3, 500 epochs
    python -m cellpose --use_gpu --dir "test" --pretrained_model "train3/models/cellpose_residual_on_style_on_concatenation_off_train3_2022_05_31_20_12_03.723997"  --save_png

# trained with cd8 part + cd3 + cd4, 2500 epochs 
    python -m cellpose --use_gpu --dir "test" --pretrained_model "train4/models/cellpose_residual_on_style_on_concatenation_off_train4_2022_06_01_11_02_56.663779"  --save_png



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



# get image width and height
img = io.imread('JM_Les_Pos8_img_CD3-gray_CD4-green_CD8-red_aligned.tif') # image
height = img.shape[1]
width = img.shape[2]
roifiles2mask("JM_Les_Pos8_CD4_with_CD3_input_RoiSet_1350/*", width, height)


maskfile2outline('M872956_Position8_CD8_test_image_dc_masks_cytoplasm.png')
    


pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
for t in thresholds:
    labels = io.imread('M872956_Position8_CD8_test_masks.png')
    #y_pred = io.imread('M872956_Position8_CD8_test_img_cp_pretrained_masks.png') #0.45
    #y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train1.png') #0.71
    #y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train2.png') #0.78
    #y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train3.png') #0.73
    y_pred = io.imread('M872956_Position8_CD8_test_img_cp_masks_train4.png') #0.74
    pred_vec = csi_old([labels], [y_pred], threshold=t, verbose=0) 
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
img = io.imread('train/M872956_Position8_CD4_img.png') # for image
height = img.shape[0]
width =  img.shape[1]
data=np.zeros((height,width,3)); data[:,:,0]=img[:,:,1]
io.imsave('train/M872956_Position8_CD4_img.png', data) 

img = io.imread('train/M872956_Position8_CD8_train_img.png'); print(sum(sum(img[:,:,0]))); print(sum(sum(img[:,:,1]))); print(sum(sum(img[:,:,2])))
img = io.imread('train/M872956_Position8_CD4_img.png'); print(sum(sum(img[:,:,0]))); print(sum(sum(img[:,:,1]))); print(sum(sum(img[:,:,2])))
img = io.imread('train/M872956_Position8_CD3_img.png'); print(sum(sum(img[:,:,0]))); print(sum(sum(img[:,:,1]))); print(sum(sum(img[:,:,2])))
