### Training ###
"""
Note
- To train Cellpose model, TissueNet images and ground-truth masks should be saved as .tiff files.
- "nohup" works on the server, and the linux command "ps xw" shows current working jobs.

working dir: ~/deeplearning/kdata/

To train the cellpose model (on Linux):
python -m cellpose --train --use_gpu --dir "train2"  --pretrained_model cyto2 --img_filter _img --mask_filter _masks --n_epochs 500


To run cellpose models (on Linux):
# trained with cd8 part
    python -m cellpose --use_gpu --dir "train1/test" --pretrained_model "train1/models/cellpose_residual_on_style_on_concatenation_off_train1_2022_05_29_15_05_38.088182"  --save_png
# trained with cd8 part and cd3   
    python -m cellpose --use_gpu --dir "train2/test" --pretrained_model "train2/models/cellpose_residual_on_style_on_concatenation_off_train2_2022_05_29_15_12_30.674416"  --save_png
# pretrained
    python -m cellpose --use_gpu --dir "pretrained" --pretrained_model cyto2  --save_png


"""



import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import utils, io
import glob
from read_roi import read_roi_file # pip install read-roi
from PIL import Image, ImageDraw

from utils import * # this file should be in the current working directory at this point


os.chdir("../K's training data")

# get image width and height
img = io.imread('JM_Les_Pos8_CD3-gray_CD4-green_CD8-red_aligned-CD4_CD8_GTmasks-blue.tif') # image
width = img.shape[2]
height = img.shape[1]

roifiles2mask("JM_Les_Pos8_CD3_RoiSet_1908/*", width, height)


maskfile2outline('M872956_Position8_CD8_test_img_cp_pretrained_masks.png')
    


pred_mat = []
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for t in thresholds:
    labels = io.imread('M872956_Position8_CD8_test_masks.png')
    #y_pred = io.imread('M872956_Position8_CD8_test_img_cp_pretrained_masks.png') #0.73, 0.32
    #y_pred = io.imread('M872956_Position8_CD8_test_img_cp_cd8part_masks.png') #0.89 0.58
    y_pred = io.imread('M872956_Position8_CD8_test_img_cp_cd3cd8part_masks.png') #0.92 0.63
    pred_vec = csi_old([labels], [y_pred], threshold=t, verbose=0) 
    pred_mat.append(pred_vec)
pred_mat