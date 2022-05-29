### Training ###
"""
Note
- To train Cellpose model, TissueNet images and ground-truth masks should be saved as .tiff files.
- "nohup" works on the server, and the linux command "ps xw" shows current working jobs.

working dir: ~/deeplearning/kdata/

To train the cellpose model (on Linux):
python -m cellpose --train --use_gpu --dir "train"  --pretrained_model cyto2 --img_filter _img --mask_filter _masks --n_epochs 500


To run the newly trained cellpose model (on Linux):
python -m cellpose --use_gpu --dir "tmp" --pretrained_model "train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_05_27_17_06_52.391249"  --save_tif


To run the pretrained version of the CellPose model:
python -m cellpose --use_gpu --dir "test" --pretrained_model cyto2  --save_tif


"""



import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import utils, io
import glob
from read_roi import read_roi_file # pip install read-roi
from PIL import Image, ImageDraw



# From .roi files to masks file
def roifiles2mask(roi_files, width, height):
    files = glob.glob(roi_files) 
    masks = Image.new('I', (width, height), 0)
    for idx in range(len(files)):
        print(idx)
        mask_temp = read_roi_file(files[idx])
        filename = files[idx].split('\\')[-1][:-4]
        x = mask_temp[filename]['x']
        y = mask_temp[filename]['y']
            
        polygon = []
        for i in range(len(x)):
            polygon.append((x[i], y[i]))
        
        ImageDraw.Draw(masks).polygon(polygon, outline=idx+1, fill=idx+1)
        
    masks = np.array(masks, dtype=np.uint16) # resulting masks
    plt.imshow(masks, cmap='gray') # display ground-truth masks
    plt.show()
    io.imsave(os.path.split(roi_files)[0]+'_masks.tif', masks)
    
    outlines = utils.masks_to_outlines(masks)
    plt.imsave(os.path.split(roi_files)[0] + "_masks_outline.tif", outlines, cmap='gray')


# get image width and height
img = io.imread('JM_Les_Pos8_CD3-gray_CD4-green_CD8-red_aligned-CD4_CD8_GTmasks-blue.tif') # image
width = img.shape[2]
height = img.shape[1]

roifiles2mask("JM_Les_Pos8_CD3_RoiSet_1908/*", width, height)


def maskfile2outline(mask_file):
    masks = io.imread(mask_file)
    outlines = utils.masks_to_outlines(masks)
    plt.imsave(os.path.splitext(mask_file)[0] + "_outline.tif", outlines, cmap='gray')


maskfile2outline('test/M872956_Position8_CD8_test_img_cp_masks_1.tif')
maskfile2outline('test/M872956_Position8_CD8_train_img_cp_masks_1.tif')
maskfile2outline('M872956_Position8_CD8_test_img_cp_masks_pretrained.tif')
maskfile2outline('M872956_Position8_CD8_test_masks.png')
maskfile2outline('../train/M872956_Position8_CD8_train_masks.png')
maskfile2outline('M872956_Position8_CD8_train_img_cp_masks.tif')
    


pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
for t in thresholds:
    labels = io.imread('test/M872956_Position8_CD8_test_masks.png')
    y_pred = io.imread('test/M872956_Position8_CD8_test_img_cp_masks_pretrained.tif') #0.45
    #y_pred = io.imread('test/M872956_Position8_CD8_test_img_cp_masks.tif')#0.58
    #y_pred = io.imread('test/M872956_Position8_CD8_test_img_cp_masks_1.tif')#0.70
    pred_vec = csi([labels], [y_pred], threshold=t, verbose=0)
    pred_mat.append(pred_vec)
pred_mat