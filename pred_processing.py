# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:40:31 2022

@author: Youyi
"""

# Library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose import utils, io
from utils import * # util.py should be in the current working directory at this point
import glob

# Import file
files = sorted(glob.glob('../test/*'))
file_name = []
for i in range(len(files)):
    temp = files[i]
    filename = temp.split('/')[-1]
    filename = filename.split('_masks.png')[0]
    file_name.append(filename)

pred_name = []
for i in file_name: pred_name.append('test/' + i + '_img_cp_masks.png')

# Maskfile to Outline
for i in range(len(pred_name)):
    maskfile2outline(pred_name[i])

# Compute AP
masks_name = []
for i in file_name: masks_name.append('../test/' + i + '_masks.png')

thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
res_mat = []
for i in range(len(file_name)):
    labels = io.imread(masks_name[i])
    y_pred = io.imread(pred_name[i])    
    res_vec = []
    for t in thresholds:
        res_temp = csi(labels, y_pred, threshold=t) 
        res_vec.append(res_temp)
    res_mat.append(res_vec)

#res_mat = pd.DataFrame(res_mat)
#print(list(np.mean(res_mat, axis=0))) # Average precision over four test images at given thresholds
print(list(file_name))
print(list(list(zip(*res_mat))[0])) # precisions for four test images at threshold of 0.5

#colnames = []
#for i in thresholds: colnames.append("Threshold_" + str(i))
#res_mat.columns = colnames
#res_mat.to_csv('csi.txt', header=True, index=None, sep=',')
#print(res_mat)
