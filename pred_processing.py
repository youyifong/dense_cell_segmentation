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
    
    # Bias
    #res_temp = bias(labels, y_pred)
    #res_mat.append(round(res_temp,5))
    
    # AP
    res_vec = []
    for t in thresholds:
        res_temp = csi(labels, y_pred, threshold=t) 
        res_vec.append(round(res_temp,2))
    res_mat.append(res_vec)

# Print results
# 1) AP over test images at given thresholds
#res_mat = pd.DataFrame(res_mat)
#print(list(np.mean(res_mat, axis=0))) # AP over four test images at given thresholds

# 2) AP over test images at threshold of 0.5
#file_names = np.array([file_name])
#print(" \\\\\n".join([" & ".join(map(str,line)) for line in file_names])) # latex table format
#print(list(file_name)) # csv format
res_temp = list(list(zip(*res_mat))[0]) # AP at threshold of 0.5
res_temp = np.array([res_temp]) 
#print(" \\\\\n".join([" & ".join(map(str,line)) for line in res_temp])) # latex table format
print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format

# 3) Bias over test images
#file_names = np.array([file_name])
#print(" \\\\\n".join([" & ".join(map(str,line)) for line in file_names])) # latex table format
#res_temp = np.array([res_mat])
#print(" \\\\\n".join([" & ".join(map(str,line)) for line in res_temp])) # latex table format
