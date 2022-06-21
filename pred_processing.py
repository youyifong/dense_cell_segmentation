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

# Import file
file_name = ['M872956_Position8_CD8', 'M872956_Position8_CD3', 'M872956_Position8_CD4', 'M872956_Position9_CD3']
pred_name = []
for i in file_name: pred_name.append('test/' + i + '_test_img_cp_masks.png')

# Maskfile to Outline
for i in range(len(pred_name)):
    maskfile2outline(pred_name[i])

# Compute AP
pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
masks_name = []
for i in file_name: masks_name.append('../test/' + i + '_test_masks.png')

res_mat = []
for i in range(len(file_name)):
    labels = io.imread(masks_name[i])
    y_pred = io.imread(pred_name[i])    
    res_vec = []
    for t in thresholds:
        res_temp = csi(labels, y_pred, threshold=t) 
        res_vec.append(res_temp)
    res_mat.append(res_vec)

res_mat = pd.DataFrame(res_mat)
print(list(np.mean(res_mat, axis=0)))

#colnames = []
#for i in thresholds: colnames.append("Threshold_" + str(i))
#res_mat.columns = colnames
#res_mat.to_csv('csi.txt', header=True, index=None, sep=',')
#print(res_mat)
