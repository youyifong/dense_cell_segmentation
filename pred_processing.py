# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:40:31 2022

@author: Youyi

e.g. python ../../pred_processing.py 0 csi

The first argument gives the folder hwere the predictions reside: 0, 1, 2, _saved
The second argument tells the type of analysis to run: csi, bias, tpfpfn, coloring
"""

# Library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose import utils, io
import glob
import sys

from utils import * # util.py should be in the current working directory at this point

# Import file
files = sorted(glob.glob('../testmasks/*')) # for test
#files = sorted(glob.glob('../train/*')) # for training
file_name = []
for i in range(len(files)):
    temp = files[i]
    filename = temp.split('/')[-1]
    filename = filename.split('_masks.png')[0]
    file_name.append(filename)

pred_name = []
#for i in file_name: pred_name.append('test/' + i + '_img_cp_masks.png') # for test
#for i in file_name: pred_name.append('train/' + i + '_img_cp_masks.png') # for training
pred_name = sorted(glob.glob('testimages'+str(sys.argv[1])+'/*_test_img_cp_masks.png')) # for test

#if sys.argv[1]=='0':
#    print (', '.join(pred_name))

# Maskfile to Outline
#for i in range(len(pred_name)):
#    maskfile2outline(pred_name[i])

# Compute AP
masks_name = []
for i in file_name: masks_name.append('../testmasks/' + i + '_masks.png') # for test
#for i in file_name: masks_name.append('../train/' + i + '_masks.png') # for training

thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
res_mat = []
for i in range(len(file_name)):
    labels = io.imread(masks_name[i])
    y_pred = io.imread(pred_name[i])
    
    if sys.argv[2]=='bias':
        res_temp = bias(labels, y_pred)
        res_mat.append(round(res_temp,5))
    elif sys.argv[2]=='csi': 
        res_vec = []
        for t in thresholds:
            res_temp = csi(labels, y_pred, threshold=t) 
            res_vec.append(round(res_temp,6))
        res_mat.append(res_vec)
    elif sys.argv[2]=='tpfpfn': 
        res_vec = tpfpfn(labels, y_pred, threshold=0.5) 
        res_mat.append(res_vec)
    elif sys.argv[2]=='coloring':
        color_fp_fn(masks_name[i], pred_name[i])
        
        

# Print results
# 1) AP over test images at given thresholds
#res_mat = pd.DataFrame(res_mat)
#print(list(np.mean(res_mat, axis=0))) # AP over four test images at given thresholds

if sys.argv[2]=='bias':
    res_temp = np.array([res_mat])
    print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
elif sys.argv[2]=='csi':
    #APs at threshold of 0.5
    res_temp = list(list(zip(*res_mat))[0]) # AP at threshold of 0.5
    res_temp = np.array([res_temp]) 
    #print(" \\\\\n".join([" & ".join(map(str,line)) for line in res_temp])) # latex table format
    print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format
elif sys.argv[2]=='tpfpfn':
    res_temp = np.array([res_mat])
    print (', '.join(pred_name))
    print(" \\\\\n".join([",".join(map(str,line)) for line in res_temp])) # csv format

