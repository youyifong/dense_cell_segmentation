# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:40:31 2022

@author: Youyi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import utils, io

import os
from utils import * # util.py should be in the current working directory at this point

file_name='test/M872956_Position8_CD8_test_img_cp_masks.png'


maskfile2outline(file_name) # no need to compute csi


pred_mat = []
thresholds = [0.5,0.6,0.7,0.8,0.9,1.0]
labels = io.imread('../M872956_Position8_CD8_test_masks.png')
y_pred = io.imread(file_name) 
for t in thresholds:
    pred_vec = csi(labels, y_pred, threshold=t) 
    pred_mat.append(pred_vec)
print(pred_mat)
