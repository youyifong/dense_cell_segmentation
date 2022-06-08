### Evaluation ###
### Library
import os
import numpy as np
import deepcell
import cv2


### Import test data
#prediction_folder = "models"
#MODEL_DIR = os.path.join("/fh/fast/fong_y/tissuenet_1.0/mesmer", prediction_folder)
#NPZ_DIR = "/fh/fast/fong_y/tissuenet_1.0/"
# TissueNet data
#npz_name = "tissuenet_v1.0_"
#test_dict = np.load(NPZ_DIR + npz_name + "test.npz")
#X_test, y_test = test_dict['X'], test_dict['y']

### Images from K
os.getcwd()
dapi_img = cv2.imread('/home/shan/M872956_Position8_DAPI_img.tif', 0) # for grayscale image 
cd3_img = cv2.imread('/home/shan/M872956_Position8_CD3_img.png') 

X_test = np.stack((dapi_img, cd3_img[:,:,2]), axis=2) # first channel is DAPI, second channel is CD3


### Load trained model
from deepcell.model_zoo.panopticnet import PanopticNet
prediction_model = PanopticNet(
    backbone='resnet50',
    #input_shape=X_test.shape[1:],
    input_shape=X_test.shape,
    norm_method='std', # not sure norm_method=None or norm_method=''std
    num_semantic_heads=4,
    num_semantic_classes=[1, 3, 1, 3], # inner distance, pixelwise, inner distance, pixelwise
    location=True,  # should always be true
    include_top=True)

#model_name = npz_name + 'deep_watershed'
#model_path = os.path.join(MODEL_DIR, '{}.h5'.format(model_name)) # load the trained model
#prediction_model.load_weights(model_path, by_name=True) # load updated weights


### Prediction on test data
import skimage.io as io
from timeit import default_timer
from deepcell_toolbox.deep_watershed import deep_watershed

start = default_timer()
test_images = prediction_model.predict(X_test) # four outputs for each image (maybe two for cell and two for nuclear)
watershed_time = default_timer() - start
print('Watershed segmentation of shape', test_images[0].shape, 'in', watershed_time, 'seconds.')

masks = deep_watershed(
    test_images,
    min_distance=10,
    detection_threshold=0.1,
    distance_threshold=0.01,
    exclude_border=False,
    small_objects_threshold=0)

group = 'test'
for i in range(masks.shape[0]):
    pred_mask = masks[i][:,:,0]
    io.imsave(os.path.join(NPZ_DIR, 'images', group, 'res_mesmer_TN', group+str(i)+'_masks.tif'), pred_mask)


masks_true_path = os.path.join('/fh/fast/fong_y/tissuenet_1.0/images/test/res_mesmer_TN/test0_masks.tif')
masks_true = io.imread(masks_true_path)
