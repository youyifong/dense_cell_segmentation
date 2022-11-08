"""
This python script, comes from (https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/cytoplasm_segmentation_test.py), is to train Mesmer on TissueNet with two-channels input. 

Note
- First run "ml Anaconda3; ml CUDA; ml cuDNN" on Linux
- "nohup" can be used, but seems to be stopped after log-off Volta (the linux command "ps xw" shows current working jobs)

To train mesmer (on Linux):
nohup python training_mesmer.py
"""


from skimage import io
import numpy as np 

img = io.imread("test/M872956_Position8_CD8_test_img.png")

x=np.expand_dims(img[:,:,2], -1) 
x=np.expand_dims(x, 0) 


## prediction
def _semantic_loss(y_pred, y_true):
    if n_classes > 1:
        return 0.01 * losses.weighted_categorical_crossentropy(
            y_pred, y_true, n_classes=n_classes)
    return MSE(y_pred, y_true)
    
from tensorflow.keras.models import load_model

model = load_model("deepcell_cyto", custom_objects={"_semantic_loss": _semantic_loss})


from deepcell.applications import CytoplasmSegmentation
app = CytoplasmSegmentation(model)

y = app.predict(x)
y = app.predict(x, image_mpp=.5)
print(np.unique(y))
io.imsave('M872956_Position8_CD8_test_image_dc_masks_cytoplasm.png', y[0,:,:,0])

#x = np.random.rand(1, 500, 500, 1)
#y = app.predict(x)
#np.unique(y) # all 0




### another way of making prediction
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

from deepcell.model_zoo.panopticnet import PanopticNet

prediction_model = PanopticNet(
    backbone='resnet50',
    input_shape=(256, 256, 1),
    norm_method=None,
    num_semantic_heads=2,
    num_semantic_classes=[1, 3], # inner distance, pixelwise
    location=True,  # should always be true
    include_top=True)

prediction_model.load_weights("/fh/fast/fong_y/tissuenet_1.0/mesmer/yfong/tissuenet_v1.0_deep_watershed.h5", by_name=True) # load updated weights


### Prediction on test data
import skimage.io as io
from deepcell_toolbox.deep_watershed import deep_watershed

test_images = prediction_model.predict(x) # four outputs for each image (maybe two for cell and two for nuclear)

#image_mpp 


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



a=[0,0,1,1,2,2]
b=[0,0,1,1,1,2]
np.histogram2d(a,b,bins=(np.append(np.unique(a), np.inf),np.append(np.unique(a), np.inf)))
