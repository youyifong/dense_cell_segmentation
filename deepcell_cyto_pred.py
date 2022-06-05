"""
This python script, comes from (https://github.com/vanvalenlab/deepcell-tf/blob/master/deepcell/applications/cytoplasm_segmentation_test.py), is to train Mesmer on TissueNet with two-channels input. 

Note
- First run "ml Anaconda3; ml CUDA; ml cuDNN" on Linux
- "nohup" can be used, but seems to be stopped after log-off Volta (the linux command "ps xw" shows current working jobs)

To train mesmer (on Linux):
nohup python training_mesmer.py
"""



def _semantic_loss(y_pred, y_true):
    if n_classes > 1:
        return 0.01 * losses.weighted_categorical_crossentropy(
            y_pred, y_true, n_classes=n_classes)
    return MSE(y_pred, y_true)
    
from tensorflow.keras.models import load_model
model = load_model("deepcell_cyto", custom_objects={"_semantic_loss": _semantic_loss})


from deepcell.applications import CytoplasmSegmentation
app = CytoplasmSegmentation(model)

from skimage import io
import numpy as np 
img = io.imread("test/M872956_Position8_CD8_test_img.png")
x  =np.expand_dims(img, -1) 
x  =np.expand_dims(x, 0) 
x=x[:,:,0:209,:]

# make prediction
y = app.predict(x)
y.shape
np.unique(y) # all 0
io.imsave('M872956_Position8_CD8_test_image_dc_masks_cytoplasm.png', y[0,:,:,0])


x = np.random.rand(1, 500, 500, 1)
y = app.predict(x)
np.unique(y) # all 0

#io.imsave('M872956_Position8_CD8_test_image_dc_masks_cytoplasm.png', y[0,:,:,0])


