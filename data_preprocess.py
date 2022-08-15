### 1. For TissueNet ###
# Library
import os
import numpy as np
import skimage.io as io
from skimage.exposure import rescale_intensity


# Import TissueNet dataset
start = os.getcwd()
#npz_dir = '/Users/shan/Desktop/Paper/YFong/6.DL/Images/tissuenet_1.0' # on local
npz_dir = '/fh/fast/fong_y/tissuenet_1.0' # on server
train_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.0_train.npz'))
val_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.0_val.npz'))
test_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.0_test.npz'))


# Utils
def create_rgb_image(input_data, channel_colors):
    """
    Note
    - This function comes from deepcell.utils.plot_utils.
    - Original TissueNet image consists of 2 channels, but each image is converted to have RGB channels.
    - One thing is that rescaling pixels is included in this function, but not sure if it is necessary.
    The rescaled pixel intensity is from 0 (if original intensity is less than 5% percentile) to 1 (if original intensity is greater than 95% percentile).
    The rescaling affects to remove noise pixels and highlight true signals.
    """
    
    """Takes a stack of 1- or 2-channel data and converts it to an RGB image
    Args:
        input_data: 4D stack of images to be converted to RGB
        channel_colors: list specifying the color for each channel
    Returns:
        numpy.array: transformed version of input data into RGB version
    Raises:
        ValueError: if ``len(channel_colors)`` is not equal
            to number of channels
        ValueError: if invalid ``channel_colors`` provided
        ValueError: if input_data is not 4D, with 1 or 2 channels
    """
    
    if len(input_data.shape) != 4:
        raise ValueError('Input data must be 4D, '
                         'but provided data has shape {}'.format(input_data.shape))
    
    if input_data.shape[3] > 2:
        raise ValueError('Input data must have 1 or 2 channels, '
                         'but {} channels were provided'.format(input_data.shape[-1]))
    
    valid_channels = ['red', 'green', 'blue']
    channel_colors = [x.lower() for x in channel_colors]
    
    if not np.all(np.isin(channel_colors, valid_channels)):
        raise ValueError('Only red, green, or blue are valid channel colors')
    
    if len(channel_colors) != input_data.shape[-1]:
        raise ValueError('Must provide same number of channel_colors as channels in input_data')
    
    rgb_data = np.zeros(input_data.shape[:3] + (3,), dtype='float32') # contrainer for RGB data
    
    # rescale channels to aid plotting
    for img in range(input_data.shape[0]):
        for channel in range(input_data.shape[-1]):
            current_img = input_data[img, :, :, channel]
            non_zero_vals = current_img[np.nonzero(current_img)]
            
            # if there are non-zero pixels in current channel, we rescale
            if len(non_zero_vals) > 0:
                percentiles = np.percentile(non_zero_vals, [5, 95])
                rescaled_intensity = rescale_intensity(current_img, in_range=(percentiles[0], percentiles[1]), out_range='float32')
                
                # get rgb index of current channel
                color_idx = np.where(np.isin(valid_channels, channel_colors[channel]))
                rgb_data[img, :, :, color_idx] = rescaled_intensity
    
    # create a blank array for red channel
    return rgb_data


# Convert image with two channels to RGB channels
group = ['train', 'val', 'test'][0]
if group == 'train':
    image_X, image_y = train_dict['X'], train_dict['y']
elif group == 'val':
    image_X, image_y = val_dict['X'], val_dict['y']
elif group == 'test':
    image_X, image_y = test_dict['X'], test_dict['y']

rgb_images = create_rgb_image(image_X, channel_colors=['green', 'blue']) # green for nuclear, blue for cytoplasm
for img in range(len(rgb_images)):
    print(img)
    io.imsave(os.path.join(npz_dir, 'images', group, group+str(img)+'_img.tif'), rgb_images[img]) # image
    temp = image_y[img,:,:,0].astype(np.uint8) # whole cell (first channel), not nuclear (second channel)
    io.imsave(os.path.join(npz_dir, 'images', group, group+str(img)+'_masks.tif'), temp)


### Appendix ###
import matplotlib.pyplot as plt
plt.imshow(image_X[0,:,:,0]); plt.axis('off'); plt.show() # the first channel in image is nuclear
plt.imshow(image_X[0,:,:,1]); plt.axis('off'); plt.show() # the second channel in image is cytoplasm
plt.imshow(image_y[0,:,:,0]); plt.axis('off'); plt.show() # the first channel in ground-truth mask is cell
plt.imshow(image_y[0,:,:,1]); plt.axis('off'); plt.show() # the second channel in ground-truth mask is nuclear





#######################################################################################################





### 2. Splitting train and test using K's ground-truth masks ###
# Library
import os
import numpy as np
import skimage.io as io
import glob
from read_roi import read_roi_file # pip install read-roi
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# Import image and RoI files
root_path = '/Users/shan/Desktop/Paper/YFong/8.New/Result/kdata/images/single/CD3_pos-9'
img = io.imread(os.path.join(root_path, 'M872956_Position9_CD3_img.png')) # image
files = glob.glob(os.path.join(root_path, 'JM_Les_Pos9_CD3_RoiSet_1986/*')) # RoI files


# From .roi files to masks file
height = img.shape[0]
width = img.shape[1]
masks = Image.new('I', (width, height), 0)

for idx in range(len(files)):
    print(idx)
    filename = files[idx].split('/')[-1][:-4]
    mask_temp = read_roi_file(files[idx])
    x = mask_temp[filename]['x']
    y = mask_temp[filename]['y']
    
    polygon = []
    for i in range(len(x)):
        polygon.append((x[i], y[i]))
    
    ImageDraw.Draw(masks).polygon(polygon, outline=idx+1, fill=idx+1)

masks = np.array(masks, dtype=np.uint16) # resulting masks
plt.imshow(masks, cmap='gray') # display ground-truth masks
plt.show()
np.save(os.path.join(root_path, 'M872956_Position9_CD3_masks'), masks) # save masks as .npy file
io.imsave(os.path.join(root_path, 'M926910_Position7_CD3_masks.png'), masks) # save masks as plot


# Split image/mask into training (5/6) and test (1/6)
root_path = '/Users/shan/Desktop/Paper/YFong/7.New/Result/kdata/images/single/CFL_P7_CD3'
img = io.imread(os.path.join(root_path, 'M926910_Position7_CD3_img.png'))
width = img.shape[1]
test_img = img[:, :(int(width/6)+1), :]
train_img = img[:, (int(width/6)+1):, :]

masks = np.load(os.path.join(root_path, 'M926910_Position7_CD3_masks.npy'))
width = masks.shape[1]
test_mask = masks[:, :(int(width/6)+1)]
train_mask = masks[:, (int(width/6)+1):]

rm_idx = np.unique(test_mask[:,-1]) # cells on the cut line for test image
#rm_idx = np.unique(train_mask[:,0]) # cells on the cut line for train image
bound_masks_idx = np.setdiff1d(np.unique(rm_idx),np.array([0]))
img_copy = train_img.copy()
mask_copy = train_mask.copy()
for idx in bound_masks_idx:
    print(idx)
    coor = np.where(mask_copy == idx)
    mask_copy[coor[0], coor[1]] = 0
    img_copy[coor[0], coor[1]] = 0

plt.imshow(img_copy); plt.show()
io.imsave(os.path.join(root_path, 'M926910_Position7_CD3_train_img.png'), img_copy) # for image with rgb
io.imsave(os.path.join(root_path, 'M926910_Position7_CD3_train_img_white.png'), img_copy[:,:,0]) # for image with white
plt.imshow(mask_copy, cmap='gray'); plt.show()
io.imsave(os.path.join(root_path, 'M926910_Position7_CD3_train_masks.png'), mask_copy) # for masks





#######################################################################################################





### 3. Compute TP, FP, FN and coloring FP
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from read_roi import read_roi_file # pip install read-roi
from PIL import Image, ImageDraw
from cellpose import utils, io

# utility
def compute_iou(mask_true, mask_pred):
    '''
    Compute the IoU for ground-truth mask (mask_true) and predicted mask (mask_pred).
    '''
    true_objects = (np.unique(mask_true))
    pred_objects = (np.unique(mask_pred))
    
    # Compute intersection between all objects
    # compute the 2D histogram of two data samples; it returns frequency in each bin
    # important to append n.inf otherwise the number of bins will be 1 less than the number of unique masks
    intersection = np.histogram2d(mask_true.flatten(), mask_pred.flatten(), bins=(np.append(true_objects, np.inf),np.append(pred_objects, np.inf)))[0] 
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(mask_true, bins=np.append(true_objects, np.inf))[0]
    area_pred = np.histogram(mask_pred, bins=np.append(pred_objects, np.inf))[0]
    area_true = np.expand_dims(area_true, -1) # makes true_objects * 1
    area_pred = np.expand_dims(area_pred, 0) # makes 1 * pred_objects
    
    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union
    return iou[1:, 1:] # exclude background; remove frequency for bin [0,1)

# import cp resulting masks
pred_name = 'M872956_Position8_CD8_test_img_cp_masks.png' # on volta
masks_name = 'M872956_Position8_CD8_test_masks.png'

masks = io.imread(masks_name)
masks_idx = np.setdiff1d(np.unique(masks), np.array([0])) # remove background 0
pred = io.imread(pred_name)
pred_idx = np.setdiff1d(np.unique(pred), np.array([0])) # remove background 0
iou = compute_iou(mask_true=masks, mask_pred=pred)

matches = iou >= 0.5
true_positives = np.sum(matches, axis=1) >= 1
sum(true_positives); tp_idx = masks_idx[true_positives]

false_positives = np.sum(matches, axis=0) == 0
sum(false_positives); fp_idx = pred_idx[false_positives]

false_negatives = np.sum(matches, axis=1) == 0
sum(false_negatives); fn_idx = masks_idx[false_negatives]

tp, fp, fn = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives))
tp / (tp + fp + fn)

# FP in Train5
total_idx = pred_idx
pred_fp = pred.copy()
for idx in total_idx:
    print(idx)
    if(sum(idx == fp_idx) == 0):
        temp = np.where(pred_fp == idx)
        pred_fp[temp[0], temp[1]] = 0

total_outlines = utils.masks_to_outlines(pred)
fp_outlines = utils.masks_to_outlines(pred_fp)

res = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
res[np.where(total_outlines)[0], np.where(total_outlines)[1], 0] = 255
res[np.where(total_outlines)[0], np.where(total_outlines)[1], 1] = 255
res[np.where(total_outlines)[0], np.where(total_outlines)[1], 2] = 255
res[np.where(fp_outlines)[0], np.where(fp_outlines)[1], 0] = 0
res[np.where(fp_outlines)[0], np.where(fp_outlines)[1], 2] = 0
plt.imshow(res); plt.show()
plt.imsave('M872956_Position10_CD3_test_img_cp_masks_outline_color.png', res)

# FN in GT masks
total_idx = masks_idx
masks_fn = masks.copy()
for idx in total_idx:
    print(idx)
    if(sum(idx == fn_idx) == 0):
        temp = np.where(masks_fn == idx)
        masks_fn[temp[0], temp[1]] = 0

total_outlines = utils.masks_to_outlines(masks)
fn_outlines = utils.masks_to_outlines(masks_fn)

res = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
res[np.where(total_outlines)[0], np.where(total_outlines)[1], 0] = 255
res[np.where(total_outlines)[0], np.where(total_outlines)[1], 1] = 255
res[np.where(total_outlines)[0], np.where(total_outlines)[1], 2] = 255
res[np.where(fn_outlines)[0], np.where(fn_outlines)[1], 0] = 0
res[np.where(fn_outlines)[0], np.where(fn_outlines)[1], 2] = 0
plt.imshow(res); plt.show()
plt.imsave('M872956_Position10_CD3_test_masks_outline_color.png', res)





#######################################################################################################





### Appendix ###
### 1. Split P8 CD3 training image into sub-training (1/2) and validation (1/2)
root_path = '/Users/shan/Desktop/Paper/YFong/7.New/Result/kdata/images/single/CD3_pos-8'
img = io.imread(os.path.join(root_path, 'train', 'M872956_Position8_CD3_train_img.png'))
width = img.shape[1]
val_img = img[:, :(int(width/2)+1), :]
subtrain_img = img[:, (int(width/2)+1):, :]

masks = io.imread(os.path.join(root_path, 'train', 'M872956_Position8_CD3_train_masks.png'))
width = masks.shape[1]
val_mask = masks[:, :(int(width/2)+1)]
subtrain_mask = masks[:, (int(width/2)+1):]

rm_idx = np.unique(val_mask[:,-1]) # cells on the cut line for test image
#rm_idx = np.unique(subtrain_mask[:,0]) # cells on the cut line for train image
bound_masks_idx = np.setdiff1d(np.unique(rm_idx),np.array([0]))
img_copy = subtrain_img.copy()
mask_copy = subtrain_mask.copy()
for idx in bound_masks_idx:
    print(idx)
    coor = np.where(mask_copy == idx)
    mask_copy[coor[0], coor[1]] = 0
    img_copy[coor[0], coor[1]] = 0

plt.imshow(img_copy); plt.show()
io.imsave(os.path.join(root_path, 'M872956_Position8_CD3_subtrain_img.png'), img_copy) # for image with rgb
io.imsave(os.path.join(root_path, 'M872956_Position8_CD3_subtrain_img_white.png'), img_copy[:,:,0]) # for image with white
plt.imshow(mask_copy, cmap='gray'); plt.show()
io.imsave(os.path.join(root_path, 'M872956_Position8_CD3_subtrain_masks.png'), mask_copy) # for masks



### 2. Create a new image by putting together 4 copies
# Remove masks that are across or on edges of images
root_path = '/Users/shan/Desktop/Paper/YFong/8.New/Result/kdata/images/single'
img = io.imread(os.path.join(root_path, 'CD8_pos-8/train', 'M872956_Position8_CD8_train_img.png'))
masks = io.imread(os.path.join(root_path, 'CD8_pos-8/train', 'M872956_Position8_CD8_train_masks.png'))

idx1 = np.unique(masks[0,:]) # first row
idx2 = np.unique(masks[-1,:]) # last row
idx3 = np.unique(masks[:,0]) # first column
idx4 = np.unique(masks[:,-1]) # last column
idx = np.union1d(np.union1d(np.union1d(idx1, idx2), idx3), idx4)
bound_masks_idx = np.setdiff1d(np.unique(idx),np.array([0]))

img_copy = img.copy()
masks_copy = masks.copy()
for idx in bound_masks_idx:
    print(idx)
    coor = np.where(masks_copy == idx)
    masks_copy[coor[0], coor[1]] = 0
    img_copy[coor[0], coor[1]] = 0

plt.imshow(img_copy); plt.show()
io.imsave(os.path.join(root_path, 'M872956_Position8_CD8_modified_train_img.png'), img_copy) # for image

plt.imshow(masks_copy, cmap='gray'); plt.show()
io.imsave(os.path.join(root_path, 'M872956_Position8_CD8_modified_train_masks.png'), masks_copy) # for masks



### 3. Putting together 4 copies
# For images
root_path = '/Users/shan/Desktop/Paper/YFong/8.New/Result/kdata/images/single'
img = io.imread(os.path.join(root_path, 'CD8_pos-8/train', 'M872956_Position8_CD8_modified_train_img.png'))

height = img.shape[0]
width = img.shape[1]
img_total = np.zeros((2*height, 2*width, 3), dtype='uint8')
img_total[:height, :width, ] = img
img_total[:height, width:, ] = img
img_total[height:, :width, ] = img
img_total[height:, width:, ] = img
plt.imshow(img_total); plt.show()
io.imsave(os.path.join(root_path, 'M872956_Position8_CD8_2x2copied_train_img.png'), img_total)

# for masks
root_path = '/Users/shan/Desktop/Paper/YFong/8.New/Result/kdata/images/single/'
masks = io.imread(os.path.join(root_path, 'CD8_pos-8/train', 'M872956_Position8_CD8_modified_train_masks.png'))

max_idx = masks.max()
masks_0 = masks
masks_1 = masks + (1*max_idx)
masks_1[np.where(masks_1 == max_idx)] = 0 # in masks_1, pixels having the value of max_idx are background
masks_2 = masks + (2*max_idx)
masks_2[np.where(masks_2 == 2*max_idx)] = 0 # in masks_2, pixels having the value of 2*max_idx are background
masks_3 = masks + (3*max_idx)
masks_3[np.where(masks_3 == 3*max_idx)] = 0 # in masks_3, pixels having the value of 3*max_idx are background

height = masks.shape[0]
width = masks.shape[1]
masks_total = np.zeros((2*height, 2*width), dtype='uint16')
masks_total[:height, :width] = masks_0
masks_total[:height, width:] = masks_1
masks_total[height:, :width] = masks_2
masks_total[height:, width:] = masks_3

# reorder the number of masks
total_masks_idx = np.union1d(np.union1d(np.union1d(np.unique(masks_0), np.unique(masks_1)), np.unique(masks_2)), np.unique(masks_3))
total_masks_idx = np.setdiff1d(total_masks_idx, np.array([0]))
for i,idx in enumerate(total_masks_idx):
    print(idx)
    masks_total[np.where(masks_total == idx)] = (i+1)

plt.imshow(masks_total, cmap='gray'); plt.show()
io.imsave(os.path.join(root_path, 'M872956_Position8_CD8_2x2copied_train_masks.png'), masks_total)
