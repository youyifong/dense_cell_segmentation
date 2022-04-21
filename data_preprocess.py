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
    - Original TissueNet image with 2 channels is converted to an image with RGB channels.
    - One thing is that rescaling the intensity of pixels is included in this function, but not sure it is necessary. 
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


# Convert image with two channels to image with RGB channels
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
