# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 09:44:53 2022

@author: Youyi
"""

import cv2, os, glob, torch, random
import numpy as np
from skimage import io
from syotil import normalize99
from scipy import ndimage as ndi
from torch.utils.data import Dataset



### Dataset and DataLoader (for training)
class TrainDataset(Dataset):
    def __init__(self, root, data_source):
        self.root = root
        self.data_source = data_source
                
        # Load image and mask files, and sort them
        self.img_paths = sorted(glob.glob(os.path.join(self.root, '*_img.*')))
        self.masks =     sorted(glob.glob(os.path.join(self.root, '*_masks.*')))
    
    def __getitem__(self, idx):
        '''Get the image and the mask'''
        # image
        img_path = self.img_paths[idx]
        img = io.imread(img_path)
        
        if self.data_source.lower()=="cellpose":
            # cellpose images are [height, width, [nuclear, cyto, empty]] 
            # train with cellpose cyto image        
            img=img[:,:,1] 
        elif self.data_source.lower()=="tissuenet":
            # tissuenet images are [height, width, [empty, nuclear, cyto]]        
            # train with tissuenet nuclear image
            img=img[:,:,1] 
        elif self.data_source.lower()=="kaggle":
            # Kaggle images are [height, width, [R,G,B,alpha]]        
            # traing with Kaggle red channel
            img=img[:,:,0] 
        elif self.data_source.lower()=="k":
            # K images in training_resized are [height, width]        
            img=img 

        img=np.expand_dims(img, axis=0)
        
        img = normalize_img(img) # normalize image
        
        # mask
        mask_path = self.masks[idx]
        mask = io.imread(mask_path)
        mask = np.array(mask) # convert to a numpy array
        
        # Transformation
        img_trans, mask_trans = random_rotate_and_resize(X=[img], Y=[mask], scale_range=1., xy=(img.shape[1],img.shape[2]), 
                                        do_flip=True, rescale=[1], random_per_image=True) # rescale value can be changed; xy=(args.patch_size,args.patch_size)
        while len(np.unique(mask_trans)) == 1: # if the patch does not have any gt mask, redo transformation
            img_trans, mask_trans = random_rotate_and_resize(X=[img], Y=[mask], scale_range=1., xy=(img.shape[1],img.shape[2]), 
                                        do_flip=True, rescale=[1], random_per_image=True) # not sure if another seed should be set here; xy=(args.patch_size,args.patch_size)
        
        # Split a mask map into multiple binary mask map
        obj_ids = np.unique(mask_trans) # get list of gt masks, e.g. [0,1,2,3,...]
        obj_ids = obj_ids[1:] # remove background 0
        masks = mask_trans == obj_ids[:, None, None] # masks contain multiple binary mask map
        
        # Get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            boxes.append([xmin, ymin, xmax, ymax])
        
        # Convert everything into a torch.Tensor
        img = torch.as_tensor(img_trans, dtype=torch.float32) # for image
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # all 1
        masks = torch.as_tensor(masks, dtype=torch.uint8) # dtpye needs to be changed to uint16 or uint32
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # calculating height*width for bounding boxes
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd; if instances are crowded in an image, 1
        
        # Remove too small box (too small gt box makes an error in training)
        keep_box_idx = torch.where(area > 10) # args.min_box_size
        boxes = boxes[keep_box_idx]
        labels = labels[keep_box_idx]
        masks = masks[keep_box_idx]
        image_id = image_id
        area = area[keep_box_idx]
        iscrowd = iscrowd[keep_box_idx]
        
        # Required target for the Mask R-CNN
        target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
                }
        
        return img, target
    
    def __len__(self):
        return len(self.img_paths)




### Dataset and DataLoader (prediction)
class TestDataset(Dataset):
    def __init__(self, root, data_source):
        self.root = root
        self.data_source = data_source
        
        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = sorted(glob.glob(os.path.join(self.root, '*_img.png')))
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]

        img = io.imread(img_path)
        # img = Image.open(img_path).convert("RGB") # to see pixel values, do np.array(img)
        
        if self.data_source.lower()=="cellpose":
            # cellpose images are [height, width, [nuclear, cyto, empty]] 
            # train with cellpose cyto image        
            img=img[:,:,1] 
        elif self.data_source.lower()=="tissuenet":
            # tissuenet images are [height, width, [empty, nuclear, cyto]]        
            # train with tissuenet nuclear image
            img=img[:,:,1] 
        elif self.data_source.lower()=="kaggle":
            # Kaggle images are [height, width, [R,G,B,alpha]]        
            # traing with Kaggle red channel
            img=img[:,:,0] 
        elif self.data_source.lower()=="k":
            # K images in test_images are [height, width]        
            img=img 

        img=np.expand_dims(img, axis=0)
        
        img = normalize_img(img) # normalize image
        
        # Convert image into tensor
        img = torch.as_tensor(img, dtype=torch.float32) # for image
        
        return {'image': img, 'image_id': idx}
    
    def __len__(self):
        return len(self.imgs)





def random_rotate_and_resize(X, Y=None, scale_range=1., xy = (448,448),
                             do_flip=True, rescale=None, random_per_image=True):
    """ augmentation by random rotation and resizing
        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)
        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        xy: tuple, int (optional, default (224,224))
            size of transformed images to return
        do_flip: bool (optional, default True)
            whether or not to flip images horizontally
        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations
        random_per_image: bool (optional, default True)
            different random rotate and resize per image
        Returns
        -------
        imgi: ND-array, float
            transformed image in array [nchan x xy[0] x xy[1]]
        labeled: ND-array, float
            transformed label in array [nchan x xy[0] x xy[1]]
        
        Notes
        -----
        1. X should be nomalized before iputting this function.
        2. Some gt masks transformed by this function can have the same pixel values in x-axis or y-axis. E.g. boxes=[0,1,0,10] or [5,5,10,5].
        3. Some patch generated by this funciton can have no gt masks (all pixel values are 0)
    """
    nimg = len(X)
    if X[0].ndim>2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi  = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)
    
    lbl = []
    if Y is not None:
        if Y[0].ndim>2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)
    
    scale_range = max(0, min(2, float(scale_range)))
    scale = np.ones(nimg, np.float32)
    
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]
        
        if random_per_image or n==0:
            # generate random augmentation parameters
            flip = np.random.rand()>.5
            theta = 0 # np.random.rand() * np.pi * 2
            scale[n] = 1 # (1-scale_range/2) + scale_range * np.random.rand()
            if rescale is not None:
                scale[n] *= 1. / rescale[n]
            dxy = np.maximum(0, np.array([Lx*scale[n]-xy[1],Ly*scale[n]-xy[0]]))
            dxy = (np.random.rand(2,) - .5) * dxy
            
            # create affine transform
            cc = np.array([Lx/2, Ly/2])
            cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
            pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
            pts2 = np.float32([cc1,
                    cc1 + scale[n]*np.array([np.cos(theta), np.sin(theta)]),
                    cc1 + scale[n]*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
            M = cv2.getAffineTransform(pts1,pts2)
        
        img = X[n].copy()
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim<3:
                labels = labels[np.newaxis,:,:] # increase dim
        
        if flip and do_flip:
            img = img[..., ::-1] # horizontal flip 
            if Y is not None:
                labels = labels[..., ::-1]
        
        # transform image, one channel at a time
        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
            imgi[n,k] = I
        
        # transform masks, one class of objects at a time
        if Y is not None:
            for k in range(nt):
                if k==0:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=cv2.INTER_NEAREST)
                else:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR) # no need for mask rcnn
        
    return imgi[0], lbl[0]




def normalize_img(img):
    """ normalize each channel of the image so that so that 0.0=0 percentile and 1.0=100 percentile of image intensities
    
    Parameters
    ------------
    img: ND-array (at least 3 dimensions)
    
    Returns
    ---------------
    img: ND-array, float32
        normalized image of same size
    """
    if img.ndim<3:
        error_message = 'Image needs to have at least 3 dimensions'
        #transforms_logger.critical(error_message)
        raise ValueError(error_message)
    
    img = img.astype(np.float32)
    for k in range(img.shape[0]):
        # ptp can still give nan's with weird images
        i100 = np.percentile(img[k],100)
        i0 = np.percentile(img[k],0)
        if i100 - i0 > +1e-3: #np.ptp(img[k]) > 1e-3:
            img[k] = normalize99(img[k])
        else:
            img[k] = 0
    return img



# the first three functions are copied from cellpose

### Random seed
def fix_all_seeds_torch(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # PYTHONHASHSEED controls set operations
    # the line below does not actually work, it needs to be set in bash, e.g. export PYTHONHASHSEED=1
    os.environ['PYTHONHASHSEED'] = str(seed)



#################################################
# copied from tf2 mrcnn model.py and modified to delete calls to resize_image

def mold_inputs(images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matrices [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matrices:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE)
        molded_image = mold_image(molded_image, self.config)
        # Build image_meta
        image_meta = compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows

def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                      image_shape, window):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    mrcnn_mask: [N, height, width, num_classes]
    original_image_shape: [H, W, C] Original image shape before resizing
    image_shape: [H, W, C] Shape of the image after resizing and padding
    window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
            image is excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = utils.norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty(original_image_shape[:2] + (0,))

    return boxes, class_ids, scores, full_masks
