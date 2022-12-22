# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 09:44:53 2022

@author: Youyi
"""

import cv2, os, glob, torch, random
import numpy as np
from skimage import io
from syotil import normalize99
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
        # print(f"idx {idx}")
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
        img_trans, mask_trans = random_rotate_and_resize     (X=[img], Y=[mask], scale_range=0, do_flip=False, do_rotate=False,
                                                              xy=(img.shape[1],img.shape[2]), # no cropping
                                                              rescale=None, random_per_image=True)
        # if the patch does not have any gt mask, redo transformation
        while len(np.unique(mask_trans)) == 1: 
            img_trans, mask_trans = random_rotate_and_resize (X=[img], Y=[mask], scale_range=0, do_flip=False, do_rotate=False,
                                                              xy=(img.shape[1],img.shape[2]),  # no cropping
                                                              rescale=None, random_per_image=True)
        
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
                             do_flip=True, rescale=None, random_per_image=True, do_rotate=True):
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
    scale_range = max(0, min(2, float(scale_range)))
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
    
    scale = np.ones(nimg, np.float32)
    
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]
        
        if random_per_image or n==0:
            # generate random augmentation parameters
            flip_horizontally = np.random.rand()>.5
            flip_vertically = np.random.rand()>.5
            if do_rotate:
                theta = np.random.rand() * np.pi * 2
            else:
                theta = 0            
            scale[n] = (1-scale_range/2) + scale_range * np.random.rand()
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
        
        if flip_horizontally and do_flip:
            img = img[..., ::-1] 
            if Y is not None:
                labels = labels[..., ::-1]
        
        if flip_vertically and do_flip:
            img = img[..., ::-1, :] 
            if Y is not None:
                labels = labels[..., ::-1, :]
        
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
# copied and modified from CellSeg CVsegmenter.py

# img needs to be of shape ... x height x width
def crop_with_overlap(img, overlap, nrows, ncols):
    crop_height, crop_width = img.shape[-2]//nrows, img.shape[-1]//ncols
    crops = []
    for row in range(nrows):
        for col in range(ncols):
            x1, y1, x2, y2 = col*crop_width, row*crop_height, (col+1)*crop_width, (row+1)*crop_height
            x1, x2, y1, y2 = get_overlap_coordinates(overlap, nrows, ncols, row, col, x1, x2, y1, y2)
            crops.append(img[..., y1:y2, x1:x2])
    print("Dividing image into", len(crops), "crops with", nrows, "rows and", ncols, "columns")
    return crops


def get_overlap_coordinates(overlap, rows, cols, i, j, x1, x2, y1, y2):
    half = overlap // 2
    if i != 0:
        y1 -= half
    if i != rows - 1:
        y2 += half
    if j != 0:
        x1 -= half
    if j != cols - 1:
        x2 += half
    return (x1, x2, y1, y2)

