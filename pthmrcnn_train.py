'''
ml Python/3.9.6-GCCcore-11.2.0
ml IPython/7.26.0-GCCcore-11.2.0

ml cuDNN/8.2.2.26-CUDA-11.4.1 (works with torchvision 0.13.1+cu102)
or
ml cuDNN/8.4.1.50-CUDA-11.7.0 (works with torchvision 0.14.1+cu117)


tv013
on volta 
    32 sec/epoch when using pretrained=True
    32 s/e when using weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    40 s/e when using weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 
on gizmoj23: 
    51s/epoch when using pretrained=True

tv014
on gizmoj23: 
    49s/epoch

'''

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import os, time, datetime # warnings
import numpy as np
import glob
import cv2
#import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io
from syotil import normalize99, fix_all_seeds_torch

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# import sys
# sys.stdout = open(os.devnull, "w") # to not batch print


### Check whether gpu is available
if torch.cuda.is_available() :
    gpu = True
    device = torch.device('cuda')
else :
    gpu = False
    device = torch.device('cpu')

#device = torch.device('cpu') # try this when cuda is out of memory


### Set arguments
parser = argparse.ArgumentParser()

# Kaggle
data_source="Kaggle"
parser.add_argument('--dir', default='/fh/fast/fong_y/Kaggle_2018_Data_Science_Bowl_Stage1/train', type=str, help='folder directory containing training images')
parser.add_argument('--pretrained_model', required=False, default='pretrained', type=str, help='pretrained model to use for starting training')
parser.add_argument('--batch_size', default=8, type=int, help='batch size. Default: %(default)s')
parser.add_argument('--n_epochs',default=500, type=int, help='number of epochs. Default: %(default)s')

# # K's train
# data_source="K"
# parser.add_argument('--dir', default='/home/yfong/deeplearning/dense_cell_segmentation/images/training_resized/', type=str, help='folder directory containing training images')
# parser.add_argument('--pretrained_model', required=False, default='/fh/fast/fong_y/Kaggle_2018_Data_Science_Bowl_Stage1/train/models0/maskrcnn_trained_model_2022_12_17_10_50_30.pth', type=str, help='pretrained model to use for starting training')
# parser.add_argument('--batch_size', default=1, type=int, help='batch size. Default: %(default)s')
# parser.add_argument('--n_epochs',default=500, type=int, help='number of epochs. Default: %(default)s')

parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use. Default: %(default)s')

parser.add_argument('--normalize', action='store_true', help='normalization of input image in training (False by default)')
parser.add_argument('--patch_size', default=448, type=int, help='path size. Default: %(default)s')
parser.add_argument('--min_box_size', default=10, type=int, help='minimum size of gt box to be considered for training. Default: %(default)s')
parser.add_argument('--box_detections_per_img', default=500, type=int, help='maximum number of detections per image, for all classes. Default: %(default)s')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)


fix_all_seeds_torch(args.gpu_id)


### Set Directory
root = args.dir
save_path = os.path.join(root, 'models'+str(args.gpu_id))
if not os.path.isdir(save_path):
    os.makedirs(save_path)


### Utility

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
                labels = labels[np.newaxis,:,:]
        
        if flip and do_flip:
            img = img[..., ::-1] # flip is actually done, but description of this function does not mention random flipping
            if Y is not None:
                labels = labels[..., ::-1]
        
        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
            imgi[n,k] = I
        
        # pre-processing for mask: convert labels (mask map) into binary mask map having value 1 (for mask) or 0 (for background)
        labels_bi = labels.copy()
        labels_bi[0][np.where(labels_bi[0] != 0)] = 1
        
        if Y is not None:
            for k in range(nt):
                if k==0:
                    lbl[n,k] = cv2.warpAffine(labels_bi[k], M, (xy[1],xy[0]), flags=cv2.INTER_NEAREST)
                else:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR) # no need for mask rcnn
        
        # post-processing for masks: restore binary mask map into mask map having values 0, 1, 2, ...
        labeled, n = ndi.label(lbl[0])
    
    return imgi[0], labeled[0]


### Dataset and DataLoader (for training)
class TrainDataset(Dataset):
    def __init__(self, root, data_source):
        self.root = root
        self.data_source = data_source
                
        # Load image and mask files, and sort them
        self.img_paths = sorted(glob.glob(os.path.join(self.root, '*_img.*')))
        self.masks = sorted(glob.glob(os.path.join(self.root, '*_masks.*')))
    
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
        keep_box_idx = torch.where(area > args.min_box_size) # default min_box_size is 10
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


### Define train and test dataset
train_ds = TrainDataset(root=root, data_source=data_source)
#train_ds[0]


# Define Dataloader
batch_size = args.batch_size
if gpu:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) # on linux
    n_batches = len(train_dl)
else:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) # on local
    n_batches = len(train_dl)



### Define Mask R-CNN Model
# normalize
if args.normalize:
    resnet_mean = (0.485, 0.456, 0.406)
    resnet_std = (0.229, 0.224, 0.225)

box_detections_per_img = args.box_detections_per_img # default is 100, but 539 is used in a reference


# smaller min size leads to faster training
if data_source.lower()=="kaggle":
    min_size=448
elif data_source.lower()=="k":
    min_size=112

# initial weight for training
if args.pretrained_model == 'pretrained':
    initial_weight = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 #'COCO_V1'
else:
    initial_weight = None


def get_model():
    num_classes = 2 # background or foreground (cell)
    
    # normalization for input image (training DataLoader already does normalization for input image)
    if args.normalize:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=initial_weight, # pretrained weights on COCO data
                min_size = 137,
                max_size = 720,
                box_detections_per_img=box_detections_per_img,
                image_mean=resnet_mean, # mean values used for input normalization
                image_std=resnet_std # std values used for input normalization
                )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
                weights=initial_weight,
                # pretrained=True,
                min_size = min_size, # IMAGE_MIN_DIM
                max_size = min_size, # IMAGE_MAX_DIM
                box_score_thresh=0.7, # DETECTION_MIN_CONFIDENCE
                rpn_pre_nms_top_n_train=1000, # RPN_NMS_ROIS_TRAINING
                rpn_pre_nms_top_n_test=2000, # RPN_NMS_ROIS_INFERENCE
                rpn_post_nms_top_n_train=1000, # RPN_NMS_ROIS_TRAINING
                rpn_post_nms_top_n_test=2000, # RPN_NMS_ROIS_INFERENCE
                rpn_nms_thresh=0.9, # RPN_NMS_THRESHOLD
                rpn_batch_size_per_image=1500, # RPN_TRAIN_ANCHORS_PER_IMAGE
                box_batch_size_per_image=300, # TRAIN_ROIS_PER_IMAGE
                box_detections_per_img=box_detections_per_img # DETECTION_MAX_INSTANCE
                )
    
    # get the number of inpute features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes) # a value is changed from 91 to 2
    
    return model

model = get_model() # get mask r-cnn
model.to(device)
#model.state_dict()


# Load pre-trained model 
if args.pretrained_model != 'pretrained':
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    # print(model.state_dict())


### Declare which parameteres are trained or not trained (freeze)
# Print parameters in mask r-cnn model 
#for name, param in model.named_parameters():
#    print("Name: ", name, "Requires_Grad:", param.requires_grad)


# If requires_grad = false, you are freezing the part of the model as no changes happen to its parameters. 
# All layers have the parameters modified during training as requires_grad is set to true.
for param in model.parameters(): 
    param.requires_grad = True


### Training stage (training loop) start from here!
model.train()
num_epochs = args.n_epochs
momentum = 0.9
learning_rate = 0.001
weight_decay = 0.0005
use_scheduler = False # scheduler
d = datetime.datetime.now()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(1, num_epochs+1):
    time_start = time.time()
    loss_accum = 0.0 # sum of total losses
    loss_mask_accum = 0.0 # sum of mask loss
    
    for batch_idx, (images, targets) in enumerate(train_dl, 1): # images, targets = next(iter(train_dl))
        # predict
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # k:key, v:value
        
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values()) # sum of losses
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        loss_mask = loss_dict['loss_mask'].item()
        loss_accum += loss.item()
        loss_mask_accum += loss_mask
                
        if batch_idx % 30 == 0:
            print(f"Batch {batch_idx}/{n_batches} Batch mask-only loss: {loss_mask:5.3f}, train loss: {loss.item():5.3f}")
    
    if use_scheduler:
        lr_scheduler.step()
    
    # Train losses
    train_loss = loss_accum / n_batches
    train_loss_mask = loss_mask_accum / n_batches
    
    elapsed = time.time() - time_start
    
    # Print loss
    # if epoch==1 or epoch==5 or epoch%10==0:
    prefix = f"[Epoch {epoch}/{num_epochs}]"
    print(f"{prefix} Train mask-only loss: {train_loss_mask:5.3f}, Train loss: {train_loss:5.3f}, [{elapsed:.0f} secs]")

    # Save the trained parameters
    if epoch%50 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'maskrcnn_trained_model' + d.strftime("_%Y_%m_%d_%H_%M_%S") + "_"+ str(epoch) + '.pth'))
    
