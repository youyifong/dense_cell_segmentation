'''
cd maskrcnn_train/train/train1
ml Anaconda3; ml CUDA
'''

### Library
import os, time, warnings
import numpy as np
import random
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


### Check whether gpu is available
if torch.cuda.is_available() :
    gpu = True
    device = torch.device('cuda')
else :
    gpu = False
    device = torch.device('cpu')


### Random seed
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
fix_all_seeds(123)


### Set Directory and test files
root = '.'
imgs = sorted(glob.glob(os.path.join(root, 'test/*_img.png'))) # test images
filenames = []
for item in imgs:
    filenames.append(item.replace('./test/', '').replace('.png', ''))


### Define data augmentation functions
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:] # image = (channels, h, w)
            image = image.flip(-2) # -2 does vertical flip for each channel
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]] # flip ymin and ymax
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1) # -1 does horizontal flip for each channel
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]] # flip xmin and xmax
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, resnet_mean, resnet_std)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    
    # Normalize
    if normalize:
        transforms.append(Normalize())
    
    # Data augmentation for training dataset (can add other transformations)
    if train:
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))
    
    return Compose(transforms)


### Dataset and DataLoader (prediction)
normalize = False
height_test = 1040
width_test = 233

class TestDataset(Dataset):
    def __init__(self, root, transforms=None, resize=False):
        self.root = root
        self.transforms = transforms
        
        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = sorted(glob.glob('test/*_img.png'))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB") # to see img, do np.array(img)
        
        if self.transforms is not None:
            image, _ = self.transforms(image=img, target=None) # not vertical/horizontal flips, but still normalization can be done
        return {'image': image, 'image_id': idx}
    
    def __len__(self):
        return len(self.imgs)

test_ds = TestDataset(root=root, transforms=get_transform(train=False)) # not transformation for test
#test_ds[0]


### Define Mask R-CNN Model
normalize = False
resnet_mean = (0.485, 0.456, 0.406)
resnet_std = (0.229, 0.224, 0.225)
box_detections_per_img = 539 # maximum number of detections per image, for all classes.

def get_model():
    num_classes = 2 # background or foreground (cell)
    
    if normalize:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True, # pretrained weights on COCO data
                box_detections_per_img=box_detections_per_img,
                image_mean=resnet_mean, # not sure how image_mean and image_std are used
                image_std=resnet_std
                )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True,
                box_detections_per_img=box_detections_per_img # we may ingnore to set box_detections_per_img
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


### Load pre-trained model 
#device = torch.device("cpu") 
#PATH = "/Users/shan/Desktop/maskrcnn_resnet50_ep12.pth"
#model.load_state_dict(torch.load(PATH, map_location=device))
#print(model.state_dict())


### Utilities for prediction
def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask


### Prediction
model.eval()
masks = []
min_score = 0.3
mask_threshold = 0.5
for idx, sample in enumerate(test_ds):
    img = sample['image']
    image_id = sample['image_id']
    with torch.no_grad():
        result = model([img.to(device)])[0]
    
    previous_masks = []
    for i, mask in enumerate(result['masks']):
        # filter-out low-scoring results
        score = result['scores'][i].cpu().item()
        if score < min_score:
            continue
        
        # keep only highly likely pixels
        mask = mask.cpu().numpy()
        binary_mask = mask > mask_threshold
        binary_mask = remove_overlapping_pixels(binary_mask, previous_masks) # if two masks are overlapped, remove the overlapped pixels?
        previous_masks.append(binary_mask)
        
    mask_map = np.zeros((height_test, width_test), dtype='int16')
    for val, ind_mask_map in enumerate(previous_masks):
        tmp = np.where(ind_mask_map[0,:,:])
        mask_map[tmp] = val+1
    
    # save masks
    if mask_map.max() < 2**16:
        mask_map = mask_map.astype(np.uint16) 
        cv2.imwrite(os.path.join(root,'test',filenames[idx] + '_mr_masks.png'), mask_map)
    else:
        warnings.warn('found more than 65535 masks in each image, cannot save PNG, saving as TIF')








