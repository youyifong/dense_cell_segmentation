'''
cd maskrcnn_train/train/train1
ml Anaconda3; ml CUDA
'''

### Library
import os, time, warnings, datetime
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


### Set Directory
root = '.'
save_path = os.path.join(root, 'models')
if not os.path.isdir(save_path):
    os.makedirs(save_path)


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
            height, width = image.shape[-2:] # image shape is (channels, height, width)
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


### Dataset and DataLoader (for training)
height_train = 1040 # height for our training image
width_train = 1159 # width for our training image
normalize = False 
resnet_mean = (0.485, 0.456, 0.406)
resnet_std = (0.229, 0.224, 0.225)

class TrainDataset(Dataset):
    def __init__(self, root, transforms=None, resize=False):
        self.root = root
        self.transforms = transforms
        
        # Resize
        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(height_train * resize)
            self.width = int(width_train * resize)
        else:
            self.height = height_train
            self.width = width_train
        
        # Load image and mask files, and sort them
        self.imgs = sorted(glob.glob('*_img.png'))
        self.masks = sorted(glob.glob('*_masks.png'))
    
    def __getitem__(self, idx):
        '''Get the image and the mask'''
        # Image
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB") # to see pixel values, do np.array(img)
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        
        # Mask
        mask_path = os.path.join(self.root, self.masks[idx])
        mask = Image.open(mask_path)
        if self.should_resize:
            mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = np.array(mask) # convert to a numpy array
        
        # Split a mask map into multiple binary mask map
        obj_ids = np.unique(mask) # get list of gt masks, e.g. [0,1,2,3,...]
        obj_ids = obj_ids[1:] # remove background 0
        masks = mask == obj_ids[:, None, None] # masks contain multiple binary mask maps
        
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
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # all 1
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # calculating height*width for bounding boxes        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd; if instances are crowded in an image, 1
        
        # Required target for the Mask R-CNN
        target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
                }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target) # for target, doing transforms only works for boxes and masks
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)


### Define train and test dataset
train_ds = TrainDataset(root=root, transforms=get_transform(train=True)) # transformation for training
#train_ds[0]


# Define Dataloader
batch_size = 8
if gpu:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) # on linux
    n_batches = len(train_dl)
else:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) # on local
    n_batches = len(train_dl)


### Define Mask R-CNN Model
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
#model.state_dict()

# Load pre-trained model 
#device = torch.device("cpu") 
#PATH = "/Users/shan/Desktop/maskrcnn_resnet50_ep12.pth"
#model.load_state_dict(torch.load(PATH, map_location=device))
#print(model.state_dict())


### Declare which parameteres are trained or not trained (freeze)
# Print parameters in mask r-cnn model 
for name, param in model.named_parameters():
    print("Name: ", name, "Requires_Grad:", param.requires_grad)

# If requires_grad = false, you are freezing the part of the model as no changes happen to its parameters. 
# All layers have the parameters modified during training as requires_grad is set to true.
for param in model.parameters(): 
    param.requires_grad = True


### Training stage (training loop) start from here!
model.train()
num_epochs = 500 # [8, 12]
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
        
        if batch_idx % 50 == 0:
            print(f"[Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}, Mask-only loss: {loss_mask:7.3f}")
    
    if use_scheduler:
        lr_scheduler.step()
    
    # Train losses
    train_loss = loss_accum / n_batches
    train_loss_mask = loss_mask_accum / n_batches
    
    elapsed = time.time() - time_start
    
    # Save the trained parameters
    if epoch%100 == 1:
        torch.save(model.state_dict(), os.path.join(save_path, 'maskrcnn_trained_model' + d.strftime("_%Y_%m_%d_%H_%M_%S") + '.pth'))
    
    # Print loss
    if epoch==1 or epoch==5 or epoch%10==0:
        prefix = f"[Epoch {epoch:2d} / {num_epochs:2d}]"
        print(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}, Train loss: {train_loss:7.3f}, [{elapsed:.0f} secs]")





