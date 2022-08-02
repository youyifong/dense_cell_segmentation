'''
cd maskrcnn_train
ml Anaconda3; ml CUDA
'''

### Library
import os, time
import numpy as np
import random
from PIL import Image

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
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
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
height = 536
width = 559
normalize = False 
resnet_mean = (0.485, 0.456, 0.406)
resnet_std = (0.229, 0.224, 0.225)

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None, resize=False):
        self.root = root
        self.transforms = transforms
        
        # Resize
        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(height * resize)
            self.width = int(width * resize)
        else:
            self.height = height
            self.width = width
        
        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def __getitem__(self, idx):
        '''Get the image and the mask'''
        # Image
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB") # to see img, do np.array(img)
        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        
        # Mask
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        mask = Image.open(mask_path)
        if self.should_resize:
            mask = mask.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = np.array(mask) # convert PIL into a numpy array
        
        # Split each mask into a set of binary masks
        obj_ids = np.unique(mask) # unique gt masks
        obj_ids = obj_ids[1:] # remove background 0
        masks = mask == obj_ids[:, None, None] # split
        
        # Get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # calculating height*width for bounding boxes        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd
        
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
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)


### Define train and test dataset
train_ds = PennFudanDataset(root='PennFudanPed', transforms=get_transform(train=True)) # transformation for training
#train_ds[0]
test_ds = PennFudanDataset(root='PennFudanPed', transforms=get_transform(train=False)) # not transformation for test
#test_ds[0]
    
# Split data into train and test
indices = torch.randperm(len(train_ds)).tolist()
train_ds = torch.utils.data.Subset(train_ds, indices[:-50]) # 120 training data
test_ds = torch.utils.data.Subset(test_ds, indices[-50:]) # 50 test data

# Define Dataloader
batch_size = 8
#torch.set_default_tensor_type("torch.FloatTensor") # on my local laptop
if gpu:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) # on linux
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) # on linux
    n_batches = len(train_dl)
else:
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) # on local
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) # on linux
    n_batches = len(train_dl)


### Define Mask R-CNN Model
box_detections_per_img = 539

def get_model():
    num_classes = 2 # this is just a dummy value for the classification head
    
    if normalize:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True, # pretrained weights on COCO data
                box_detections_per_img=box_detections_per_img,
                image_mean=resnet_mean,
                image_std=resnet_std
                )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True, # pretrained weights on COCO data
                box_detections_per_img=box_detections_per_img
                )
    
    # get the number of inpute features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

model = get_model() # get mask r-cnn
model.to(device)
#model.state_dict()

# Load pre-trained model 
#device = torch.device("cpu") 
#PATH = "/Users/shan/Desktop/maskrcnn_resnet50_ep12.pth"
#model.load_state_dict(torch.load(PATH, map_location=device)) # pre-trained model on livecell
#print(model.state_dict())

# Try removing this for
for param in model.parameters():
    param.requires_grad = True


### Training stage (training loop) start from here!
model.train()
num_epochs = 8 # [8, 12]
momentum = 0.9
learning_rate = 0.001
weight_decay = 0.0005
use_scheduler = False # scheduler

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(1, num_epochs+1):
    print(f"Starting epoch {epoch} of {num_epochs}")
    
    time_start = time.time()
    loss_accum = 0.0 # sum of total losses
    loss_mask_accum = 0.0 # sum of mask loss
    
    for batch_idx, (images, targets) in enumerate(train_dl, 1): # images, targets = next(iter(train_dl))
        # predict
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # use [targets] if an error is occurred
        
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
    
    torch.save(model.state_dict(), f"pytorch_model-e{epoch}.bin")
    prefix = f"[Epoch {epoch:2d} / {num_epochs:2d}]"
    print(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}")
    print(f"{prefix} Train loss: {train_loss:7.3f}, [{elapsed:.0f} secs]")

# Save the trained parameters
device = torch.device("cuda")
model.to(device)
PATH = '/home/shan/maskrcnn_resnet50_ep8_livecell_train.pth'
torch.save(model.state_dict(), PATH)


### Dataset and DataLoader (prediction)
class TestDataset(Dataset):
    def __init__(self, root, transforms=None, resize=False):
        self.root = root
        self.transforms = transforms
        
        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB") # to see img, do np.array(img)
        
        if self.transforms is not None:
            image, _ = self.transforms(image=img, target=None)
        return {'image': image, 'image_id': idx}
    
    def __len__(self):
        return len(self.imgs)

test_ds = TestDataset(root='PennFudanPed', transforms=get_transform(train=False)) # not transformation for test
#test_ds[0]


## Utilities for prediction
def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask


### Prediction
model.eval()
masks = []
min_score = 0.59
mask_threshold = 0.5
for sample in test_ds:
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
        
    mask_map = np.zeros((height, width), dtype='int16')
    for idx, ind_mask_map in enumerate(previous_masks):
        tmp = np.where(ind_mask_map[0,:,:])
        mask_map[tmp[0], tmp[1]] = idx+1
    masks.append(mask_map)

masks





### Appendix ###
# Model option 1) Fine-tuning from a pretrained model
import torchvision # pip install torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # load a model pre-trained on COCO
num_classes = 2  # 1 class (person) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # replace the pre-trained head with a new one

# Model option 2) Modifying the model to add a different backbone
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

backbone = torchvision.models.mobilenet_v2(pretrained=True).features # load a pre-trained model for classification and return only the features
backbone.out_channels = 1280 # FasterRCNN needs to know the number of output channels in a backbone. For mobilenet_v2, it's 1280

# let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios. We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will use to perform the region of interest cropping, as well as the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to be [0]. More generally, the backbone should return an OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

### Referrence ###
### Library
import collections
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToPILImage

# Utilities
def rle_decode(mask_rle, shape, color=1):
    '''
    To make masks by using annotations(prediction)
    '''    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1 
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)

# Utilities
def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))
