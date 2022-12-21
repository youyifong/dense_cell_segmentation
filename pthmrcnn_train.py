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

scaling is controlled by 
  scale = torch.min(self_min_size / min_size, self_max_size / max_size)    
ref: https://github.com/pytorch/vision/blob/657c0767c5ca5564c8b437ac44263994c8e01352/torchvision/models/detection/transform.py#L74

'''

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import os, time, datetime # warnings
from syotil import fix_all_seeds_torch
from pthmrcnn_utils import TrainDataset

#import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

### Set arguments
parser = argparse.ArgumentParser()

# Kaggle
data_source="Kaggle"
parser.add_argument('--dir', default='/fh/fast/fong_y/Kaggle_2018_Data_Science_Bowl_Stage1/train', type=str, help='folder directory containing training images')
parser.add_argument('--pretrained_model', required=False, default='coco', type=str, help='pretrained model to use for starting training')
parser.add_argument('--batch_size', default=4, type=int, help='batch size. Default: %(default)s')
parser.add_argument('--n_epochs',default=100, type=int, help='number of epochs. Default: %(default)s')

# # K's train
# data_source="K"
# parser.add_argument('--dir', default='/home/yfong/deeplearning/dense_cell_segmentation/images/training_resized/', type=str, help='folder directory containing training images')
# parser.add_argument('--pretrained_model', required=False, default='/fh/fast/fong_y/Kaggle_2018_Data_Science_Bowl_Stage1/train/models0/maskrcnn_trained_model_2022_12_17_10_50_30.pth', type=str, help='pretrained model to use for starting training')
# parser.add_argument('--batch_size', default=1, type=int, help='batch size. Default: %(default)s')
# parser.add_argument('--n_epochs',default=500, type=int, help='number of epochs. Default: %(default)s')

parser.add_argument('--gpu_id', default=2, type=int, help='which gpu to use. Default: %(default)s')

parser.add_argument('--normalize', action='store_true', help='normalization of input image in training (False by default)')
parser.add_argument('--min_box_size', default=10, type=int, help='minimum size of gt box to be considered for training. Default: %(default)s')
parser.add_argument('--box_detections_per_img', default=500, type=int, help='maximum number of detections per image, for all classes. Default: %(default)s')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
### Check whether gpu is available
if torch.cuda.is_available() :
    gpu = True
    device = torch.device('cuda') # this will use the visible gpu
else :
    gpu = False
    device = torch.device('cpu')
#device = torch.device('cpu') # try this when cuda is out of memory



fix_all_seeds_torch(args.gpu_id)


### Set Directory
root = args.dir
save_path = os.path.join(root, 'models'+str(args.gpu_id))
if not os.path.isdir(save_path):
    os.makedirs(save_path)


### Utility

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
    min_size=256
elif data_source.lower()=="k":
    min_size=112

# initial weight for training
if args.pretrained_model == 'coco':
    initial_weight = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1 #'COCO_V1'
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
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=initial_weight,
                # pretrained=True,
                # min_size = min_size, # IMAGE_MIN_DIM
                # max_size = 10000, # IMAGE_MAX_DIM, set to a largen number so that it has no impact
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
if args.pretrained_model != 'coco':
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
        # loss = sum(loss for loss in loss_dict.values()) # sum of losses
        loss = 1 * loss_dict['loss_classifier'] +\
               loss_dict['loss_box_reg'] +\
               loss_dict['loss_mask'] +\
               loss_dict['loss_objectness'] +\
               loss_dict['loss_rpn_box_reg'] 
        
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

    # Save the trained parameters every xx epochs
    if epoch%20 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'maskrcnn_trained_model' + d.strftime("_%Y_%m_%d_%H_%M_%S") + "_"+ str(epoch) + '.pth'))
    

