'''
ml Python/3.9.6-GCCcore-11.2.0
ml cuDNN/8.2.2.26-CUDA-11.4.1
ml IPython/7.26.0-GCCcore-11.2.0
'''

### Library
import argparse, os, warnings, glob, cv2
import numpy as np
from skimage import io
import syotil

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pthmrcnn_utils import TestDataset

### Check whether gpu is available
if torch.cuda.is_available() :
    gpu = True
    device = torch.device('cuda')
else :
    gpu = False
    device = torch.device('cpu')


### Set arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--gpu_id', default=1, type=int, help='which gpu to use. Default: %(default)s')
parser.add_argument('--dir', default="/home/yfong/deeplearning/dense_cell_segmentation/images/test_images_cut", type=str, help='folder directory containing test images')
parser.add_argument('--the_model', required=False, default='/fh/fast/fong_y/Kaggle_2018_Data_Science_Bowl_Stage1/train/models0/maskrcnn_trained_model_2022_12_19_12_17_17_100.pth', type=str, help='pretrained model to use for prediction')
parser.add_argument('--normalize', action='store_true', help='normalization of input image in prediction (False by default)')
parser.add_argument('--box_detections_per_img', default=500, type=int, help='maximum number of detections per image, for all classes. Default: %(default)s')
parser.add_argument('--min_score', default=0.5, type=float, help='minimum score threshold, confidence score or each prediction. Default: %(default)s')
parser.add_argument('--mask_threshold', default=0.5, type=float, help='mask threshold, the predicted masks for each instance, in 0-1 range. In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5). Default: %(default)s')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"]="1"



### Set Directory and test files
root = args.dir
imgs = sorted(glob.glob(os.path.join(root, '*_img.png'))) # test images
filenames = []
for item in imgs:
    tmp = os.path.splitext(item)[0]
    filenames.append(tmp.split('/')[-1])


test_ds = TestDataset(root=root, data_source="K")
#test_ds[0]

### Define Mask R-CNN Model
# normalize
if args.normalize:
    resnet_mean = (0.485, 0.456, 0.406)
    resnet_std = (0.229, 0.224, 0.225)

box_detections_per_img = args.box_detections_per_img # default is 100, but 539 is used in a reference

def get_model():
    num_classes = 2 # background or foreground (cell)
    
    if args.normalize:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                # pretrained=pretrained, # pretrained weights on COCO data
                box_detections_per_img=box_detections_per_img,
                image_mean=resnet_mean, # mean values used for input normalization
                image_std=resnet_std # std values used for input normalization
                )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                #min_size = 256, # 448, # IMAGE_MIN_DIM
                #max_size = 1024, # 448, # IMAGE_MAX_DIM
                #box_score_thresh=0, # DETECTION_MIN_CONFIDENCE
                #rpn_pre_nms_top_n_train=1000, # RPN_NMS_ROIS_TRAINING
                #rpn_pre_nms_top_n_test=2000, # RPN_NMS_ROIS_INFERENCE
                #rpn_post_nms_top_n_train=1000, # RPN_NMS_ROIS_TRAINING
                #rpn_post_nms_top_n_test=2000, # RPN_NMS_ROIS_INFERENCE
                rpn_nms_thresh=0.7, # RPN_NMS_THRESHOLD (for inference)
                #rpn_batch_size_per_image=1500, # RPN_TRAIN_ANCHORS_PER_IMAGE
                #box_batch_size_per_image=300, # TRAIN_ROIS_PER_IMAGE
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


### Load pre-trained model
model.load_state_dict(torch.load(args.the_model, map_location=device))
# print(model.state_dict())


### Utilities for prediction
def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

def remove_overlaps(masks, cellpix, medians):
    """ replace overlapping mask pixels with mask id of closest mask
        masks = Nmasks x Ly x Lx
    """
    overlaps = np.array(np.nonzero(cellpix>1.5)).T # 1.5
    dists = ((overlaps[:,:,np.newaxis] - medians.T)**2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:,0], overlaps[:,1]] = 0
    masks[tocell, overlaps[:,0], overlaps[:,1]] = 1
    
    # labels should be 1 to mask.shape[0]
    masks = masks.astype(int) * np.arange(1,masks.shape[0]+1,1,int)[:,np.newaxis,np.newaxis]
    masks = masks.sum(axis=0)
    return masks


### Prediction
model.eval()
min_score = args.min_score
mask_threshold = args.mask_threshold
AP_arr=[]

for idx, sample in enumerate(test_ds):
    print(f"Prediction with {idx:2d} test image")
    img = sample['image']
    image_id = sample['image_id']
    with torch.no_grad():
        result = model([img.to(device)])[0]
    
    # # Stringer approach
    # overlap_masks = []
    # for i, mask in enumerate(result['masks']):
    #     score = result['scores'][i].cpu().item()
    #     if score < min_score:
    #         continue
    #     mask = mask.cpu().numpy()
    #     overlap_masks.append(mask)
    
    # mask_temp = np.zeros((len(overlap_masks), overlap_masks[0].shape[1], overlap_masks[0].shape[2])) # n of (1,H,W) to (n,H,W)
    # for i in range(len(overlap_masks)):
    #     mask_temp[i,:,:] = overlap_masks[i][0,:,:]
    
    # medians = []
    # for m in range(mask_temp.shape[0]): # mask_temp = [nmasks, H, W]
    #     ypix, xpix = np.nonzero(mask_temp[m,:,:])
    #     medians.append(np.array([ypix.mean(), xpix.mean()])) # median x and y coordinates
    
    # masks = remove_overlaps(mask_temp, np.transpose(mask_temp,(1,2,0)).sum(axis=-1), np.array(medians))
    
    # sartorios approach
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
        
    height_test, width_test = previous_masks[0].shape[1:]
    masks = np.zeros((height_test, width_test), dtype='int16')
    for val, ind_mask_map in enumerate(previous_masks):
        tmp = np.where(ind_mask_map[0,:,:])
        masks[tmp] = val+1
    

    # save masks
    if masks.max() < 2**16:
        masks = masks.astype(np.uint16) 
        cv2.imwrite(os.path.join(root, filenames[idx].replace("_img", "_mrmasks") + '.png'), masks)
    else:
        warnings.warn('found more than 65535 masks in each image, cannot save PNG, saving as TIF')
    

    truth=io.imread("images/test_gtmasks_cut/"+os.path.basename(test_ds.imgs[idx]).replace("_img","_masks"))
    AP_arr.append(syotil.csi(masks, truth))

print(np.mean(AP_arr))