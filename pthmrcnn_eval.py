'''
ml Python/3.9.6-GCCcore-11.2.0
ml cuDNN/8.2.2.26-CUDA-11.4.1
ml IPython/7.26.0-GCCcore-11.2.0
venv tv013
'''

import argparse, os, warnings, glob, cv2, syotil
import numpy as np
from skimage import io

import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pthmrcnn_utils import TestDataset, crop_with_overlap, remove_overlapping_pixels
from cvstitch import CVMaskStitcher

verbose = False

### Set arguments
maps=[]
# the for loop  on e makes it easier to evaluation models from multiple epochs when running in ipython
# when called from shell script, as long as there is only one element in [], it will be okay and e is not actually used
for e in [40] : # 100,80,60, 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default="/home/yfong/deeplearning/dense_cell_segmentation/images/test_images", type=str, help='folder directory containing test images')
    parser.add_argument('--the_model', required=False, default=f'/fh/fast/fong_y/Kaggle_2018_Data_Science_Bowl_Stage1/train/models2/maskrcnn_trained_model_2022_12_22_11_17_12_{e}.pth', type=str, help='pretrained model to use for prediction')
    # parser.add_argument('--the_model', required=False, default=f'/home/yfong/deeplearning/dense_cell_segmentation/images/training/models2/maskrcnn_trained_model_2022_12_23_16_53_13_{e}.pth', type=str, help='pretrained model to use for prediction')
    
    parser.add_argument('--mask_dir', default="/home/yfong/deeplearning/dense_cell_segmentation/images/test_gtmasks/", type=str, help='folder directory containing test images')
    parser.add_argument('--normalize', action='store_true', help='normalization of input image in prediction (False by default)')
    parser.add_argument('--box_detections_per_img', default=500, type=int, help='maximum number of detections per image, for all classes. Default: %(default)s')
    parser.add_argument('--min_score', default=0.2, type=float, help='minimum score threshold, confidence score or each prediction. Default: %(default)s')
    parser.add_argument('--mask_threshold', default=0.2, type=float, help='mask threshold, the predicted masks for each instance, in 0-1 range. In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5). Default: %(default)s')
    args = parser.parse_args()
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    ### this has to be done after visible device is set
    if torch.cuda.is_available() :
        gpu = True
        device = torch.device('cuda')
    else :
        gpu = False
        device = torch.device('cpu')
    
    
    ### Set Directory and test files
    root = args.dir
    imgs = sorted(glob.glob(os.path.join(root, '*_img.png'))) # test images
    filenames = []
    for item in imgs:
        tmp = os.path.splitext(item)[0]
        filenames.append(tmp.split('/')[-1])
    
    
    test_ds = TestDataset(root=root, data_source="K")
    # print(test_ds.imgs)
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
    
    
    ### Prediction
    model.eval()
    min_score = args.min_score
    mask_threshold = args.mask_threshold
    AP_arr=[]
    
    OVERLAP = 80
    THRESHOLD = 2
    # 1040x233
    AUTOSIZE_MAX_SIZE=256 # 5x1 .35
    # AUTOSIZE_MAX_SIZE=300 # 4x1, .36
    # AUTOSIZE_MAX_SIZE=500 # 3x1
    # AUTOSIZE_MAX_SIZE=1000 # 2x1
    # AUTOSIZE_MAX_SIZE=2000 # 1x1, .20
    
    for idx, sample in enumerate(test_ds): # sample = next(iter(test_ds))
        # print(f"Prediction with {idx:2d} test image")
        img = sample['image']
        image_id = sample['image_id']
        
        # # no tiling    
        # with torch.no_grad():
        #     result = model([img.to(device)])[0]
        # result_masks = result['masks'].cpu().numpy()
        # result_scores = result['scores'].cpu().numpy()        
        # # sartorios approach for removing overlap
        # previous_masks = []
        # for i, mask in enumerate(result_masks):
        #     # filter-out low-scoring results
        #     score = result_scores[i]
        #     if score < min_score:
        #         continue            
        #     # keep only highly likely pixels
        #     # mask = mask.cpu().numpy()
        #     binary_mask = mask > mask_threshold
        #     binary_mask = remove_overlapping_pixels(binary_mask, previous_masks) # if two masks are overlapped, remove the overlapped pixels?
        #     previous_masks.append(binary_mask)
        # # make a 2D array masks    
        # height_test, width_test = previous_masks[0].shape[1:]
        # masks = np.zeros((height_test, width_test), dtype='int16')
        # for val, ind_mask_map in enumerate(previous_masks):
        #     tmp = np.where(ind_mask_map[0,:,:])
        #     masks[tmp] = val+1

        # tiling, based on CVsegementer.py
        shape=img.shape
        if verbose: print(f"image shape: {shape}")
        nrows, ncols = int(np.ceil(shape[-2] / AUTOSIZE_MAX_SIZE)), int(np.ceil(shape[-1] / AUTOSIZE_MAX_SIZE))
        if verbose: print(f"nrow: {nrows}, ncol: {ncols}")
        crops = crop_with_overlap(img, OVERLAP, nrows, ncols)
        masks_ls = []
        for row in range(nrows):
            for col in range(ncols):
                crop = crops[row*ncols + col]    
                if verbose: print(f"crop shape: {crop.shape}")
                with torch.no_grad():
                    result1 = model([crop.to(device)])[0] # result1 is a dict: 'boxes', 'labels', 'scores', 'masks'
    
                result_masks = result1['masks'].cpu().numpy()
                result_scores = result1['scores'].cpu().numpy()
                if verbose: print(f"crop number of instances: {result_masks.shape[0]}")
                if result_masks.shape[0] == 0:
                    print("no masks found")
                    exit
                
                # sartorios approach for removing overlap
                previous_masks = []
                for i, mask in enumerate(result_masks):
                    # filter-out low-scoring results
                    score = result_scores[i]
                    if score < min_score:
                        continue            
                    # keep only highly likely pixels
                    binary_mask = mask > mask_threshold
                    binary_mask = remove_overlapping_pixels(binary_mask, previous_masks) # if two masks are overlapped, remove the overlapped pixels?
                    previous_masks.append(binary_mask)
                # make a 2D array masks    
                height_test, width_test = previous_masks[0].shape[1:]
                maskarr = np.zeros((height_test, width_test), dtype='int16')
                for val, ind_mask_map in enumerate(previous_masks):
                    tmp = np.where(ind_mask_map[0,:,:])
                    maskarr[tmp] = val+1

                masks_ls.append(maskarr)
        
        if len(masks_ls) > 1:
            stitcher = CVMaskStitcher(overlap=OVERLAP)
            masks = stitcher.stitch_masks(masks_ls, nrows, ncols)
            if masks.shape != shape[1:]:
                print("stitched mask has a different shape from the original")
                exit
        else:
            masks = maskarr
        
        
        # # Stringer approach for removing overlap
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
                
    
        # save masks
        if masks.max() < 2**16:
            masks = masks.astype(np.uint16) 
            cv2.imwrite(os.path.join(root, filenames[idx].replace("_img", "_mrmasks") + '.png'), masks)
        else:
            warnings.warn('found more than 65535 masks in each image, cannot save PNG, saving as TIF')
        
        print(os.path.basename(test_ds.imgs[idx]))
        truth=io.imread(args.mask_dir + os.path.basename(test_ds.imgs[idx]).replace("_img","_masks"))
        print(f"tpfpfn: {syotil.tpfpfn(truth, masks)}, AP: {syotil.csi(truth, masks):.2f}")
        AP_arr.append(syotil.csi(truth, masks))
    
    maps.append(np.mean(AP_arr))
    
    # write AP to a file
    with open('csi.txt', 'a') as file:
        file.write(','.join(["{0:0.6f}".format(i) for i in AP_arr])+"\n")
    
    
print ('mAPs: '+' '.join(["{0:0.2f}".format(i) for i in maps]))