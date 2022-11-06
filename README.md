# Training DeepCell
DeepCell_tn_nuclear_Kxx.ipynb all train with K' training images, starting from a model trained with Tissuenet 1.0 nuclear data. All training is done for 200 epochs.

DeepCell_tn_nuclear_K1a.ipynb trains with images that are cut into 7x4 non-overlapping 512x512 patches. 
    Best mAP 0.37 at mpp=1.
DeepCell_tn_nuclear_K1b.ipynb trains with images that are cut into 7x25 non-overlapping 512x512 patches. 
    Best mAP 0.37 at mpp=1.
DeepCell_tn_nuclear_K2a.ipynb trains with images that are first resized by a factor of 2 and then cut into 175 overlapping 512x512 patches. 
    Best mAP 0.48 at mpp=1.3.
DeepCell_tn_nuclear_K2b.ipynb trains with images that are resized by 1.54 (1/0.65) and then cut into 112 overlapping 512x512 patches. 
    Best mAP 0.44 at mpp=1.3.
DeepCell_tn_nuclear_K2c.ipynb trains with images that are resized by 1.54 (1/0.65) and then cut into 112 overlapping 512x512 patches with edge cases removed. 
    Best mAP 0.48 at mpp=1.3.
DeepCell_tn_nuclear_K3.ipynb trains with CroppingDataGenerator using images resized by 2. Each epoch only trains with 7 images. 
    Best mAP 0.42 at mpp=1.3.

It makes sense that the best mAP for models trained with enlarged images are obtained at a higher mpp than for models with trained with un-enlarged images.

When making predictions with the pretrained nuclear model, mpp 1-1.2 has the best performance. But I thought it is hard to resize an image by a fraction. So I chose mpp 1.3, which translates to an enlargement factor of 2. That is the notebook K2a. Last night I thought that as a sensitivity analysis to also try mpp 1, which leads to the second resize factor and that is the notebook 2b. In cellpose, this rescaling kind of happens under the hood by the ratio between estimated cell sizes. I wonder if there are similar statistics in deepcell. Beyond that ratio, integer vs fraction may also be an issue. Overall, I feel like a resize factor of 2 in our case is probably optimal.




# cell_segmentation

Baseline models: Cellpose, RetinaMask, DeepCell, Mask R-CNN

all trained on TissueNet training images (not including val and test images; val images can be used in training process) 
will be evaluate on TissueNet test images



**Installation**

1. Mesmer
- pip install deepcell
- ml Anaconda3; ml CUDA; python 
- To train Mesmer, refer to https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/Mesmer_training_notebook.ipynb

2. Cellpose
- In the Mesmer paper, Cellpose was trained as following link, https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/notebooks/training/Cellpose_training.py

3. RetinaMask
- pip install deepcell-retinamask
- In the Mesmer paper, RetinaMask was trained as following link, https://github.com/vanvalenlab/publication-figures/blob/master/2021-Greenwald_Miller_et_al-Mesmer/notebooks/training/Benchmark_training_retinamask.ipynb

4. Mask R-CNN
- In the Cellpose paper, Mask R-CNN was trained as following link, https://github.com/MouseLand/cellpose/blob/main/paper/1.0/train_maskrcnn.py
- Training settings differ between the Mesmer paper and the Cellpose paper. We should make the setting equal.
