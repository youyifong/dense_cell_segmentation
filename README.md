# Training DeepCell
DeepCell_tn_nuclear_Kxx.ipynb all train with K' training images, starting from a model trained with Tissuenet 1.0 nuclear data.
DeepCell_tn_nuclear_K1.ipynb trains with images that are cut into 28 non-overlapping 512x512 patches. Best mAP 0.37 at mpp=1.
DeepCell_tn_nuclear_K2a.ipynb trains with images that are first resized by a factor of 2 and then cut into 175 overlapping 512x512 patches. Best mAP 0.48 at mpp=1.3.
DeepCell_tn_nuclear_K2b.ipynb trains with images that are first resized by a factor of 1.54 (1/0.65) and then cut into 112 overlapping 512x512 patches. Best mAP 0.44 at mpp=1.3





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
