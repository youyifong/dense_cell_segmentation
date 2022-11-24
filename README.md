# Summary

Cellpose training is implemented in shell scripts. DeepCell training is implemented in jupyter notebooks. The AP results are saved in csv files under the APresults folder. Tabular and graphical summary of the results are made with summary.R



# Cellpose training 

All training is done for 500 epochs. Cellpose training is repeated three times with seeds 1-3. 

- Results in Section 2 are obtained with cellpose_pred_model_zoo.sh. 
- Results in Section 3 are obtained with cellpose_train_pred_loop.sh, which calls cellpose_train_pred.sh. 
- Results in Section 4 are obtained with cellpose_train_pred_seeds.sh, which calls cellpose_train_pred.sh. 



# DeepCell training 

DeepCell_tn_nuclear_Kxx.ipynb all train with K' training images, starting from a model trained with Tissuenet 1.0 nuclear data. All training is done for 200 epochs. 

- DeepCell_tn_nuclear_K1a.ipynb trains with images that are cut into 7x4 non-overlapping 512x512 patches. 
    Best mAP 0.37 at mpp=1.
- DeepCell_tn_nuclear_K1b.ipynb trains with images that are cut into 7x25 non-overlapping 512x512 patches. 
    Best mAP 0.37 at mpp=1.
- * DeepCell_tn_nuclear_K2a.ipynb trains with images that are first resized by a factor of 2 and then cut into 175 overlapping 512x512 patches. 
    Best mAP 0.48 at mpp=1.3.
- DeepCell_tn_nuclear_K2b.ipynb trains with images that are resized by 1.54 (1/0.65) and then cut into 112 overlapping 512x512 patches. 
    Best mAP 0.44 at mpp=1.3.
- DeepCell_tn_nuclear_K2c.ipynb trains with images that are resized by 1.54 (1/0.65) and then cut into 112 overlapping 512x512 patches with edge cases removed. 
    Best mAP 0.48 at mpp=1.3.
- DeepCell_tn_nuclear_K3.ipynb trains with CroppingDataGenerator using images resized by 2. Each epoch only trains with 7 images. 
    Best mAP 0.42 at mpp=1.3.

It makes sense that the best mAP for models trained with enlarged images are obtained at a higher mpp than for models with trained with un-enlarged images.

When making predictions with the pretrained nuclear model, mpp 1.2 has the best performance. But I thought it is hard to resize an image by a fraction. So I chose mpp 1.3, which translates to an enlargement factor of 1.3/0.65=2. That is the notebook K2a. As a sensitivity analysis, we also try mpp 1, which leads to the second resize factor and that is the notebook 2b. 


DeepCell_tn_nuclear_K2a.ipynb is the best. 

The training data was not normalized.
