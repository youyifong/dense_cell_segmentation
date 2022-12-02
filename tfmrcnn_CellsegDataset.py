"""
refactored code from Stringer https://github.com/MouseLand/cellpose/blob/2e05d45fe1658829f164426a43dc4cee378b88f4/paper/1.0/train_maskrcnn.py

defines a class for loading image files
cellpose image files have three channels: R-nuclear, G-cyto, B-blank

we will use this class for training with tissuenet images as well
we will write tissuenet nuclear data in three channel format (train_nuclear_rgb) and
put the nuclear data in the G channel to be consistent with cellpose images
because majority of celpose image files are blank in the nuclear channel


"""

import os, sys, datetime, glob, pdb
import numpy as np

from mrcnn import utils # class inherits utils.Dataset
import skimage.io


class CellsegDataset(utils.Dataset):
    # def load_data
    # overwrite load_mask, and image_reference
        
    def load_data(self, dataset_dir, subset):
        """
        dataset_dir: directory containing the images. 
                     it is assumed that image files are named *_img.png
                     it is assumed that the training dataset_dir contains data that will be 
                       split into a training subset and a val subset in 7:21 ratio
                     dataset_dir may also the dir containing the test images
        subset:      if dataset_dir points to the training images, 
                       subset should be either train or val        
        """
        
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("cellseg", 1, "cellseg")

        fs = glob.glob(os.path.join(dataset_dir, '*_img.png'))
        
        indices = np.arange(0, len(fs), 1, int)
        
        # split traing and val
        val = np.zeros(len(indices), bool)
        val[np.arange(0,len(indices),8,int)] = True
        if subset == "val":
            # next line takes one every 8 images
            indices = indices[::8]
        elif subset == "train":
            indices = indices[~val]
        # otherwise, we assume it is testing and there will be no traingin/val split
        
        # Add images
        for i in indices:
            fn = os.path.basename(fs[i])
            self.add_image(
                "cellseg",
                image_id=os.path.splitext(fn)[0],
                path=os.path.join(dataset_dir, fn)
            )

    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # this implementation works for both cellpose and tissuenet file naming convention
        m = skimage.io.imread(info['path'].replace("img","masks"))
        mask = []
        for k in range(m.max()):
            # we don't assume that the mask indcies are contiguous
            if np.sum(m==(k+1))>0:
                mask.append(m==(k+1)) # skip 0 because 0 is assumed to be background
            # else:
            #     print(k+1)
        mask = np.stack(mask,axis=-1).astype(bool)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cellseg":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
