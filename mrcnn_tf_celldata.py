import os, sys, datetime, glob, pdb
import numpy as np

from mrcnn import utils
import skimage.io


class CellDataset(utils.Dataset):
    def load_image(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        fs = glob.glob(os.path.join(dataset_dir, '*_img.png'))
        image_ids = np.arange(0, len(fs), 1, int)
        #assert subset in ["train", "val"]
        val = np.zeros(len(image_ids), bool)
        val[np.arange(0,len(image_ids),8,int)] = True
        if subset == "val":
            # next line takes one every 8 images
            image_ids = image_ids[::8]#np.arange(0,81,1,int)
        else:
            # Get image ids from directory names
            if subset == "train":
                image_ids = image_ids[~val]
        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, "%03d_img.png"%image_id))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        m = skimage.io.imread(info['path'][:-7]+'masks.png')
        mask = []
        for k in range(m.max()):
            mask.append(m==(k+1))
        mask = np.stack(mask,axis=-1).astype(bool)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
