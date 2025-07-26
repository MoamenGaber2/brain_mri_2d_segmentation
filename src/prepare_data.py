import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, ShiftScaleRotate
from src.utils.data_loader import load_raw_split, create_dir, extract_h5 

def augment_data(images, masks, save_path, augment = True):
    H = 256
    W = 256
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total = len(images)):
        name = x.split('.')[0].split('_')[-1]
        x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if augment == True:
            aug = HorizontalFlip(p = 1.0)
            augmented = aug(image = x, mask = y) # Will produce a dictionary with image and mask as keys
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0)
            augmented = aug(image = x, mask = y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            X = [x, x1, x2]
            Y = [y, y1, y2]

        else:
            X = [x]
            Y = [y]

        idx = 0
        for i, m in zip(X, Y): # zip is used to map every image to its corresponding label
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W,H))

            if len(X) == 1:
                tmp_image_path = f"{name}.png"
                tmp_mask_path = f"{name}_mask.png"
            else:
                tmp_image_path = f"{name}_{idx}.png"
                tmp_mask_path = f"{name}_{idx}_mask.png"

            image_path = os.path.join(save_path, "images", tmp_image_path)
            mask_path = os.path.join(save_path, "masks", tmp_mask_path)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            idx += 1

if __name__ == "__main__":
    create_dir("brain_mri_2d_segmentation/data/extracted_data")
    extract_h5("brain_mri_2d_segmentation/data/hdf5_data", "brain_mri_2d_segmentation/data/extracted_data")
    
    data_dir = "brain_mri_2d_segmentation/data/extracted_data"
    (train_x, train_y), (test_x, test_y), (valid_x, valid_y) = load_raw_split(data_dir, split = 0.8)

    create_dir("brain_mri_2d_segmentation/data/preprocessed/train/images")
    create_dir("brain_mri_2d_segmentation/data/preprocessed/train/masks")
    create_dir("brain_mri_2d_segmentation/data/preprocessed/valid/images")
    create_dir("brain_mri_2d_segmentation/data/preprocessed/valid/masks")
    create_dir("brain_mri_2d_segmentation/data/preprocessed/test/images")
    create_dir("brain_mri_2d_segmentation/data/preprocessed/test/masks")

    augment_data(train_x, train_y, "brain_mri_2d_segmentation/data/preprocessed/train/",augment = True)
    augment_data(valid_x, valid_y, "brain_mri_2d_segmentation/data/preprocessed/valid/",augment = False)
    augment_data(test_x, test_y, "brain_mri_2d_segmentation/data/preprocessed/test/",augment = False)