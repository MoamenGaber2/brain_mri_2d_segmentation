import os
from glob import glob
from sklearn.model_selection import train_test_split
import h5py
import cv2
import numpy as np
import tensorflow as tf

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_h5(h5_path, save_path):
    new_min = 0
    new_max = 255
    idx = 0
    for folder in os.listdir(h5_path):
        folder_dir = os.path.join(h5_path, folder)
        for file in os.listdir(folder_dir):
            file_dir = os.path.join(folder_dir, file)
            name = file.split(".")[0]
            img_dir = f"{name}_{idx}.png"
            mask_dir = f"{name}_{idx}_mask.png"
            f = h5py.File(file_dir, 'r')
            image = np.array(f['x'])
            mask = np.array(f['y'])

            binary_mask = (mask > 0).astype(np.uint8)
            image_norm = (((image - np.min(image)) * (new_max - new_min) / (np.max(image) - np.min(image))) + new_min).astype(np.uint8)
            
            cv2.imwrite(os.path.join(save_path, img_dir), image_norm)
            cv2.imwrite(os.path.join(save_path, mask_dir), binary_mask)
            idx += 1

def load_raw_split(data_path, split = 0.8):
    images = []
    masks = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png") and not filename.endswith("_mask.png"):
            images.append(os.path.join(data_path, filename))
            masks.append(os.path.join(data_path, filename.replace(".png", "_mask.png")))
    img_split_size = int(len(images) * split)
    train_data, test_x = train_test_split(images, train_size = img_split_size, random_state = 42)
    train_label, test_y = train_test_split(masks, train_size = img_split_size, random_state = 42)
    valid_split_size = int(len(train_data) * split)
    train_x, valid_x = train_test_split(train_data, train_size = valid_split_size, random_state = 42)
    train_y, valid_y = train_test_split(train_label, train_size = valid_split_size, random_state = 42)
    return (train_x, train_y), (test_x, test_y), (valid_x, valid_y)

def load_processed_folder(path):
    x = sorted(glob(os.path.join(path, "images", "*.png")))
    y = sorted(glob(os.path.join(path, "masks", "*_mask.png")))
    return x, y

def load_test_data(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)
    
    image = tf.image.decode_png(image, channels=1)
    mask = tf.image.decode_png(mask, channels=1)
    
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    image = tf.image.resize(image, [256, 256])
    mask = tf.image.resize(mask, [256, 256], method='nearest')
    
    mask = tf.cast(mask > 0, tf.float32) 
    image.set_shape([256, 256, 1])
    mask.set_shape([256, 256, 1])
    
    return image, mask