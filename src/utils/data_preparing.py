import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

def shuffling(x, y):
  x, y = shuffle(x, y, random_state= 42)
  return x, y

def load_and_preprocess(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_png(image, channels=1)
    mask = tf.image.decode_png(mask, channels=1)

    image = tf.image.convert_image_dtype(image, tf.float32)  # [0, 1]
    mask = tf.cast(mask, tf.float32)  # Keep as 0.0 and 1.0

    image.set_shape([256, 256, 1])
    mask.set_shape([256, 256, 1])

    return image, mask

def create_dataset(image_paths, mask_paths, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset