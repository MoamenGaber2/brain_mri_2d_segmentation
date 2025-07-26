import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import os
import numpy as np
from src.model import UNetModel
from src.utils.data_preparing import shuffling, create_dataset
from src.utils.data_loader import load_processed_folder, create_dir
from src.losses import dice_loss, dice_coef, iou_coef

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("brain_mri_2d_segmentation/outputs/files")

    batch_size = 32
    lr = 1e-4
    num_epochs = 100
    model_path = os.path.join("brain_mri_2d_segmentation/outputs/files", "model.h5")
    csv_path = os.path.join("brain_mri_2d_segmentation/outputs/files", "data.csv")

    dataset_path = "brain_mri_2d_segmentation/data/preprocessed"
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")

    train_x, train_y = load_processed_folder(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_processed_folder(valid_path)

    train_dataset = create_dataset(train_x, train_y, batch_size = batch_size)
    valid_dataset = create_dataset(valid_x, valid_y, batch_size = batch_size)

    model = UNetModel()
    model.compile(loss = dice_loss, optimizer = optimizers.Nadam(lr), metrics = [dice_coef, iou_coef])

    callbacks = [
      EarlyStopping(monitor = "val_loss", patience = 8, verbose = 1, restore_best_weights = False),
      ModelCheckpoint(model_path, verbose = 1, save_best_only = True),
      ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 4, min_lr = 1e-7, verbose = 1),
      CSVLogger(csv_path),
      TensorBoard()
    ]

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        shuffle=False
    )