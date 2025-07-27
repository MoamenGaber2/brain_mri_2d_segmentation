import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.data_loader import load_test_data, create_dir, load_processed_folder, shuffling
from src.losses import dice_loss, dice_coef, iou_coef, iou_coef_test
import matplotlib.pyplot as plt

def create_dataset(image_paths, mask_paths, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_test_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def evaluate_model(model, test_dataset):
    y_true = []
    y_pred = []
    
    for images, masks in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(masks.numpy().flatten())
        y_pred.extend((preds > 0.5).astype(np.float32).flatten())
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'iou': iou_coef_test(y_true, y_pred),
        'dice': dice_coef(y_true, y_pred)
    }
    return metrics, y_pred, y_true

def save_predictions(test_dataset, y_pred, output_path):
    num_samples = len(y_pred) // (256 * 256)
    y_pred_reshaped = np.array(y_pred).reshape((num_samples, 256, 256, 1))
    test_images = []
    for images, _ in test_dataset.unbatch():
        test_images.append(images.numpy())

    y_pred_squeezed = y_pred_reshaped.squeeze()
    for i in range(len(test_images)):
        img = test_images[i].squeeze()
        pred = y_pred_squeezed[i]

        red_mask = np.zeros((*pred.shape, 4))  # RGBA
        red_mask[..., 0] = 1.0                 # Red channel
        red_mask[..., 3] = (pred > 0.5).astype(float) * 0.5  # Alpha mask

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(img, cmap = 'bone')
        ax.imshow(red_mask)
        ax.axis('off')

        plt.savefig(os.path.join(output_path, f'overlay_{i:04}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    custom_objects = {'dice_loss':dice_loss,
                 'dice_coef':dice_coef,
                 'iou_coef': iou_coef}
    final_model = load_model("outputs/files/model.h5", custom_objects = custom_objects)

    # Create test dataset with proper resizing
    dataset_path = "data/preprocessed"
    test_path = os.path.join(dataset_path, "test")
    test_x, test_y = load_processed_folder(test_path)
    test_x, test_y = shuffling(test_x, test_y)
    test_dataset = create_dataset(test_x, test_y, batch_size = 32)

    # Run evaluation
    metrics, y_pred, y_true = evaluate_model(final_model, test_dataset)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    create_dir("outputs/predictions")
    output_path = 'outputs/predictions'
    save_predictions(test_dataset, y_pred, output_path)