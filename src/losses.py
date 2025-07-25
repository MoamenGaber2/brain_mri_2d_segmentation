import tensorflow as tf

def dice_loss(y_true, y_pred, eps = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    dice = (numerator + eps) / (denominator + eps)
    
    return 1 - dice

def dice_coef(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    
    return (numerator + epsilon) / (denominator + epsilon)

def iou_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis = [1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis = [1,2,3]) - intersection
    iou = (intersection + eps) / (union + eps)
    
    return tf.reduce_mean(iou)

def iou_coef_test(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    iou = (intersection + eps) / (union + eps)
    
    return tf.reduce_mean(iou)