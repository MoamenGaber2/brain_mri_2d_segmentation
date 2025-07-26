import tensorflow as tf
from tensorflow import keras
from keras import layers

class UNetModel(tf.keras.Model):
    def __init__(self, input_shape = (256, 256, 1)):
        super(UNetModel, self).__init__()
        self.build_model(input_shape)

    def build_model(self, input_shape):
        inputs = layers.Input(input_shape)

        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(inputs)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c1)
        c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(p1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c2)
        c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(p2)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c3)
        c3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(p3)
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c4)
        c4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c5)

        # Decoder
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)          
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(u6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c6)

        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(u7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c7)

        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(u8)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c8)

        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(u9)
        c9 = layers.BatchNormalization()(c9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = "he_normal")(c9)

        c9 = layers.Dropout(0.4)(c9)
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)

if __name__ == "__main__":
    model = UNetModel()
    model.build_model()
    model.summary()