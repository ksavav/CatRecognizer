import tensorflow as tf
from keras import models, layers


class AlexNet:
    def __init__(self, train_shape):
        self.train_shape = train_shape
        self.model = models.Sequential()

    def create_model(self):
        self.model = models.Sequential([
            layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear",
                                                       input_shape=self.train_shape[1:]),
            layers.Conv2D(96, 11, strides=4, padding='same'),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Activation('relu'),
            layers.MaxPooling2D(3, strides=2),
            layers.Conv2D(256, 5, strides=4, padding='same'),
            layers.Lambda(tf.nn.local_response_normalization),
            layers.Activation('relu'),
            layers.MaxPooling2D(3, strides=2),
            layers.Conv2D(384, 3, strides=4, padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(384, 3, strides=4, padding='same'),
            layers.Activation('relu'),
            layers.Conv2D(256, 3, strides=4, padding='same'),
            layers.Activation('relu'),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

    def get_model(self):
        return self.model
