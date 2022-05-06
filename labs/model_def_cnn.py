"""
This example shows how to use Determined to implement an image
classification model for the Fashion-MNIST dataset using tf.keras.

Based on: https://www.tensorflow.org/tutorials/keras/classification.

After about 5 training epochs, accuracy should be around > 85%.
This mimics theoriginal implementation. Continue training or increase
the number of epochs to increase accuracy.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras

from determined.keras import TFKerasTrial, TFKerasTrialContext, InputData

import data


class FashionMNISTTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(self.context.get_hparam("filters1"), (3,3), input_shape=(28,28,1), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(self.context.get_hparam("filters2"), (3,3), padding="same", activation="relu"),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Dropout(self.context.get_hparam("dropout1")),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.context.get_hparam("dense1"), activation="relu"),
                tf.keras.layers.Dropout(self.context.get_hparam("dropout2")),
                tf.keras.layers.Dense(10),
            ]
        )

        # Wrap the model.
        model = self.context.wrap_model(model)

        # Create and wrap the optimizer.
        optimizer = tf.keras.optimizers.Adam()
        optimizer = self.context.wrap_optimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return model

    def build_training_data_loader(self) -> InputData:
        train_images, train_labels = data.load_training_data()
        train_images = train_images / 255.0
        train_images = np.expand_dims(train_images, axis=-1)
        
        return train_images, train_labels
        
    def build_validation_data_loader(self) -> InputData:
        test_images, test_labels = data.load_validation_data()
        test_images = test_images / 255.0
        test_images = np.expand_dims(test_images, axis=-1)
 
        return test_images, test_labels
