"""
This example shows how to use Determined to implement an image
classification model for the Fashion-MNIST dataset using tf.keras.

Based on: https://www.tensorflow.org/tutorials/keras/classification.

After about 5 training epochs, accuracy should be around > 85%.
This mimics theoriginal implementation. Continue training or increase
the number of epochs to increase accuracy.
"""


import tensorflow as tf
from tensorflow import keras
from determined.keras import TFKerasTrial, TFKerasTrialContext


class FashionMNISTTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        # Initialize the trial class.
        self.context = context

    def build_model(self):
        # Define and compile model graph.

        return model

    def build_training_data_loader(self):
        # Create the training data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.
        
        return train_images, train_labels

    def build_validation_data_loader(self):
        # Create the validation data loader. This should return a keras.Sequence,
        # a tf.data.Dataset, or NumPy arrays.
       
       return test_images, test_labels