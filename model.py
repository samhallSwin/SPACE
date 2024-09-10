"""
Filename: model.py
Description: Class for ML model.
Author: Elysia Guglielmo and Connor Bett
Date: 2024-08-02
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation.
- 2024-09-10: Updated to reflect current fl_core / added model and data set selection.

Usage: 

"""
import tensorflow as tf
import numpy as np

from typing import Tuple

class Model():

    def __init__(self):
        self.model_type = ""
        self.data_set = ""

    def set_model_type(self, model_type: str):
        self.model_type = model_type
    def get_model_type(self) -> str:
        return self.model_type
    def set_data_set(self, model_type: str):
        self.data_set = model_type
    def get_data_set(self) -> str:
        return self.data_set

    # Function to load and preprocess the MNIST dataset
    def load_data(self, test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_type == "SimpleCNN":
            if self.data_set == "MNIST":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                x_train, x_test = x_train / 255.0, x_test / 255.0
                x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
                return (x_test, y_test) if test else (x_train, y_train)
            if self.data_set == "BigEarthNet":
                pass
        if self.model_type == "ResNet50":
            if self.data_set == "MNIST":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

                # Convert to 3 channels by stacking the grayscale images
                x_train = np.stack([x_train] * 3, axis=-1)
                x_test = np.stack([x_test] * 3, axis=-1)

                # Resize images to 64x64 to fit ResNet50's input size
                x_train = tf.image.resize(x_train, [32, 32])
                x_test = tf.image.resize(x_test, [32, 32])

                # Normalize the data
                x_train, x_test = x_train / 255.0, x_test / 255.0

                return (x_test, y_test) if test else (x_train, y_train)
            if self.data_set == "BigEarthNet":
                pass
    # Create model based on 
    def create_model(self) -> tf.keras.Model:
        """Create and return a model based on the selected type."""
        if self.model_type == "SimpleCNN":
            return tf.keras.Sequential([
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ])
        if self.model_type == "ResNet50":
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet', include_top=False, input_shape=(32, 32, 3))

            # Add custom layers on top of ResNet50
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            predictions = tf.keras.layers.Dense(10, activation='softmax')(x)  # 10 output classes for MNIST

            # Combine the base model with the custom layers
            self.model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

            # Freeze the base_model layers to only train the top layers
            for layer in base_model.layers:
                layer.trainable = False

            # Compile the model
            self.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

            return self.model

        raise ValueError(f"Model type '{self.model_type}' is not supported.")
