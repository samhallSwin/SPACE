"""
Filename: resnet50.py
Description: Class for ResNet50 model.
Author: Ahmed Mubeenul Azim & Michelle Abrigo
Date: 2024-08-24
Version: 1.0
Python Version: 

Changelog:
- 2024-08-24: Initial creation and added ResNet50 model to train on MNIST dataset.
- 2024-08-31: Added callback to track train times

Usage: 
Run this file directly to start training the ResNet50 model on MNIST.

"""

import tensorflow as tf
import numpy as np
import time

class MNISTModel:
    def __init__(self):
        self.model = None
        self.epoch_times = []

    def load_data(self):
        """Load and preprocess the MNIST dataset."""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Convert to 3 channels by stacking the grayscale images
        x_train = np.stack([x_train] * 3, axis=-1)
        x_test = np.stack([x_test] * 3, axis=-1)

        # Resize images to 64x64 to fit ResNet50's input size
        x_train = tf.image.resize(x_train, [32, 32])
        x_test = tf.image.resize(x_test, [32, 32])

        # Normalize the data
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return (x_train, y_train), (x_test, y_test)

    def build_model(self):
        """Build a ResNet50 model adapted for the MNIST dataset."""
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

    def train_model(self, x_train, y_train, epochs=5, batch_size=32, validation_data=None):
        """Train the ResNet50 model."""
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_model() before training.")

        # Define a custom callback to track train times
        class TimeHistory(tf.keras.callbacks.Callback):
            def __init__(self, epoch_times):
                self.epoch_times = epoch_times

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)

        time_callback = TimeHistory(self.epoch_times)
        start_time = time.time()
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[time_callback]
        )
        total_time = time.time() - start_time
        average_epoch_time = np.mean(self.epoch_times)

        return total_time, average_epoch_time, self.epoch_times

    def evaluate_model(self, x_test, y_test):
        """Evaluate the trained ResNet50 model on the test data."""
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_model() before evaluation.")
        return self.model.evaluate(x_test, y_test)


# Usage
mnist_model = MNISTModel()
(x_train, y_train), (x_test, y_test) = mnist_model.load_data()
mnist_model.build_model()
total_time, average_epoch_time, epoch_times = mnist_model.train_model(x_train, y_train, validation_data=(x_test, y_test))
test_loss, test_acc = mnist_model.evaluate_model(x_test, y_test)
print(f"Test accuracy: {test_acc}")
print(f"Epoch times: {epoch_times}")
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average epoch time: {average_epoch_time:.2f} seconds")
