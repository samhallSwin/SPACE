"""
Filename: MTFL.py
Description: Runs a multithreading instance of Flower FL using the Minst Data Set. 
Author: Connor Bett
Date: 2024-08-11
Version: 1.1
Python Version: 3.12.2

Changelog:
- 2024-08-04: Initial creation with 3 clients and 4 round.
- 2024-08-11: modularised code to allow for easy customsiation of the FL set up. exposed variables in main for rounds, client count and model/data set.

Usage:
Run this file directly to start a Multithreading instance of Flower FL with the chosen number of clients rounds and model.

Todo:
- Error catch for custom model input
- remove/ rework get_eval_fn function, was the attempt at providing clients with an agregation function :(
"""

import flwr as fl
import tensorflow as tf
from typing import Tuple, List
import numpy as np
import multiprocessing
import time

def create_model(dataset_model_type: str) -> tf.keras.Model:
    """Create and return a simple CNN model."""
    if dataset_model_type == "mnist":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    else:
        # Placeholder for other model types
        model = None  # to be replaced with connection and getters for ResNet :)
    return model

def load_mnist_data(test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
    if test:
        return x_test, y_test
    else:
        return x_train, y_train

def start_server(num_rounds: int, num_clients: int, dataset_model_type: str):
    """Start the Flower server."""

    def get_eval_fn(model): # Currently Redundant
        """Return an evaluation function for server-side evaluation."""
        x_test, y_test = load_mnist_data(test=True)

        def evaluate(parameters: fl.common.Parameters) -> Tuple[float, float]:
            weights = fl.common.parameters_to_ndarrays(parameters)
            model.set_weights(weights)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, accuracy

        return evaluate

    model = create_model(dataset_model_type)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy, server_address="localhost:8080")

def start_client(client_id: int, dataset_model_type: str):
    """Start a Flower client."""

    class MNISTClient(fl.client.NumPyClient):
        def __init__(self, model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray):
            self.model = model
            self.x_train = x_train
            self.y_train = y_train

        def get_parameters(self, config: dict) -> List[np.ndarray]:
            return self.model.get_weights()

        def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
            self.model.set_weights(parameters)
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
            new_parameters = self.model.get_weights()
            return new_parameters, len(self.x_train), {}

        def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
            self.model.set_weights(parameters)
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            loss, accuracy = self.model.evaluate(self.x_train, self.y_train)
            return loss, len(self.x_train), {"accuracy": accuracy}

    model = create_model(dataset_model_type)
    x_train, y_train = load_mnist_data()

    client = MNISTClient(model, x_train, y_train)

    fl.client.start_numpy_client(server_address="localhost:8080", client=client) # Local port for comms

if __name__ == "__main__":
    # Default customization values
    num_rounds = 5
    num_clients = 3
    dataset_model_type = "mnist"  # Options: "mnist", "ResNet"(Not Implemented)) 

    # Start the server in a separate process
    server_process = multiprocessing.Process(target=start_server, args=(num_rounds, num_clients, dataset_model_type))
    server_process.start()

    # Sleep time for server to start, this is less important for devices with faster processors 
    time.sleep(5)

    # Start the clients in separate processes (number of clients are based on defined count)
    client_processes = []
    for i in range(num_clients):
        client_process = multiprocessing.Process(target=start_client, args=(i, dataset_model_type))
        client_process.start()
        client_processes.append(client_process)

    # Wait for all client processes to complete
    for client_process in client_processes:
        client_process.join()

    # Stop the server process (once rounds are complete)
    server_process.terminate()
    server_process.join()
