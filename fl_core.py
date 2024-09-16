"""
Filename: federated_learning.py
Description: Manage FLOWER Federated Learning epochs.
Author: Elysia Guglielmo
Date: 2024-08-02
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation.
- 2024-08-11: Added a Model Manager Class, exposed variables in standalone execution for: round count, client count and model/data set.
- 2024-08-11: Refactored to adhere to OOP principals

Usage: 
Run this file directly to start a Multithreading instance of Flower FL with the chosen number of clients rounds and model.

"""
import numpy as np
import tensorflow as tf
import flwr as fl

from flwr.server.driver.grpc_driver import GrpcDriver as p2p
from model import Model
from fl_output import FLOutput

from typing import Tuple, List
import multiprocessing
import time

## To Be Replaced by algorithm module
class ModelManager:
    """Handles model creation and dataset loading."""

    def __init__(self, model_type: str):
        self.model_type = model_type

    def create_model(self) -> tf.keras.Model:
        """Create and return a model based on the selected type."""
        if self.model_type == "mnist":
            return tf.keras.Sequential([
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ])
        # Future expansion for other models like ResNet
        raise ValueError(f"Model type '{self.model_type}' is not supported.")

    def load_data(self, test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the dataset."""
        if self.model_type == "mnist": 
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0
            x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
            return (x_test, y_test) if test else (x_train, y_train) 
        raise ValueError(f"Dataset for model type '{self.model_type}' is not supported.")

class FederatedLearning:
    """Manages the Flower FL server and clients."""

    def __init__(self):
        self.num_rounds = None
        self.num_clients = None 
        self.model_manager = Model()

    """Parse and Set Values from Handler"""
    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
    def set_model_hyperparameters(self, params) -> None:
        pass
        # model_manager.set_model_hyperparameters(params)
    def model_config(self,) -> None:
        pass

    def start_server(self):
        """Start the Flower server."""
        model = self.model_manager.create_model()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0, 
            min_fit_clients=self.num_clients, 
            min_available_clients=self.num_clients,
        )

        fl.server.start_server(config=fl.server.ServerConfig(num_rounds=self.num_rounds), strategy=strategy, server_address="localhost:8080")

    def start_client(self, client_id: int):
        """Start a Flower client."""
        client = FlowerClient(self.model_manager)
        fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    def run(self):
        """Run the Flower FL process with multiple clients."""
        # Start the server in a separate process
        server_process = multiprocessing.Process(target=self.start_server)
        server_process.start()

        # Sleep time for server to start, this is less important for devices with faster processors 
        time.sleep(5)

        # Start the clients in separate processes 
        client_processes = []
        for i in range(self.num_clients):
            client_process = multiprocessing.Process(target=self.start_client, args=(i,))
            client_process.start() 
            client_processes.append(client_process)

        # Wait for all client processes to complete
        for client_process in client_processes:
            client_process.join()

        # Stop the server process
        server_process.terminate()
        server_process.join() 

# Define the Flower client class
class FlowerClient(fl.client.NumPyClient):
    """Flower client implementation for federated learning."""

    def __init__(self, model_manager: ModelManager):
        self.model = model_manager.create_model()
        self.x_train, self.y_train = model_manager.load_data()  

    def get_parameters(self, config: dict) -> List[np.ndarray]:
        return self.model.get_weights() 

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self.model.set_weights(parameters) 
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {} 

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.model.set_weights(parameters)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train)
        return loss, len(self.x_train), {"accuracy": accuracy}


# Setup later...    
class ClientNode():
    def __init__(self) -> None:
        pass

class AggregatorNode():
    def __init__(self) -> None:
        pass 


if __name__ == "__main__":
    """STAND ALONE ENTRY POINT"""
    # Default customisation values
    num_rounds = 5
    num_clients = 3
    model_type = "mnist"  # Options: "mnist", "ResNet" (Future implementation)

    # Initialise and run the FederatedLearning instance
    fl_instance = FederatedLearning(num_rounds, num_clients, model_type)
    fl_instance.run()