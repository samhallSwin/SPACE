"""
Filename: federated_learning.py
Description: Manage FLOWER Federated Learning epochs.
Author: Elysia Guglielmo and Connor Bett
Date: 2024-08-02
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation.
- 2024-08-11: Added a Model Manager Class, exposed variables in standalone execution for: round count, client count and model/data set.
- 2024-08-11: Refactored to adhere to OOP principals
- 2024-09-16: Offloaded Model Manger to model.py, Refactored Standalone

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

class FederatedLearning:
    """Manages the Flower FL server and clients."""

    def __init__(self):
        self.num_rounds = None
        self.num_clients = None 
        self.output = FLOutput()
        #self.model_manager = ModelManager("mnist")
        self.model_manager = None


    """Parse and Set Values from Handler"""
    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
        print(f"round count set to: {self.num_rounds}")
    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
        print(f"client count set to: {self.num_clients}")

    def set_model_hyperparameters(self, params) -> None:
        pass
        # model_manager.set_model_hyperparameters(params)
    def set_model(self, model: Model) -> None:
        self.model_manager = model

    def start_server(self):
        """Start the Flower server."""
        model = self.model_manager.create_model()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        print(f"CLIENT count set to{self.num_clients}")
        print(self.num_clients)

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

    def __init__(self, model_manager: Model):
        self.model = model_manager.create_model()
        (self.x_train, self.y_train) = model_manager.load_data()
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
    model_type = "ResNet50" # ResNet50 or SimpleCNN
    data_set = "MNIST" # MNIST or BigEarthNet(WIP)

    # Initialise and run the FederatedLearning instance
    fl_instance = FederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    fl_instance.model_manager.set_model_type(model_type)
    fl_instance.model_manager.set_data_set(data_set)
    fl_instance.run()