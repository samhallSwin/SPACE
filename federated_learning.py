"""
Filename: federated_learning.py
Description: Manage FLOWER Federated Learning epochs.
Author: Elysia Guglielmo
Date: 2024-08-02
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation.

Usage: 

"""
import flwr as fl

from model import Model

class FederatedLearning():

    # Have MNIST as default model ?
    def __init__(self, model: Model = Model()):
        self.model = model

    def start_client(self):
        # Load data
        train_data, test_data = self.model.load_data()

        # Build model
        model = self.model.build_model()

        # Create and start Flower client
        client = MnistClient(model, train_data, test_data)
        fl.client.start_client(
            server_address="localhost:8080",
            client=client.to_client()
        )
        
    def start_server(self):
        # Define the strategy for federated learning
        strategy = fl.server.strategy.FedAvg()

        # Start the Flower server on the specified local address and port
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=3),  # Number of federated learning rounds
            strategy=strategy  # assign stratergy
        )
    
    def start_simulation(self):
        history = fl.simulation.start_simulation(

        )

# Define the Flower client class
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}
    

# Setup later...    
class ClientNode():
    def __init__(self) -> None:
        pass

class AggregatorNode():
    def __init__(self) -> None:
        pass