"""
Filename: federated_learning.py
Description: Manage FLOWER Federated Learning epochs.
Author: Joshua Zimmerman
Date: 2025-05-07
Version: 1.0
Python Version: 3.10.0

Changelog:
- 2024-08-02: Initial creation.
- 2024-08-11: Added a Model Manager Class, exposed variables in standalone execution for: round count, client count and model/data set.
- 2024-08-11: Refactored to adhere to OOP principals
- 2024-09-16: Offloaded Model Manger to model.py, Refactored Standalone
- 2025-05-07: Remade file: Added TensorFlow + PyTorch conversion and saving functionality. Removed all Flower functionality
- 2025-05-09: Added processing time tracking for rounds and overall training

Usage:
Run this file directly to start a Multithreading instance of Tensorflow FL with the chosen number of clients rounds and model.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from typing import List, Dict
import numpy as np
import sys
import os
import time
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_output import FLOutput
from federated_learning.fl_visualization import FLVisualization

class FederatedLearning:
    """Custom Federated Learning engine."""

    def __init__(self):
        self.num_rounds = None
        self.num_clients = None
        self.global_model = None
        self.client_data = []
        self.round_times = {}  # Dictionary to store processing times for each round
        self.total_training_time = 0  # Total training time
        self.round_accuracies = []  # List to store accuracies for each round

    def set_num_rounds(self, rounds: int) -> None:
        self.num_rounds = rounds
        print(f"Number of rounds set to: {self.num_rounds}")

    def set_num_clients(self, clients: int) -> None:
        self.num_clients = clients
        print(f"Number of clients set to: {self.num_clients}")

    def initialize_data(self):
        """Split dataset among clients."""
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        for i in range(self.num_clients):
            start = i * (len(dataset) // self.num_clients)
            end = (i + 1) * (len(dataset) // self.num_clients)
            subset = torch.utils.data.Subset(dataset, range(start, end))
            client_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)
            self.client_data.append(client_loader)
        print("Client data initialized.")

    def initialize_model(self):
        """Define a simple PyTorch model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(28 * 28, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                return self.fc(x)

        self.global_model = SimpleModel()
        print("Global model initialized.")

    def train_client(self, model, data_loader):
        """Train a single client."""
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0

        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        accuracy = correct / total if total > 0 else 0
        return model.state_dict(), accuracy

    def federated_averaging(self, client_models: List[dict]):
        """Aggregate client models into a global model."""
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client[key] for client in client_models], dim=0).mean(dim=0)
        self.global_model.load_state_dict(global_state)
        print("Global model updated via federated averaging.")

    def get_round_metrics(self) -> Dict:
        """Get metrics for the current training session."""
        return {
            "round_times": self.round_times,
            "round_accuracies": self.round_accuracies,
            "total_training_time": self.total_training_time,
            "average_round_time": self.total_training_time / self.num_rounds if self.num_rounds > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }

    def run(self):
        """Run the federated learning process with a moving parameter server."""
        self.initialize_data()
        self.initialize_model()

        total_start_time = time.time()
        round_accuracies = []

        for round_num in range(self.num_rounds):
            print(f"\nStarting round {round_num + 1}...")

            round_start_time = time.time()
            # Randomly select a client to act as the parameter server
            # In this case, we are using a round-robin approach
            server_client = round_num % self.num_clients
            print(f"Client {server_client + 1} is the parameter server for this round.")

            client_models = []
            client_accuracies = []
            round_correct = 0
            round_total = 0

            # Each client gets a copy of the global model
            for client_id, data_loader in enumerate(self.client_data):
                # Create a new model instance and load the global weights
                client_model = type(self.global_model)()
                client_model.load_state_dict(self.global_model.state_dict())
                print(f"Training client {client_id + 1}...")
                state_dict, accuracy = self.train_client(client_model, data_loader)
                client_models.append(client_model)
                client_accuracies.append(accuracy)
                round_correct += accuracy * len(data_loader.dataset)
                round_total += len(data_loader.dataset)

            # The server client becomes the new global model
            self.global_model.load_state_dict(client_models[server_client].state_dict())
            print(f"Global model updated by Client {server_client + 1}.")

            # Calculate round time and accuracy
            round_time = time.time() - round_start_time
            self.round_times[f"round_{round_num + 1}"] = round_time
            round_accuracy = round_correct / round_total if round_total > 0 else 0
            round_accuracies.append(round_accuracy)
            print(f"Round {round_num + 1} completed in {round_time:.2f} seconds with average accuracy {round_accuracy:.2%}.")

        self.total_training_time = time.time() - total_start_time
        self.round_accuracies = round_accuracies

        print(f"\nFederated learning process completed in {self.total_training_time:.2f} seconds.")
        print("\nRound-wise processing times and accuracies:")
        for round_num, round_time in self.round_times.items():
            print(f"{round_num}: {round_time:.2f} seconds, Accuracy: {round_accuracies[int(round_num.split('_')[1]) - 1]:.2%}")
        print(f"Average round time: {self.total_training_time/self.num_rounds:.2f} seconds")

if __name__ == "__main__":
    """Standalone entry point for testing FederatedLearning."""
    # Default customization values
    num_rounds = 5
    num_clients = 3
    model_type = "SimpleCNN"
    data_set = "MNIST"

    # Create timestamped run directory under results_from_output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(os.path.dirname(__file__), "results_from_output")
    run_dir = os.path.join(results_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Initialize and run the FederatedLearning instance
    fl_instance = FederatedLearning()
    fl_instance.set_num_rounds(num_rounds)
    fl_instance.set_num_clients(num_clients)
    fl_instance.run()

    # Evaluate the model
    output = FLOutput()
    output.evaluate_model(fl_instance.global_model, fl_instance.total_training_time)

    # Add timing and accuracy metrics
    timing_metrics = fl_instance.get_round_metrics()
    for metric_name, value in timing_metrics.items():
        output.add_metric(metric_name, value)

    # Add round accuracies explicitly
    output.add_metric("round_accuracies", fl_instance.round_accuracies)

    # Save results into this run folder
    log_file = os.path.join(run_dir, f"results_{timestamp}.log")
    metrics_file = os.path.join(run_dir, f"metrics_{timestamp}.json")
    model_file = os.path.join(run_dir, f"model_{timestamp}.pt")

    # Log results and save files
    output.log_result(log_file)
    output.write_to_file(metrics_file, format="json")
    output.save_model(model_file)

    print("\nResults have been saved to:")
    print("Log file:", log_file)
    print("Metrics file:", metrics_file)
    print("Model file:", model_file)

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz = FLVisualization(results_dir=run_dir)
    viz.visualize_from_json(metrics_file)
    print(f"Visualizations saved under {run_dir}")
