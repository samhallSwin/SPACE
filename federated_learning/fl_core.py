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
import threading
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
        self.round_times = {}
        self.total_training_time = 0
        self.round_accuracies = []
        self.timestep = 0  # Global timestep counter

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

    def load_flam_schedule(self, flam_path):
        """Parse FLAM CSV and return a list of dicts with timestep, phase, etc."""
        schedule = []
        with open(flam_path, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Timestep:"):
                # Parse header
                parts = line.split(", ")
                timestep = int(parts[0].split(":")[1])
                phase = parts[3].split(":")[1].strip()
                schedule.append({"timestep": timestep, "phase": phase})
                # Skip the adjacency matrix (5 lines)
                i += 6
            else:
                i += 1
        return schedule

    def run(self, flam_path=None):
        """Run the federated learning process, synchronized with FLAM phases."""
        self.initialize_data()
        self.initialize_model()
        total_start_time = time.time()
        round_accuracies = []

        # Load FLAM schedule if provided
        flam_schedule = []
        if flam_path:
            flam_schedule = self.load_flam_schedule(flam_path)
            print(f"Loaded FLAM schedule with {len(flam_schedule)} timesteps.")

        # Start the timestep counter in a background thread
        stop_event = threading.Event()
        def timestep_counter():
            while not stop_event.is_set():
                time.sleep(12) # Increment every 12 seconds
                self.timestep += 1
                print(f"Timestep: {self.timestep}")

        counter_thread = threading.Thread(target=timestep_counter)
        counter_thread.daemon = True
        counter_thread.start()

        try:
            flam_idx = 0
            round_num = 0
            while flam_idx < len(flam_schedule) and round_num < self.num_rounds:
                flam_entry = flam_schedule[flam_idx]
                # Wait until the global timestep matches the FLAM timestep exactly
                while self.timestep < flam_entry["timestep"]:
                    time.sleep(0.5)
                if self.timestep == flam_entry["timestep"]:
                    phase = flam_entry["phase"]
                    print(f"\nFLAM Timestep {flam_entry['timestep']} Phase: {phase}")

                    if phase == "TRAINING":
                        # Only train if we haven't already trained for this round
                        if not hasattr(self, "_pending_client_models"):
                            print(f"Training round {round_num + 1}...")
                            client_models = []
                            client_accuracies = []
                            round_correct = 0
                            round_total = 0
                            for client_id, data_loader in enumerate(self.client_data):
                                client_model = type(self.global_model)()
                                client_model.load_state_dict(self.global_model.state_dict())
                                print(f"Training client {client_id + 1}...")
                                state_dict, accuracy = self.train_client(client_model, data_loader)
                                client_models.append(state_dict)
                                client_accuracies.append(accuracy)
                                round_correct += accuracy * len(data_loader.dataset)
                                round_total += len(data_loader.dataset)
                            self._pending_client_models = client_models
                            self._pending_round_correct = round_correct
                            self._pending_round_total = round_total
                            self._pending_round_start_time = time.time()
                        else:
                            print(f"Already trained for round {round_num + 1}, waiting for TRANSMITTING phase...")
                    elif phase == "TRANSMITTING":
                        if hasattr(self, "_pending_client_models"):
                            self.federated_averaging(self._pending_client_models)
                            print("Global model updated via federated averaging.")
                            round_time = time.time() - self._pending_round_start_time
                            round_accuracy = self._pending_round_correct / self._pending_round_total if self._pending_round_total > 0 else 0
                            self.round_times[f"round_{round_num + 1}"] = round_time
                            round_accuracies.append(round_accuracy)
                            print(f"Round {round_num + 1} completed in {round_time:.2f} seconds with average accuracy {round_accuracy:.2%}.")
                            round_num += 1
                            # Clean up
                            del self._pending_client_models
                            del self._pending_round_correct
                            del self._pending_round_total
                            del self._pending_round_start_time
                flam_idx += 1

            self.total_training_time = time.time() - total_start_time
            self.round_accuracies = round_accuracies

            print(f"\nFederated learning process completed in {self.total_training_time:.2f} seconds.")
            print("\nRound-wise processing times and accuracies:")
            for idx, round_time in self.round_times.items():
                print(f"{idx}: {round_time:.2f} seconds, Accuracy: {round_accuracies[int(idx.split('_')[1]) - 1]:.2%}")
            print(f"Average round time: {self.total_training_time/self.num_rounds:.2f} seconds")
        finally:
            stop_event.set()
            counter_thread.join(timeout=2)
            print("Counter thread stopped.")
            print("Total timestep count:", self.timestep)

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
    flam_path = os.path.join(os.path.dirname(__file__), "../synth_FLAMs/flam_4n_100t_flomps_2025-06-03_15-15-29.csv")
    fl_instance.run(flam_path=flam_path)

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
