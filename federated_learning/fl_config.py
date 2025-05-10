"""
Filename: fl_config.py
Description: Converts Federated Learning JSON options to local data types, passes model and algorithm (FLAM) config to the core engine, and sets them using dedicated setters.
Author: Kimsakona SOK
Date: 10/05/2025
Version: 1.0
Python Version: 3.10.0

Changelog:
- 2024-08-02: Initial creation.
- 2024-09-16: Supported standalone execution, refactored Model Setters
- 2025-05-10: Added FLAM algorithm data passing to core and print confirmation.

Usage:
- Create a FLConfig instance with a FederatedLearning object and optional algorithm (FLAM)
- Call read_options() with a config dict to initialize model and training settings
- Algorithm data (if provided) is passed to the core using set_flam()

Example:
    fl_core = FederatedLearning()
    flam = {"key": "alg1", "node_processing_time": 5, "search_depth": 2}
    config = FLConfig(fl_core, algorithm=flam)
    config.read_options({"num_rounds": 5, "num_clients": 3, "model_type": "SimpleModel", "data_set": "MNIST"})

"""

import json
from dataclasses import dataclass
from federated_learning.fl_core import FederatedLearning
from model import *
from interfaces.config import Config

@dataclass
class AlgorithmOptions:
    key: str
    node_processing_time: int
    search_depth: int

class FLConfig(Config):

    def __init__(self, federated_learning: FederatedLearning, algorithm=None):
        # Save the FL core and optional algorithm settings (FLAM)
        self.fl_core = federated_learning
        self.model = Model()
        self.options = None
        self.model_options = None
        self.algorithm = algorithm

    def read_options(self, options):
        # Save the FL options, then pass them to the model and core
        self.options = options
        self.set_federated_learning_model()
        self.set_federated_learning()

        # If we got algorithm settings (FLAM), send them to the core system
        if self.algorithm:
            self.fl_core.set_flam(self.algorithm)
            print("FLAM algorithm data passed to core:", self.algorithm)

    def read_options_from_file(file):
        return super().read_options_from_file()

    def set_federated_learning(self) -> None:
        # Configure how many rounds and clients for FL
        self.fl_core.set_num_rounds(self.options["num_rounds"])
        self.fl_core.set_num_clients(self.options["num_clients"])
        print("FL setters called")

    def set_federated_learning_model(self) -> None:
        # Set model type and dataset, then pass the model to the core
        self.model.set_model_type(self.options["model_type"])
        self.model.set_data_set(self.options["data_set"])
        self.fl_core.set_model(self.model)
        print("MODEL setters called")

# fl_core.py (append to the existing class)

# Called by FLConfig to hand off algorithm (FLAM) settings.
# Saves the settings to self.flam_data and shows them in the console.
def set_flam(self, flam_data):
    self.flam_data = flam_data
    print("FL Core received FLAM data:", flam_data)
