"""
Filename: fl_config.py
Description: Converts Federated Learning JSON options to local data types and configures core Federated Learning module via setters. 
Author: Elysia Guglielmo
Date: 2024-08-02
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation.

Usage: 

"""
import json
from dataclasses import dataclass
from fl_core import FederatedLearning
from model import *

from interfaces.config import Config

@dataclass
class AlgorithmOptions:
    key: str
    node_processing_time: int
    search_depth: int

class FLConfig(Config):

    # Constructor, accepts core module
    def __init__(self, federated_learning: FederatedLearning):
        self.fl_core = federated_learning
        self.model = Model()
        self.options = None
        self.model_options = None

    # Traverse JSON options to check for nested objects?
    def read_options(self, options):
        self.options = options
        self.set_federated_learning()
        self.set_federated_learning_model()

    def read_options_from_file(file):
        return super().read_options_from_file()

    def set_federated_learning(self) -> None:
        self.fl_core.set_num_rounds(self.options["num_rounds"])
        self.fl_core.set_num_clients(self.options["num_clients"])
        print ("setters called")

    def set_federated_learning_model(self) -> None:
                self.model.set_model_type(self.options["model_type"])
                self.model.set_data_set(self.options["data_set"])
