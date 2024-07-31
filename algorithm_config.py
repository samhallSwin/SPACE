"""
Filename: algorithm_config.py
Description: Converts Algorithm JSON options to local data types and configures core Algorithm module via setters. 
Author:
Date: 2024-07-11
Version: 1.0
Python Version: 

Changelog:
- 2024-07-11: Initial creation.

Usage: 

"""
import json
from dataclasses import dataclass

from interfaces.config import Config

@dataclass
class AlgorithmOptions:
    key: str
    node_processing_time: int
    search_depth: int

class AlgorithmConfig(Config):

    # Constructor, accepts core module
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.options = None

    # Traverse JSON options to check for nested objects?
    def read_options(self, options):
        self.options = options
        self.set_algorithm()

    def read_options_from_file(file):
        return super().read_options_from_file()

    def set_algorithm(self):
        self.algorithm.set_node_processing_time(self.options["node_processing_time"])
        self.algorithm.set_search_depth(self.options["search_depth"])
        
