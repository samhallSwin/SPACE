"""
Filename: algorithm_config.py
Description: Converts Algorithm JSON options to local data types and configures core Algorithm module via setters. 
Author:
Date: 2024-07-11
Version: 1.0
Python Version: 

Changelog:
- 2024-07-11: Initial creation.
- 2024-08-24: Reading JSON Options no_of_nodes and node_names to configure the setters by Yuganya Perumal

Usage: 

"""
import json
from dataclasses import dataclass

from interfaces.config import Config

@dataclass
class AlgorithmOptions:
    key: str
    '''
    node_processing_time: int
    search_depth: int
    '''
    no_of_nodes : int
    node_names = []

class AlgorithmConfig(Config):

    # Constructor, accepts core module
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.options = None

    # Traverse JSON options to check for nested objects
    def read_options(self, options):
        self.options = options
        self.set_algorithm()

    def read_options_from_file(file):
         return super().read_options_from_file()


    def set_algorithm(self):
        # self.algorithm.set_node_processing_time(self.options["node_processing_time"])
        # self.algorithm.set_search_depth(self.options["search_depth"])
        self.algorithm.set_no_of_nodes(self.options["no_of_nodes"])
        self.algorithm.set_node_names(self.options["node_names"])

    def get_algorithm_no_of_nodes(self):
        return self.algorithm.get_no_of_nodes() 

    def get_algorithm_node_names(self):
        return self.algorithm.get_node_names() 
