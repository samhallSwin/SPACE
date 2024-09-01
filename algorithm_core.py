"""
Filename: algorithm_input.py
Description: Manage FLOMPS algorithm processes. 
Author: Elysia Guglielmo
Date: 2024-07-31
Version: 1.0
Python Version: 

Changelog:
- 2024-07-31: Initial creation.

Usage: 
Instantiate to setup Algorithmhandler and AlgorithmConfig.
"""
from algorithm_output import AlgorithmOutput
import numpy as npy
class Algorithm():

    # Constructor
    def __init__(self):
        '''
        self.node_processing_time = 0
        self.search_depth = 0
        self.no_of_nodes = 0
        '''
        self.satellite_names = []
        self.output = AlgorithmOutput()
    '''
    def set_node_processing_time(self, time):
        self.node_processing_time = time

    def set_search_depth(self, depth):
        self.search_depth = depth
    
    def set_no_of_nodes(self, noofnodes):
        self.no_of_nodes = noofnodes

    def get_no_of_nodes(self):
        return self.no_of_nodes
    '''
    def set_satellite_names(self, satellitenames):
        self.satellite_names = satellitenames

    def get_satellite_names(self):
        return self.satellite_names
    