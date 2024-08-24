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
Instantiate to setup AlgorithmInput and AlgorithmConfig.
"""
from algorithm_output import AlgorithmOutput

class Algorithm():

    # Constructor
    def __init__(self):
        '''
        self.node_processing_time = 0
        self.search_depth = 0
        '''
        self.no_of_nodes = 0
        self.node_names = []
        self.output = AlgorithmOutput()
    '''
    def set_node_processing_time(self, time):
        self.node_processing_time = time

    def set_search_depth(self, depth):
        self.search_depth = depth
    '''

    def set_no_of_nodes(self, noofnodes):
        self.no_of_nodes = noofnodes

    def set_node_names(self, nodenames):
        self.node_names = nodenames

    def get_no_of_nodes(self):
        return self.no_of_nodes
    
    def get_node_names(self):
        return self.node_names