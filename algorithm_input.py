"""
Filename: algorithm_input.py
Description: Reads satellite link data to perform preprocessing before satellite communication algorithm execution. 
Author:
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.

Usage: 
Instantiate AlgorithmInput and assign Algorithm. 
Then run start_algorithm() with adjacency matrices. 
"""
from interfaces.input import Input

class AlgorithmInput(Input):

    # Constructor
    def __init__(self, algorithm):
        self.algorhtm = algorithm

    def read_adjacency_matrix():
        pass

    def read_adjacency_matrices():
        pass

    def parse_input(file):
        return super().parse_input()

    # Start algorithm