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
import numpy as npy
class AlgorithmInput(Input):

    # Constructor
    def __init__(self, algorithm):
        self.algorhtm = algorithm

    def read_adjacency_matrix():
        pass    

    @staticmethod 
    def read_adjacency_matrices(self, filename):
        adjacencymatrices = []
        with open(filename, 'r') as file:
            while True:
                line = file.readline().strip()
                if not line:
                    break

                # Extract the timestamp from a line that starts with "Time: "
                if line.startswith("Time: "):
                    timestamp = line.split("Time: ")[1]
                else:
                    continue

                matrix = []
                while True:
                    line = file.readline().strip()
                    if not line or line.startswith("Time: "):
                        # If we encounter another "Time: " line, step back and break
                        if line:
                            file.seek(file.tell() - len(line) - 1)
                        break
                    matrix.append(list(map(int, line.split())))
                adjacencymatrices.append((timestamp, npy.array(matrix)))
        return adjacencymatrices

    def parse_input(file):
        return super().parse_input()

    # Start algorithm