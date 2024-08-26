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
        self.algorithm = algorithm

    '''
    def read_adjacency_matrix():
        pass    
    '''

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
                        # Step back and Break, if we see another "Time: " 
                        if line:
                            file.seek(file.tell() - len(line) - 1)
                        break
                    matrix.append(list(map(int, line.split())))
                adjacencymatrices.append((timestamp, npy.array(matrix)))
        return adjacencymatrices
    '''
    def parse_input(file):
        return super().parse_input()
    '''
    
    def select_node_with_max_connections(self, eachmatrix):
        nodeconnections = npy.sum(eachmatrix, axis=1)
        maxconnections = npy.max(nodeconnections)
        nodeindex = npy.argmax(nodeconnections)
        return nodeindex, maxconnections
    
    def get_selected_node_name(self, nodeindex):
        node_names = self.algorithm.get_node_names()
        if 0 <= nodeindex < len(node_names):
            return node_names[nodeindex]
        else:
            raise IndexError("Node index is out of range.")
        
    def parse_input(self, adjacencymatrices):
        algorithmoutput = {}
        for timestamp, eachmatrix in adjacencymatrices:
            nodeindex, _ = self.select_node_with_max_connections(eachmatrix)
            selectednode =  self.get_selected_node_name(nodeindex)

            # Set the corresponding row and column to 0s to remove connections (1s)
            eachmatrix[nodeindex, :] = 0  # Set the entire row to 0
            eachmatrix[:, nodeindex] = 0  # Set the entire column to 0

            # Store in the prepared algorithm output as dictionary data structure.
            algorithmoutput[timestamp] = {
                'selected_node': selectednode,
                'trimmed_matrix': eachmatrix.copy(),
                'aggregator_flag': True
            }
        return algorithmoutput

    # Start algorithm