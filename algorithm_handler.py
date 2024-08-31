"""
Filename: algorithm_handler.py
Description: Reads satellite link data to perform preprocessing before satellite communication algorithm execution. 
Author:
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.

Usage: 
Instantiate AlgorithmHandler and assign Algorithm. 
Then run start_algorithm() with adjacency matrices. 
"""
from interfaces.handler import Handler
import numpy as npy
class AlgorithmHandler(Handler):

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
     # Start algorithm
    def select_satellite_with_max_connections(self, eachmatrix):
        satellite_connections = npy.sum(eachmatrix, axis=1)
        maxconnections = npy.max(satellite_connections)
        satelliteindex = npy.argmax(satellite_connections)
        return satelliteindex, maxconnections
    
    def get_selected_satellite_name(self, satelliteindex):
        satellite_names = self.algorithm.get_satellite_names()
        if 0 <= satelliteindex < len(satellite_names):
            return satellite_names[satelliteindex]
        else:
            raise IndexError("Satellite does not exist for selection.")
        
    
    def parse_input(self, adjacencymatrices):
        algorithmoutput = {}

        # Initialize aggregator flags for all nodes to False
        aggregator_flags = {node_name: False for node_name in self.algorithm.get_satellite_names()}

        for timestamp, eachmatrix in adjacencymatrices:
            nrows, mcolumns = eachmatrix.shape
            # get matrix size
            if(nrows == mcolumns):
                satellite_count = nrows
            # Check if there are no connections (all zeros)
            if npy.all(eachmatrix == 0):
                # If no connections, output the matrix as it is
                algorithmoutput[timestamp] = {
                    'satellite_count': satellite_count,
                    'selected_satellite': None,
                    'federatedlearning_adjacencymatrix': eachmatrix.copy(),
                    'aggregator_flag': None
                }
                continue  # Skip to the next iteration

            satelliteconnections = npy.sum(eachmatrix, axis=1)
            maxconnections = npy.max(satelliteconnections)

            # Find all nodes with the maximum number of connections
            max_connected_satellite = [i for i, conn in enumerate(satelliteconnections) if conn == maxconnections]

            # Choose the first node in case of a tie
            selected_satellite_index = max_connected_satellite[0]
            selected_satellite = self.get_selected_satellite_name(selected_satellite_index)

            # Set the aggregator flag for the selected node
            aggregator_flags[selected_satellite] = True

            # Create a copy of the matrix to modify
            fl_am = eachmatrix.copy()

            # Retain connections (1s) for the selected node and its connections
            for i, node_name in enumerate(self.algorithm.get_satellite_names()):
                if i != selected_satellite_index:
                    # Remove connections to/from non-selected nodes that are not connected to the selected node
                    if eachmatrix[selected_satellite_index, i] == 0 and eachmatrix[i, selected_satellite_index] == 0:
                        fl_am[i, :] = 0  # Remove all connections from this node
                        fl_am[:, i] = 0  # Remove all connections to this node

            # Store in the prepared algorithm output as a dictionary data structure.
            algorithmoutput[timestamp] = {
                'satellite_count': satellite_count,
                'selected_satellite': selected_satellite,
                'federatedlearning_adjacencymatrix': fl_am,
                'aggregator_flag': aggregator_flags[selected_satellite]
            }

        return algorithmoutput




   