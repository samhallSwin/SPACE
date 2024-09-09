"""
Filename: algorithm_core.py
Description: Manage FLOMPS algorithm processes. 
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal
Date: 2024-07-31
Version: 1.0
Python Version: 

Changelog:
- 2024-07-31: Initial creation.
- 2024-08-24: getters and setters for satellite names by Yuganya Perumal
- 2024-09-09: Move Algorithm Steps from Algorithm Handler to Algorithm Core by Yuganya Perrumal

Usage: 
Instantiate to setup Algorithmhandler and AlgorithmConfig.
"""
from algorithm_output import AlgorithmOutput
import numpy as npy
class Algorithm():

    # Constructor
    def __init__(self):
        self.satellite_names = []
        self.adjacency_matrices = None
        self.output = AlgorithmOutput()
    def set_satellite_names(self, satellite_names):
        self.satellite_names = satellite_names

    def get_satellite_names(self):
        return self.satellite_names
    
    def set_adjacency_matrices(self, adjacency_matrices):
        
        self.adjacency_matrices = adjacency_matrices
    
    def get_adjacency_matrices(self):
        return self.adjacency_matrices

    # chooses the satellite with maximum number of connections
    def select_satellite_with_max_connections(self, each_matrix):
        satellite_connections = npy.sum(each_matrix, axis=1)
        max_connections = npy.max(satellite_connections)
        satellite_index = npy.argmax(satellite_connections)
        return satellite_index, max_connections
    
    def get_selected_satellite_name(self, satellite_index):
        satellite_names = self.get_satellite_names()
        if 0 <= satellite_index < len(satellite_names):
            return satellite_names[satellite_index]
        else:
            raise IndexError("Satellite does not exist for selection.")
    
    def start_algorithm_steps(self):
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}
        # Start Algorithm Component Steps
        # Step 1: Initialize aggregator flags for all satellite to False
        aggregator_flags = {sat_name: False for sat_name in self.get_satellite_names()}

        # Step 2: Loop through each incoming Adjacency Matrix associated with timestamp. 
        #         If No connectivitiy found at a particular timestamp set the matrix it to algorithm output 
        for time_stamp, each_matrix in adjacency_matrices:
            nrows, mcolumns = each_matrix.shape
            # get matrix size to know number of satellites involved.
            if(nrows == mcolumns):
                satellite_count = nrows
            # Check if there are no connections (all zeros)
            if npy.all(each_matrix == 0):
                # If no connectivity between them, set algorithm output (Federated Learning Adjacency Matrix (FLAM)) without changes.
                algorithm_output[time_stamp] = {
                    'satellite_count': satellite_count,
                    'selected_satellite': None,
                    'federatedlearning_adjacencymatrix': each_matrix.copy(),
                    'aggregator_flag': None
                }
                continue  # Skip to the next iteration
        # Step 3: Incomming Adjacency Matrix has connections.
        # Step 4: Perform sum of rows
            satellite_connections = npy.sum(each_matrix, axis=1)
        # Step 5: Find the maximum number of connections in that particular timestamp.
            max_connections = npy.max(satellite_connections)

        # Step 6: Find all satellite with the maximum number of connections in that particular timestamp.
            max_connected_satellite = [i for i, conn in enumerate(satellite_connections) if conn == max_connections]

        # Step 7: Choose the first satellite in case there are more than one satellites with max number of connectivity.
            selected_satellite_index = max_connected_satellite[0]
            selected_satellite = self.get_selected_satellite_name(selected_satellite_index)

        # Step 8: Set the aggregator flag to true for the selected satellite for that particular timestamp.
            aggregator_flags[selected_satellite] = True

        # Step 9: Create a copy of the Adjacency Matrix to trim down the connectivity for non-selected satellites.
            fl_am = each_matrix.copy()

        # Step 10: Retain connections (1s) for the selected node
            for i, satellite_name in enumerate(self.get_satellite_names()):
                if i != selected_satellite_index:
                    # Remove connections to/from non-selected nodes that are not connected to the selected node
                    if each_matrix[selected_satellite_index, i] == 0 and each_matrix[i, selected_satellite_index] == 0:
                        fl_am[i, :] = 0  # Remove all connections from this node
                        fl_am[:, i] = 0  # Remove all connections to this node

        # Step 11: Store algorithm output (Federated Learning Adjacency Matrix (FLAM)) as a dictionary data structure.
            algorithm_output[time_stamp] = {
                'satellite_count': satellite_count,
                'selected_satellite': selected_satellite,
                'federatedlearning_adjacencymatrix': fl_am,
                'aggregator_flag': aggregator_flags[selected_satellite]
            }
        self.output.write_to_file(algorithm_output) # write to file.
        self.output.set_result(algorithm_output) # set result to AlgorithmOutput
    