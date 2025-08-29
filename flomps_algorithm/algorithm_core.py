"""
Filename: algorithm_core.py
Description: Manage FLOMPS algorithm processes.
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal, Gagandeep Singh
Date: 2024-07-31
Version: 1.0
Python Version:

Changelog:
- 2024-07-31: Initial creation.
- 2024-08-24: getters and setters for satellite names by Yuganya Perumal
- 2024-09-09: Move Algorithm Steps from Algorithm Handler to Algorithm Core by Yuganya Perrumal
- 2024-09-21: Implemented Load Balancing based on past selection of the satellite when there more than one satellite with max no. of connectivity.
- 2024-10-03: Removed validation for satellite names. auto generation of satellite names implemented if none found.


Usage:
Instantiate to setup Algorithmhandler and AlgorithmConfig.
"""

from flomps_algorithm.algorithm_output import AlgorithmOutput
import numpy as npy
import random

class Algorithm():

    def __init__(self):
        self.satellite_names = []
        self.adjacency_matrices = None
        self.output = AlgorithmOutput()
        self.selection_counts = None
        self.output_to_file = True
        self.algorithm_output_data = None

        # Round tracking variables
        self.round_number = 1
        self.current_server = None

    def set_satellite_names(self, satellite_names):
        self.satellite_names = satellite_names
        self.selection_counts = npy.zeros(len(satellite_names))

    def get_satellite_names(self):
        return self.satellite_names

    def set_adjacency_matrices(self, adjacency_matrices):
        self.adjacency_matrices = adjacency_matrices

    def get_adjacency_matrices(self):
        return self.adjacency_matrices

    def set_output_to_file(self, output_to_file):
        self.output_to_file = output_to_file

    def get_algorithm_output(self):
        return self.algorithm_output_data

    def find_best_server_for_round(self, start_matrix_index=0, round_length=5):
        """
        Find the server that connects to the MOST satellites over the next round_length timestamps.
        More realistic for sparse satellite connectivity.
        """
        if start_matrix_index >= len(self.adjacency_matrices):
            return 0, round_length  # Default fallback

        num_satellites = len(self.satellite_names)
        best_server = 0
        best_total_connections = 0

        # Check each satellite as potential server
        for server_idx in range(num_satellites):
            total_connections = 0

            # Look ahead for the next 'round_length' timestamps
            end_idx = min(start_matrix_index + round_length, len(self.adjacency_matrices))

            for matrix_idx in range(start_matrix_index, end_idx):
                _, matrix = self.adjacency_matrices[matrix_idx]
                # Count how many other satellites this server connects to
                connections_count = sum(matrix[server_idx]) - matrix[server_idx][server_idx]  # Exclude self-connection
                total_connections += connections_count

            # Apply load balancing - prefer servers that haven't been selected much
            # Subtract a small penalty for frequently selected servers
            load_penalty = self.selection_counts[server_idx] * 0.1
            adjusted_score = total_connections - load_penalty

            if adjusted_score > best_total_connections:
                best_total_connections = adjusted_score
                best_server = server_idx

        # Update selection count for the chosen server
        self.selection_counts[best_server] += 1

        print(f"Round {self.round_number}: Selected Server {best_server} ({self.satellite_names[best_server]}) "
              f"with {best_total_connections:.1f} total connections over {round_length} timestamps")

        return best_server, round_length

    def start_algorithm_steps(self):
        """
        New FLOMPS algorithm with realistic satellite connectivity and variable round lengths.
        """
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}

        current_matrix_index = 0
        round_length = 5  # Default round length

        while current_matrix_index < len(adjacency_matrices):
            # Select best server for this round
            selected_server, actual_round_length = self.find_best_server_for_round(
                current_matrix_index, round_length
            )
            self.current_server = selected_server

            # Process this round's timestamps
            round_end = min(current_matrix_index + actual_round_length, len(adjacency_matrices))

            for timestep_in_round in range(actual_round_length):
                if current_matrix_index >= len(adjacency_matrices):
                    break

                time_stamp, matrix = adjacency_matrices[current_matrix_index]

                # All timestamps are transmitting (no training delays)
                phase = "TRANSMITTING"

                # Get server info
                selected_satellite_name = self.satellite_names[selected_server]

                # Store algorithm output
                algorithm_output[time_stamp] = {
                    'satellite_count': len(matrix),
                    'satellite_names': self.satellite_names,
                    'selected_satellite': selected_satellite_name,
                    'aggregator_id': selected_server,
                    'federatedlearning_adjacencymatrix': matrix,
                    'aggregator_flag': True,  # Always true since we always select a server
                    'round_number': self.round_number,
                    'phase': phase,
                    'target_node': selected_server,
                    'round_length': actual_round_length,
                    'timestep_in_round': timestep_in_round + 1
                }

                current_matrix_index += 1

            # Move to next round
            self.round_number += 1

            # Adaptive round length based on connectivity density
            # If satellites are well connected, use shorter rounds
            # If sparse, use longer rounds
            if current_matrix_index < len(adjacency_matrices):
                _, sample_matrix = adjacency_matrices[current_matrix_index]
                connectivity_density = npy.sum(sample_matrix) / (len(sample_matrix) ** 2)

                if connectivity_density > 0.3:  # Well connected
                    round_length = 3
                elif connectivity_density > 0.1:  # Moderately connected
                    round_length = 5
                else:  # Sparse
                    round_length = 8

        # Store the algorithm output data for external access
        self.algorithm_output_data = algorithm_output

        if self.output_to_file:
            self.output.write_to_file(algorithm_output)

        self.output.set_result(algorithm_output)
