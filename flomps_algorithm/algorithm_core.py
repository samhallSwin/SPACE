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

"""
Filename: algorithm_core.py
Description: Manage FLOMPS algorithm processes with realistic satellite connectivity.
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
        New FLOMPS algorithm: Wait until selected server connects to ALL other satellites.
        """
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}

        current_matrix_index = 0

        while current_matrix_index < len(adjacency_matrices):
            # Select best server for this round
            selected_server, estimated_timestamps = self.find_best_server_for_round(current_matrix_index)
            self.current_server = selected_server

            # Run this round until server connects to ALL satellites
            round_start_index = current_matrix_index
            timestep_in_round = 0
            server_connected_to_all = False
            num_satellites = len(self.satellite_names)
            target_connections = num_satellites - 1  # All except self

            while current_matrix_index < len(adjacency_matrices) and not server_connected_to_all:
                timestep_in_round += 1
                time_stamp, matrix = adjacency_matrices[current_matrix_index]

                # Check if server now connects to all other satellites
                server_connections = sum(matrix[selected_server]) - matrix[selected_server][selected_server]
                if server_connections == target_connections:
                    server_connected_to_all = True
                    print(f"  ✓ Server {selected_server} connected to all {target_connections} satellites at timestep {timestep_in_round}")

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
                    'round_length': timestep_in_round,  # This will be final length when round completes
                    'timestep_in_round': timestep_in_round,
                    'server_connections': server_connections,
                    'target_connections': target_connections,
                    'round_complete': server_connected_to_all
                }

                current_matrix_index += 1

                # Safety timeout: If round takes too long, force completion
                if timestep_in_round >= 20:  # Max 20 timestamps per round
                    print(f"  ⚠ Round {self.round_number} timeout after {timestep_in_round} timestamps")
                    server_connected_to_all = True

            # Update round_length for all timestamps in this completed round
            actual_round_length = timestep_in_round
            for i in range(round_start_index, current_matrix_index):
                if i < len(adjacency_matrices):
                    timestamp_key = adjacency_matrices[i][0]
                    if timestamp_key in algorithm_output:
                        algorithm_output[timestamp_key]['round_length'] = actual_round_length

            print(f"Round {self.round_number} completed in {actual_round_length} timestamps")

            # Move to next round
            self.round_number += 1

        # Store the algorithm output data for external access
        self.algorithm_output_data = algorithm_output

        if self.output_to_file:
            self.output.write_to_file(algorithm_output)

        self.output.set_result(algorithm_output)
