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

    def find_best_server_for_round(self, start_matrix_index=0):
        """
        Find the server that will connect to ALL other satellites the fastest.
        Uses cumulative connectivity - server can connect to satellites one at a time.
        """
        if start_matrix_index >= len(self.adjacency_matrices):
            return 0, 5  # Default fallback

        num_satellites = len(self.satellite_names)
        best_server = 0
        best_timestamps_needed = float('inf')

        # Check each satellite as potential server
        for server_idx in range(num_satellites):
            timestamps_needed = self.calculate_cumulative_connection_time(server_idx, start_matrix_index)

            # Apply load balancing - prefer servers that haven't been selected much
            load_penalty = self.selection_counts[server_idx] * 0.5
            adjusted_time = timestamps_needed + load_penalty

            if adjusted_time < best_timestamps_needed:
                best_timestamps_needed = adjusted_time
                best_server = server_idx

        # Update selection count
        self.selection_counts[best_server] += 1

        actual_timestamps = int(best_timestamps_needed - self.selection_counts[best_server] * 0.5)

        print(f"Round {self.round_number}: Selected Server {best_server} ({self.satellite_names[best_server]}) "
            f"- will connect to all {num_satellites-1} satellites in {actual_timestamps} timestamps")

        return best_server, actual_timestamps

    def calculate_cumulative_connection_time(self, server_idx, start_matrix_index, max_lookahead=20):
        """
        Calculate how many timestamps it takes for this server to connect to ALL other satellites
        using cumulative connectivity (can connect to satellites one at a time).
        """
        num_satellites = len(self.satellite_names)
        target_satellites = set(range(num_satellites))
        target_satellites.remove(server_idx)  # Remove self from target list

        connected_satellites = set()  # Track which satellites we've connected to

        for timesteps_ahead in range(max_lookahead):
            matrix_idx = start_matrix_index + timesteps_ahead
            if matrix_idx >= len(self.adjacency_matrices):
                break

            _, matrix = self.adjacency_matrices[matrix_idx]

            # Check which satellites this server connects to in this timestamp
            for target_sat in range(num_satellites):
                if target_sat != server_idx and matrix[server_idx][target_sat] == 1:
                    connected_satellites.add(target_sat)

            # Check if we've connected to all target satellites
            if connected_satellites == target_satellites:
                return timesteps_ahead + 1  # +1 because we need to include this timestamp

        # If server never connects to all satellites within lookahead, return high penalty
        return max_lookahead

    def start_algorithm_steps(self):
        """
        Realistic FLOMPS algorithm with cumulative connectivity tracking.
        """
        adjacency_matrices = self.get_adjacency_matrices()
        algorithm_output = {}

        current_matrix_index = 0

        while current_matrix_index < len(adjacency_matrices):
            # Select best server for this round
            selected_server, estimated_timestamps = self.find_best_server_for_round(current_matrix_index)
            self.current_server = selected_server

            # Initialize tracking for this round
            round_start_index = current_matrix_index
            timestep_in_round = 0
            num_satellites = len(self.satellite_names)
            target_satellites = set(range(num_satellites))
            target_satellites.remove(selected_server)  # Remove self from target list
            connected_satellites = set()  # Track cumulative connections
            round_complete = False

            print(f"  Target: Server {selected_server} needs to connect to all {len(target_satellites)} other satellites")

            while current_matrix_index < len(adjacency_matrices) and not round_complete:
                timestep_in_round += 1
                time_stamp, matrix = adjacency_matrices[current_matrix_index]

                # Check which NEW satellites this server connects to in this timestamp
                current_timestamp_connections = set()
                for target_sat in range(num_satellites):
                    if target_sat != selected_server and matrix[selected_server][target_sat] == 1:
                        current_timestamp_connections.add(target_sat)
                        connected_satellites.add(target_sat)  # Add to cumulative set

                # Check if server has now connected to ALL other satellites (cumulatively)
                if connected_satellites == target_satellites:
                    round_complete = True
                    print(f"  ✓ Server {selected_server} connected to all {len(target_satellites)} satellites at timestep {timestep_in_round}")
                    print(f"    Final connections: {sorted(list(connected_satellites))}")
                else:
                    missing_satellites = target_satellites - connected_satellites
                    print(f"  → Server {selected_server} connected to {len(connected_satellites)}/{len(target_satellites)} satellites")
                    print(f"    Still need: {sorted(list(missing_satellites))}")

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
                    'aggregator_flag': True,
                    'round_number': self.round_number,
                    'phase': phase,
                    'target_node': selected_server,
                    'round_length': timestep_in_round,  # Will be updated when round completes
                    'timestep_in_round': timestep_in_round,
                    'server_connections_current': len(current_timestamp_connections),
                    'server_connections_cumulative': len(connected_satellites),
                    'target_connections': len(target_satellites),
                    'connected_satellites': sorted(list(connected_satellites)),
                    'missing_satellites': sorted(list(target_satellites - connected_satellites)),
                    'round_complete': round_complete
                }

                current_matrix_index += 1

                # Safety timeout: If round takes too long, force completion
                if timestep_in_round >= 20:
                    print(f"  ⚠ Round {self.round_number} timeout after {timestep_in_round} timestamps")
                    print(f"    Only connected to {len(connected_satellites)}/{len(target_satellites)} satellites")
                    round_complete = True

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
