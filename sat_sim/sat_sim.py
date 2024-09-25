"""
Filename: sat_sim.py
Author: Nahid Tanjum
Description: This script handles satellite simulations using TLE data and provides functionality 
             for generating adjacency matrices based on satellite positions, with support for 
             both GUI and CLI operations.
"""

import argparse
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import numpy as np
import os
from .sat_sim_output import SatSimOutput
from PyQt5.QtWidgets import QApplication

from .sat_sim_output import SatSimOutput
from .sat_sim_config import SatSimConfig
from .sat_sim_handler import SatSimHandler

class SatSim:
    # SatSim class manages satellite simulation based on TLE data and handles both GUI and CLI modes.
    def __init__(self, start_time=None, end_time=None, timestep=30, output_file_type="csv", gui_enabled=False, tle_data=None):
        # Initialize the SatSim instance with optional parameters for start and end times, timestep, output file type, and GUI mode.
        self.tle_data = tle_data if tle_data else None
        self.timestep = timedelta(minutes=timestep)
        self.sf_timescale = load.timescale()
        self.output = SatSimOutput()
        self.start_time = start_time
        self.end_time = end_time
        self.output_file_type = output_file_type
        self.output_path = ""
        self.gui_enabled = gui_enabled

    def set_gui_enabled(self, enabled):
        # Enable or disable GUI mode.
        self.gui_enabled = enabled

    def set_tle_data(self, tle_data):
        # Set the TLE data used in the simulation.
        self.tle_data = tle_data if tle_data else None

    def set_timestep(self, timestep):
        # Set the simulation timestep (in minutes).
        self.timestep = timedelta(minutes=timestep)

    def set_start_end_times(self, start, end):
        # Set the start and end times for the simulation.
        self.start_time = start
        self.end_time = end

    def set_output_file_type(self, file_type):
        # Set the output file format (csv or txt).
        self.output_file_type = file_type

    def run_gui(self):
        # Launch the satellite simulation GUI.
        from .sat_sim_gui import SatSimGUI
        import sys
        app = QApplication(sys.argv)
        gui = SatSimGUI(self)  # Pass the current instance of SatSim to the GUI
        gui.show()
        sys.exit(app.exec_())

    def read_tle_file(self, file_path):
        # Read TLE data from a specified file.
        try:
            tle_data = {}
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 3):
                    name = lines[i].strip()
                    tle_line1 = lines[i + 1].strip()
                    tle_line2 = lines[i + 2].strip()
                    tle_data[name] = [tle_line1, tle_line2]
            return tle_data
        except Exception as e:
            print(f"Error reading TLE file: {e}")
            return None

    def get_satellite_positions(self, time):
        # Get satellite positions at a given time.
        if isinstance(time, datetime):
            # Convert Python datetime object to Skyfield time object
            time = self.sf_timescale.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)

        positions = {}
        # Calculate satellite positions using TLE data
        for name, tle in self.tle_data.items():
            tle_line1, tle_line2 = tle
            satellite = EarthSatellite(tle_line1, tle_line2, name)
            geocentric = satellite.at(time)
            positions[name] = geocentric.position.km
        return positions

    def calculate_distance(self, pos1, pos2):
        # Calculate the Euclidean distance between two satellite positions.
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def generate_adjacency_matrix(self, positions, distance_threshold=10000):
        # Generate an adjacency matrix based on the distances between satellites.
        keys = list(positions.keys())
        size = len(keys)
        adj_matrix = np.zeros((size, size), dtype=int)

        # Compute distances between all satellite pairs and populate the adjacency matrix
        for i in range(size):
            for j in range(i + 1, size):
                dist = self.calculate_distance(positions[keys[i]], positions[keys[j]])
                adj_matrix[i, j] = adj_matrix[j, i] = 1 if dist < distance_threshold else 0

        return adj_matrix, keys

    def run_with_adj_matrix(self):
        # Run the simulation in CLI mode and generate adjacency matrices.
        if not self.tle_data:
            print("No TLE data loaded. Please provide valid TLE data.")
            return []

        current_time = self.start_time
        if isinstance(self.end_time, datetime):
            end_time = self.sf_timescale.utc(self.end_time.year, self.end_time.month, self.end_time.day,
                                             self.end_time.hour, self.end_time.minute, self.end_time.second)
        else:
            end_time = self.end_time

        matrices = []
        timestamps = []
        # Loop through time increments and generate adjacency matrices
        while current_time.utc < end_time.utc:
            try:
                if isinstance(current_time, datetime):
                    t = self.sf_timescale.utc(current_time.year, current_time.month, current_time.day,
                                              current_time.hour, current_time.minute, current_time.second)
                else:
                    t = current_time

                positions = self.get_satellite_positions(t)

                if not positions:
                    print(f"Skipping timestep {current_time}: No positions calculated.")
                    current_time += self.timestep
                    continue

                # Generate adjacency matrix
                adj_matrix, keys = self.generate_adjacency_matrix(positions)
                matrices.append((t.utc_datetime().strftime('%Y-%m-%d %H:%M:%S'), adj_matrix))
                current_time += self.timestep

            except Exception as e:
                print(f"Error during simulation: {e}")
                break

        # Save results to output file
        output_file = os.path.join(os.getcwd(), f"output.{self.output_file_type}")
        print(f"Writing output to {output_file}")

        if matrices:
            if self.output_file_type == "txt":
                self.output.write_to_file(output_file, matrices)
            elif self.output_file_type == "csv":
                timestamps = [timestamp for timestamp, _ in matrices]
                keys = list(self.tle_data.keys())
                self.output.write_to_csv(output_file, matrices, timestamps, keys)
        else:
            print("No data to write to output file.")

        return matrices

    def run(self):
        # Run the simulation based on the mode (CLI or GUI).
        if self.gui_enabled:
            self.run_gui()  # Run in GUI mode
        else:
            self.run_with_adj_matrix()  # Run in CLI mode
