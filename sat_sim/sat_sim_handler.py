"""
Filename: sat_sim_handler.py
Author: Md Nahid Tanjum
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.handler import Handler

class SatSimHandler(Handler):
    """
    Handles the input of TLE data for satellite simulation processes, 
    providing flexibility to parse data either from a file or directly from data structures.
    """

    def __init__(self, sat_sim):
        self.sat_sim = sat_sim
        self.tle_data = None
        self.data_loaded = False  # To ensure data is loaded only once

    def parse_input(self, data):
        """
        Implements the abstract method from the Handler class.
        This method will decide whether to read from a file or from a data structure.
        """
        if isinstance(data, str):  # If it's a string, assume it's a file path
            self.parse_input_file(data)
        else:  # Otherwise, assume it's a data structure (like a dictionary or list)
            self.parse_input_data(data)

    def parse_input_file(self, file):
        """
        Reads TLE data from a file and sends it to the SatSim module.
        This is useful when reading TLE data from a file source.
        """
        if self.data_loaded:  # Prevent loading the data multiple times
            return
        try:
            tle_data = self.read_tle_file(file)
            self.send_data(tle_data)
            self.data_loaded = True  # Mark as loaded to prevent multiple calls
        except Exception as e:
            print(f"Error reading TLE file: {e}")

    def parse_input_data(self, tle_data):
        """
        Sends parsed TLE data directly to the SatSim module.
        This is useful when the TLE data is directly passed as a data structure.
        Example: Passing TLE data from another module or simulation pipeline.
        """
        if self.data_loaded:  # Prevent sending the data multiple times
            return
        self.send_data(tle_data)
        self.data_loaded = True

    def send_data(self, data):
        """
        Sends parsed TLE data to the SatSim module.
        Ensures the SatSim module has the correct data for running the simulation.
        """
        try:
            self.sat_sim.set_tle_data(data)
        except Exception as e:
            print(f"Error sending data to SatSim: {e}")

    def run_module(self):
        """
        Runs the simulation module after setting TLE data. 
        It uses the SatSim instance to generate the adjacency matrix or run any simulation process.
        """
        return self.sat_sim.run_with_adj_matrix()

    def read_tle_file(self, file_path):
        """
        Reads TLE data from the specified file path.
        This method extracts satellite information from a TLE file and formats it for the simulation.
        """
        tle_data = {}
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 3):
                    name = lines[i].strip()
                    tle_line1 = lines[i+1].strip()
                    tle_line2 = lines[i+2].strip()
                    tle_data[name] = [tle_line1, tle_line2]
            return tle_data
        except Exception as e:
            print(f"Error reading TLE file at {file_path}: {e}")
            return None
