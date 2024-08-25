"""
Filename: sat_sim_handler.py
Author: Md Nahid Tanjum"""
import os
import logging
from interfaces.handler import Handler

class SatSimHandler(Handler):
    '''
    Handles the input of TLE data for satellite simulation processes.
    '''

    def __init__(self, sat_sim):
        self.sat_sim = sat_sim
        self.tle_data = None
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def parse_input(self, file):
        """Reads TLE data from a file, ensuring path correctness and data integrity."""
        try:
            tle_data = self.read_tle_file(file)
            if tle_data:
                self.send_data(tle_data)
            else:
                logging.error("No TLE data found or data is incorrect.")
        except Exception as e:
            logging.error(f"Failed to read or process TLE data: {e}")

    def send_data(self, data):
        """Sends parsed TLE data to the SatSim module."""
        if data:
            self.sat_sim.set_tle_data(data)
        else:
            logging.error("Attempted to send empty or invalid TLE data.")

    def run_module(self):
        """Runs the simulation module after setting TLE data."""
        if self.sat_sim.tle_data:
            return self.sat_sim.run_with_adj_matrix()
        else:
            logging.error("TLE data not set before running simulation.")
            return None

    def read_tle_file(self, file_path):
        """Reads TLE data from the specified file path."""
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")
        tle_data = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) % 3 != 0:
                logging.error("TLE file format error: Each entry should consist of three lines.")
                return {}
            for i in range(0, len(lines), 3):
                name = lines[i].strip()
                tle_line1 = lines[i+1].strip()
                tle_line2 = lines[i+2].strip()
                tle_data[name] = [tle_line1, tle_line2]
        return tle_data
