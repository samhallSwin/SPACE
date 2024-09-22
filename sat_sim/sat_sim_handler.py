"""
Filename: sat_sim_handler.py
Author: Md Nahid Tanjum
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.handler import Handler

class SatSimHandler(Handler):
    #Handles the input of TLE data for satellite simulation processes.

    def __init__(self, sat_sim):
        self.sat_sim = sat_sim
        self.tle_data = None

    def parse_input(self, file):
        #Reads TLE data from a file.
        tle_data = self.read_tle_file(file)
        self.send_data(tle_data)

    def send_data(self, data):
        #Sends parsed TLE data to the SatSim module.
        self.sat_sim.set_tle_data(data)

    def run_module(self):
        #Runs the simulation module after setting TLE data.
        return self.sat_sim.run_with_adj_matrix()

    def read_tle_file(self, file_path):
        #Reads TLE data from the specified file path.
        tle_data = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):

                name = lines[i].strip()
                tle_line1 = lines[i+1].strip()
                tle_line2 = lines[i+2].strip()
                tle_data[name] = [tle_line1, tle_line2]
        return tle_data
