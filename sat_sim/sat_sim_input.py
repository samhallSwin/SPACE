"""
Filename: sat_sim_input.py
Description: Reads in TLE file for Satellite Simulation
Authors: Elysia Guglielmo, Md Nahid Tanjum
Date: 2024-08-11
Version: 1.0
Python Version: 

Changelog:
- 2024-08-11: Initial creation.

Usage: 
Instantiate SatSimInput and assign SatSim. 
"""

from interfaces.input import Input
from sat_sim.sat_sim import SatSim

class SatSimInput(Input):

    def __init__(self, sat_sim: SatSim, hours, timeframe):
        self.sat_sim = sat_sim
        self.tle_data = None
        self.hours = hours
        self.timeframe = timeframe

    def parse_input(self, file):
        tle_data = self.read_tle_file(file)
        self.send_data(tle_data)
        self.start_module()

    def send_data(self, data):
        self.sat_sim.set_tle_data(data)

    def start_module(self):
        self.sat_sim.run_with_adj_matrix()

    def read_tle_file(self, file_path):
        tle_data = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                name = lines[i].strip()
                tle_line1 = lines[i + 1].strip()
                tle_line2 = lines[i + 2].strip()
                tle_data[name] = [tle_line1, tle_line2]
        return tle_data
    