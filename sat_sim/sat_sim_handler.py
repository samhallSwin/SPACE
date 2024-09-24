"""
Filename: sat_sim_handler.py
Author: Nahid Tanjum

"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.handler import Handler

class SatSimHandler(Handler):
    #Handles the input of TLE data for satellite simulation processes,
    def __init__(self, sat_sim):
        self.sat_sim = sat_sim
        self.data_loaded = False

    def parse_file(self, file):
        #Reads TLE data from a file and sends it to the SatSim module.
        if self.data_loaded:
            return
        try:
            tle_data = self.read_tle_file(file)
            self.send_data(tle_data)
            self.data_loaded = True
        except Exception as e:
            print(f"Error reading TLE file: {e}")

    def parse_data(self, data):
        #Sends parsed TLE data directly to the SatSim module.
        if self.data_loaded:
            return
        self.send_data(data)
        self.data_loaded = True

    def send_data(self, data):
        #Sends parsed TLE data to the SatSim module.
        try:
            self.sat_sim.set_tle_data(data)
        except Exception as e:
            print(f"Error sending data to SatSim: {e}")

    def run_module(self):
        #Executes the SatSim module by calling its run method.
        self.sat_sim.run()

    def read_tle_file(self, file_path):
        #Reads TLE data from the specified file path.
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
