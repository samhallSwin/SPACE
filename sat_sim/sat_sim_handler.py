"""
Filename: sat_sim_handler.py
Author: Md Nahid Tanjum

This module handles the input of TLE data for satellite simulation processes. It ensures that TLE data
is loaded correctly from files or direct inputs and then passed to the simulation module.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.handler import Handler

class SatSimHandler(Handler):
    # Handles the input of TLE data for satellite simulation processes.
    def __init__(self, sat_sim):
        self.sat_sim = sat_sim
        self.data_loaded = False

    def parse_file(self, file):
        # Reads TLE data from a file and sends it to the SatSim module if not already loaded.
        if self.data_loaded:
            print("Data already loaded, skipping.")
            return
        try:
            tle_data = self.read_tle_file(file)
            if tle_data:
                self.send_data(tle_data)
                self.data_loaded = True
            else:
                print(f"No data found in file: {file}")
        except Exception as e:
            print(f"Error reading TLE file: {e}")
            raise

    def parse_data(self, data):
        # Sends parsed TLE data directly to the SatSim module if not already loaded.
        if self.data_loaded:
            print("Data already loaded, skipping.")
            return
        self.send_data(data)
        self.data_loaded = True

    def send_data(self, data):
        # Sends parsed TLE data to the SatSim module.
        print("Sending data to SatSim...")
        if not data:
            print("No data to send.")
            return
        try:
            if self.sat_sim:
                print("Sending data to SatSim...")
                self.sat_sim.set_tle_data(data)
            else:
                print("SatSim instance not initialized.")
        except Exception as e:
            print(f"Error sending data to SatSim: {e}")

    def run_module(self):
        # Executes the SatSim module by calling its run method.
        self.sat_sim.run()

    def read_tle_file(self, file_path):
        # Reads TLE data from the specified file path.
        tle_data = {}
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            # Handle files with or without a title line (3LE or 2LE)
            i = 0
            while i < len(lines):
                if lines[i].startswith('1') or lines[i].startswith('2'):
                    # This is the 2LE format (no title line)
                    tle_line1 = lines[i].strip()
                    tle_line2 = lines[i + 1].strip()
                    tle_data[f"Satellite_{i // 2 + 1}"] = [tle_line1, tle_line2]
                    i += 2
                else:
                    # 3LE format (title line included)
                    name = lines[i].strip()
                    tle_line1 = lines[i + 1].strip()
                    tle_line2 = lines[i + 2].strip()
                    tle_data[name] = [tle_line1, tle_line2]
                    i += 3

            if not tle_data:
                return None
            return tle_data

        except OSError:
            raise
        except Exception as e:
            print(f"Error reading TLE file at {file_path}: {e}")
            raise ValueError("Error reading TLE file.")
