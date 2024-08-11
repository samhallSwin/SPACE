"""
Filename: sat_sim_config.py
Description: Converts Satellite Simulator JSON options to local data types and configures core SatSim module via setters. 
Author:
Date: 2024-08-11
Version: 1.0
Python Version: 

Changelog:
- 2024-08-11: Initial creation.

Usage: 

"""
from interfaces.config import Config

class SatSimConfig(Config):

    # Constructor, accepts core module
    def __init__(self, sat_sim):
        self.sat_sim = sat_sim
        self.options = None

    # Traverse JSON options to check for nested objects?
    def read_options(self, options):
        self.options = options
        self.set_sat_sim()

    def read_options_from_file(file):
        return super().read_options_from_file()

    def set_sat_sim(self):
        pass
        
