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
from sat_sim.sat_sim import SatSim

class SatSimConfig(Config):

    # Constructor, accepts core module
    def __init__(self, sat_sim: SatSim):
        self.sat_sim = sat_sim
        self.options = None

    # Traverse JSON options to check for nested objects?
    def read_options(self, options):
        self.options = options
        self.set_sat_sim()

    def read_options_from_file(file):
        return super().read_options_from_file()

    def set_sat_sim(self):
        self.sat_sim.set_duration(self.options["duration"])
        self.sat_sim.set_timestep(self.options["timestep"])
        self.sat_sim.set_output_file_flag(self.options["module_settings"]["output_to_file"])
        
