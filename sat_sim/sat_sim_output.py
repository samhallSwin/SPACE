"""
Filename: sat_sim_output.py
Description: Prepares and sends data to Federated Learning module
Author: Elysia Guglielmo
Date: 2024-08-15
Version: 1.0
Python Version: 

Changelog:
- 2024-08-15: Initial creation.

Usage: 
"""
import numpy as np

from interfaces.output import Output

class SatSimOutput(Output):

    # Constructor
    def __init__(self):
        self.fl_input = None
        self.matrices = []

    def write_to_file(self, file, matrices):
        with open(file, "w") as f:
            for timestamp, matrix in matrices:
                f.write(f"Time: {timestamp}\n")
                np.savetxt(f, matrix, fmt='%d')
                f.write("\n")
    
    def set_result():
        pass

    def set_fl_input(self, fl_input):
        self.fl_input = fl_input
