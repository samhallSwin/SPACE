"""
Filename: algorithm_output.py
Description: Prepares and sends data to Federated Learning module
Author: Elysia Guglielmo
Date: 2024-08-07
Version: 1.0
Python Version: 

Changelog:
- 2024-08-07: Initial creation.

Usage: 
Instantiate AlgorithmOutput and assign FLInput. 
"""
from interfaces.output import Output

class AlgorithmOutput(Output):

    # Constructor
    def __init__(self):
        self.fl_input = None

    def write_to_file(file):
        return super().write_to_file()
    
    def set_result():
        pass

    def set_fl_input(self, fl_input):
        self.fl_input = fl_input

    # Start algorithm