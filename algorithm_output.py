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
import numpy as npy
class AlgorithmOutput(Output):

    # Constructor
    def __init__(self):
        pass

    '''
    def write_to_file(file):
        return super().write_to_file()
    '''  
    def write_to_file(self, algorithmoutput):
        file_name = 'algorithm_output_data.txt'
        with open(file_name, 'w') as file:
            for timestamp, data in algorithmoutput.items():
                file.write(f"Timestamp: {timestamp}\n")
                file.write(f"Selected Node: {data['selected_node']} (Aggregator Flag: {data['aggregator_flag']})\n")
                file.write("Trimmed Matrix:\n")
                npy.savetxt(file, data['trimmed_matrix'], fmt='%d', delimiter=',')
                file.write("\n")
    def set_result():
        pass

    # Start algorithm