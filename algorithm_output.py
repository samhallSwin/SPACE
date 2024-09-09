"""
Filename: algorithm_output.py
Description: Prepares and sends data to Federated Learning module
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal
Date: 2024-08-07
Version: 1.0
Python Version: 

Changelog:
- 2024-08-07: Initial creation.
- 2024-08-24: Added method to write Federated Learning Adjacency Matrix (FLAM) to a file by Yuganya Perumal
- 2024-08-31: Added method to set result as data frame object to be passed as input FL component by Yuganya Perumal.
- 2024-09-09: Algorithm output returns the FLAM.

Usage: 
Instantiate AlgorithmOutput and assign FLInput. 
"""
from interfaces.output import Output
import numpy as npy
import pandas as pds
class AlgorithmOutput(Output):

    # Constructor
    def __init__(self):
        self.flam_output = None

    def set_flam(self, algo_output):
        self.flam_output = algo_output

    def get_flam(self):
        return self.flam_output

    # Write the Federated Learning Adjacency Matrix (FLAM) to the file.
    def write_to_file(self, algorithm_output):
        file_name = 'Federated_Learning_Adjacency_Matrix.txt'
        with open(file_name, 'w') as file:
            for timestamp, algo_op_data in algorithm_output.items():
                file.write(f"Timestamp: {timestamp}\n")
                file.write(f"satellite_count: {algo_op_data['satellite_count']}\n")
                file.write(f"selected_satellite: {algo_op_data['selected_satellite']} (Aggregator Flag: {algo_op_data['aggregator_flag']})\n")
                file.write("Federated Learning Adjacency Matrix:\n")
                npy.savetxt(file, algo_op_data['federatedlearning_adjacencymatrix'], fmt='%d', delimiter=',')
                file.write("\n")
   
    # Pass Federated Learning Adjacency Matrix (FLAM) to Federated Learning Component
    # Aggregator Flag has three values true, false, None
        # Aggregator Flag set to true when a client is selected with most number of connectivity with other clients.
        # Aggregator Flag set to false when a client is not selected due to less number of connectivity with other clients.
        # Aggregator Flag set to None when No connectivity exists between clients.
    def set_result(self, algorithm_output):
        algo_op_rows = []

        for algo_op in algorithm_output.values():
            # Convert the aggregator_flag to 'None' if there is no connectivity among satellites
            aggregator_flag = algo_op['aggregator_flag'] if algo_op['aggregator_flag'] is not None else 'None'
            # Get FLAM
            fl_am = algo_op['federatedlearning_adjacencymatrix']
            # Get satellite count
            satellite_count = algo_op['satellite_count']
            # Append the row with the aggregator flag and matrix
            algo_op_rows.append({
                'satellite_count': satellite_count,
                'aggregator_flag': aggregator_flag,
                'federatedlearning_adjacencymatrix': fl_am
            })
       
        # Convert the list of algorithm output rows to a DataFrame for easy processing and manipulation
        # algo_op_df = pds.DataFrame(algo_op_rows)
        self.set_flam(pds.DataFrame(algo_op_rows))
        return self.get_flam()