"""
Filename: sat_sim_output.py
Author: Md Nahid Tanjum"""

import numpy as np
import csv

from interfaces.output import Output

class SatSimOutput:
    #Prepares and sends data to Federated Learning module and handles writing to files.
    def __init__(self):
        self.fl_input = None
        self.matrices = None

    def write_to_file(self, file, matrices):
        #Writes matrices to the specified file with timestamps.
        self.matrices = matrices
        with open(file, "a") as f:
            for timestamp, matrix in matrices:
                f.write(f"Time: {timestamp}\n")
                np.savetxt(f, matrix, fmt='%d')
                f.write("\n")

    def write_to_csv(self, csv_file, matrices):
        #Writes matrices to the specified CSV file with timestamps and metadata.
        self.matrices = matrices
        max_size = max(len(matrix) for _, matrix in matrices) 
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            for timestamp, matrix in matrices:
                writer.writerow(['Time:', timestamp, 'Size:', len(matrix)])
                for row in matrix:
                    if len(row) < max_size:
                        # Extend rows with zeros to ensure all rows have the same number of columns
                        row.extend([0] * (max_size - len(row)))
                    writer.writerow(row)
                writer.writerow([])

    def set_result(self):
        #Placeholder for setting results, potentially integrating with a learning module.
        pass

    def set_fl_input(self, fl_input):
        #Sets the input for the Federated Learning module, if applicable.
        self.fl_input = fl_input


    def get_result(self):
        #Retrieves the stored result.
        return self.matrices
