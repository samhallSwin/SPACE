"""
Filename: sat_sim_output.py
Author: Nahid Tanjum

This module manages the output processes for the Satellite Simulator, including writing data
to text and CSV files.
"""

import numpy as np
import csv

class SatSimOutput:
    # Prepares and sends data to Federated Learning module and handles writing to files.
    def __init__(self):
        self.fl_input = None
        self.matrices = None
        self.timestamps = None

    def write_to_file(self, file, matrices):
        #Write data to a text file.
        self.matrices = matrices
        try:
            if not matrices:
                print(f"No data to write to {file}.")
                return False

            with open(file, "a") as f:
                for timestamp, matrix in matrices:
                    if matrix.size == 0:
                        print(f"Skipping empty matrix at {timestamp}")
                        continue
                    f.write(f"Time: {timestamp}\n")
                    np.savetxt(f, matrix, fmt='%d')
                    f.write("\n")
            return True
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False

    def write_to_csv(self, csv_file, matrices, timestamps, keys):
        #Write data to a CSV file.
        self.matrices = matrices
        self.timestamps = timestamps
        try:
            if not matrices or not timestamps:
                print(f"No data to write to {csv_file}.")
                return False

            # Ensure that matrix sizes are consistent.
            max_size = max((len(matrix) for _, matrix in matrices), default=0)

            if max_size == 0:
                print(f"No valid matrix data to write to CSV {csv_file}.")
                return False

            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Write header row
                writer.writerow(['Time', 'Matrix Size'])
                # Iterate through matrices and timestamps
                for timestamp, matrix in matrices:
                    if matrix.size == 0:
                        print(f"Skipping empty matrix at {timestamp}")
                        continue

                    # Write the timestamp and matrix size first
                    writer.writerow([timestamp, matrix.shape[0]])

                    # Write the matrix row by row
                    for row in matrix:
                        writer.writerow(row)
                    writer.writerow([])

            return True
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            return False

    def set_result(self):
        pass

    def set_fl_input(self, fl_input):
        self.fl_input = fl_input

    def get_result(self):
        return {
            'matrices': self.matrices,
        }
