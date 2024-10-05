"""
Filename: sat_sim_output.py
Author: Md Nahid Tanjum

This module manages the output processes for the Satellite Simulator, including writing data
to text and CSV files.
"""

import numpy as np
import csv
import os

class SatSimOutput:
    #Handles writing simulation data to text and CSV files.
    def __init__(self):
        #Initializes the SatSimOutput with default attributes.
        self.fl_input = None
        self.matrices = None
        self.timestamps = None

    def write_to_file(self, file, matrices, keys=None):
        #Write data to a text file with headers.
        self.matrices = matrices
        try:
            if not matrices:
                print(f"No data to write to {file}.")
                return False

            # Check if file exists and is non-empty
            file_exists = os.path.isfile(file)
            write_header = not file_exists or os.path.getsize(file) == 0

            with open(file, "a") as f:
                if write_header:
                    if keys and len(keys) > 0:
                        num_satellites = len(keys)
                        f.write(f"Number of satellites: {num_satellites}\n")
                        sat_labels = ','.join(keys)
                        f.write(f"{sat_labels}\n\n")
                        print(f"Written header with {num_satellites} satellites to {file}.")
                    else:
                        if matrices and len(matrices[0][1].shape) > 1:
                            num_satellites = matrices[0][1].shape[1]
                            f.write(f"Number of satellites: {num_satellites}\n")
                            sat_labels = ','.join([f'Sat{i+1}' for i in range(num_satellites)])
                            f.write(f"{sat_labels}\n\n")
                            print(f"Written default header with {num_satellites} satellites to {file}.")
                        else:
                            print("Cannot determine the number of satellites.")
                            return False

                for timestamp, matrix in matrices:
                    if matrix.size == 0:
                        print(f"Skipping empty matrix at {timestamp}")
                        continue
                    f.write(f"Time: {timestamp}\n")
                    np.savetxt(f, matrix, fmt='%d', delimiter=',')
                    f.write("\n")
                    print(f"Written matrix for time {timestamp} to {file}.")
            return True
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False

    def write_to_csv(self, csv_file, matrices, timestamps, keys=None):
        #Write data to a CSV file with headers.
        self.matrices = matrices
        self.timestamps = timestamps
        try:
            if not matrices or not timestamps:
                print(f"No data to write to {csv_file}.")
                return False

            # Determine maximum number of satellites
            max_size = max((matrix.shape[1] for _, matrix in matrices if matrix.size > 0), default=0)

            if max_size == 0:
                print(f"No valid matrix data to write to CSV {csv_file}.")
                return False

            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                
                # Write number of satellites
                writer.writerow(['Number of satellites:', max_size])
                # Write satellite labels
                if keys and len(keys) == max_size:
                    sat_labels = keys
                else:
                    sat_labels = [f'Sat{i+1}' for i in range(max_size)]
                writer.writerow(sat_labels)
                # Write header for matrix data
                writer.writerow(['Time', 'Matrix Size'])
                # Write each matrix with timestamp
                for (timestamp, matrix) in matrices:
                    if matrix.size == 0:
                        print(f"Skipping empty matrix at {timestamp}")
                        continue

                    writer.writerow([timestamp, matrix.shape[0]])
                    for row in matrix:
                        writer.writerow(row.tolist())                    
                    writer.writerow([])

            return True
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            return False

    def set_result(self):
        #Placeholder for setting results.
        pass

    def set_fl_input(self, fl_input):
        #Sets the Federated Learning input.
        self.fl_input = fl_input

    def get_result(self):
        #Retrieves the simulation results.
        return {
            'matrices': self.matrices,
        }