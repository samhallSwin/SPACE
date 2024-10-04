"""
Filename: algorithm_handler.py
Description: Reads satellite link data to perform preprocessing before satellite communication algorithm execution. 
Initial Creator: Elysia Guglielmo (System Architect)
Author: Yuganya Perumal
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.
- 2024-08-24: getters and setters for satellite names.
- 2024-09-09: setting output instead of return and algorithm core to handle algorithm steps.
- 2024-09-22: Implemented Input validation.
- 2024-10-03: Implemented auto generation of satellite names if satellite simulator component does not provide one.

Usage: 
Instantiate AlgorithmHandler and assign Algorithm. 
Then run start_algorithm() with adjacency matrices. 
"""
from interfaces.handler import Handler
from algorithm_core import Algorithm
import numpy as npy
import os
class AlgorithmHandler(Handler):

    # Constructor
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.adjacency_matrices = []
        self.sat_names = []
    @staticmethod 
    def read_adjacency_matrices(file_name):
        # Check if file exist.
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"The file '{file_name}' does not exist.")
        
        adjacency_matrices = []
        with open(file_name, 'r') as file:
            while True:
                line = file.readline().strip()
                if not line:
                    break

                # Check if line starts with string 'Time: '
                if not line.startswith("Time: "):
                    raise ValueError(f"line expected to start with 'Time: ', but got: {line}")

                # Extract the timestamp from line starting with "Time: "
                if line.startswith("Time: "):
                    time_stamp = line.split("Time: ")[1]
                else:
                    continue

                matrix = []
                while True:
                    line = file.readline().strip()
                    if not line or line.startswith("Time: "):
                        # Step back and Break, if seeing another "Time: " 
                        if line:
                            file.seek(file.tell() - len(line) - 1)
                        break
                    # Check if the matrix row structure is as expected.
                    try:
                        matrix_row = list(map(int, line.split()))
                    except ValueError:
                        raise ValueError(f"Matrix row contains unexpected values: {line}")
  
                    # add to matrix after validation
                    matrix.append(matrix_row) 
                    
                    # Check if matrix is empty
                    if not matrix:
                        raise ValueError("The matrix is empty.")
                    # Check if the matrix rows have consistent lengths
                    if len(matrix_row) != len(matrix[0]):
                        raise ValueError(f"row length is different in the matrix: {line}")                  
                adjacency_matrices.append((time_stamp, npy.array(matrix)))
        return adjacency_matrices

    # Validate incoming adjacency matrices satisfy being square and symmetry.
    def validate_adjacency_matrices(self, adjacency_matrices):
        for index, (timestamp, each_matrix) in enumerate(adjacency_matrices):
            # Check if the Adjacency matrix is empty
            if len(each_matrix) == 0 or len(each_matrix[0]) == 0:
                raise ValueError(f"Adjacency Matrix at time '{timestamp}' is empty.")

            # Check if the Adjacency matrix is square
            if len(each_matrix) != len(each_matrix[0]):
                raise ValueError(f"Adjacency Matrix at time '{timestamp}' is not square: {len(each_matrix)}x{len(each_matrix[0])}")

            # Check if the Adjacency matrix is symmetric
            if not npy.array_equal(npy.array(each_matrix), npy.array(each_matrix).T):
                raise ValueError(f"Adjacency Matrix at time '{timestamp}' is not symmetric.") 

        return True

    # Automatically generates satellite names based on the adjacency matrix size, 
    # if satellite names not provided by Satellite Simulator Componenet
    def auto_generate_satellite_names(self,adjacency_matrix_size):
        self.sat_names = [f"Satellite {k+1}" for k in range(adjacency_matrix_size)]
        self.send_satNames(self.sat_names)

    def parse_data(self, data, names):
        if data is None:
            raise ValueError("No incoming Adjacency Martix, unable to proceed.")
        else:
            self.adjacency_matrices = data
            self.sat_names = names
            if self.validate_adjacency_matrices(self.adjacency_matrices):
                if not self.sat_names:
                    no_of_rows = self.adjacency_matrices[0][1].shape[0]
                    self.auto_generate_satellite_names(no_of_rows)
                else:
                    self.algorithm.set_satellite_names(self.sat_names)  
                self.send_adjmatrices(self.adjacency_matrices)

    def parse_file(self, file_name):
        self.adjacency_matrices = self.read_adjacency_matrices(file_name)
        if self.validate_adjacency_matrices(self.adjacency_matrices):
            no_of_rows = self.adjacency_matrices[0][1].shape[0]
            self.auto_generate_satellite_names(no_of_rows) 
            self.send_adjmatrices(self.adjacency_matrices)

    def send_adjmatrices(self, adj_matrices):
        self.algorithm.set_adjacency_matrices(adj_matrices)

    def send_satNames(self, sat_names):
        self.algorithm.set_satellite_names(sat_names)

    def run_module(self):
        self.algorithm.start_algorithm_steps() 
        