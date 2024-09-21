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
- 2024-08-24: getters and setters for satellite names by Yuganya Perumal
- 2024-09-09: setting output instead of return and algorithm core to handle algorithm steps by Yuganya Perumal

Usage: 
Instantiate AlgorithmHandler and assign Algorithm. 
Then run start_algorithm() with adjacency matrices. 
"""
from interfaces.handler import Handler
from algorithm_core import Algorithm
import numpy as npy
class AlgorithmHandler(Handler):

    # Constructor
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.adjacency_matrices = []
    @staticmethod 
    def read_adjacency_matrices(file_name):
        adjacency_matrices = []
        with open(file_name, 'r') as file:
            while True:
                line = file.readline().strip()
                if not line:
                    break

                # Extract the timestamp from line starting with "Time: "
                if line.startswith("Time: "):
                    time_stamp = line.split("Time: ")[1]
                else:
                    continue

                matrix = []
                while True:
                    line = file.readline().strip()
                    if not line or line.startswith("Time: "):
                        # Step back and Break, if we see another "Time: " 
                        if line:
                            file.seek(file.tell() - len(line) - 1)
                        break
                    matrix.append(list(map(int, line.split())))
                adjacency_matrices.append((time_stamp, npy.array(matrix)))
        return adjacency_matrices
      
    def parse_input(self,file_name):
        adjacency_matrices = self.read_adjacency_matrices(file_name)
        self.send_adjmatrices(adjacency_matrices)

    def send_adjmatrices(self, adj_matrices):
        self.algorithm.set_adjacency_matrices(adj_matrices)

    def run_module(self):
        self.algorithm.start_algorithm_steps() 
        