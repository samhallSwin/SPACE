"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round. 
Author: Elysia Guglielmo
Date: 2024-08-02
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation.
- 2024-01-31: Core Integration Setup for fed avg 

Todo/notes:
- Model Intergration +
Usage: 

"""
from interfaces.handler import Handler
from fl_core import FederatedLearning
# algorithm input


class FLHandler(Handler):

    # Constructor
    def __init__(self, fl_core: FederatedLearning):
        self.federated_learning = fl_core

    # for external file support
    def parse_input(file):
        return super().parse_input()

    def run_module(self):
        self.federated_learning.run()
