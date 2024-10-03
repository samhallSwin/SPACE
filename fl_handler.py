"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round. 
Author: Elysia Guglielmo and Connor Bett
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
        self.flam = None

    def parse_data(self, data):
        if data is not None:
            self.flam = data
        else:
            raise FileNotFoundError(f"FL attempted to parse a FLAM, but was empty ")

        return super().parse_data()

    # for external file support
    def parse_file(self, file):
        return super().parse_input()

    def run_module(self):
        if self.flam is None:
            self.federated_learning.run()
        else:
            self.federated_learning.set_flam(self.flam)
            self.federated_learning.run()
