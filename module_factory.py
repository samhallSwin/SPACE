"""
Filename: module_factory.py
Description: Creates modules for simulation pipeline
Author: Elysia Guglielmo
Date: 2024-08-011
Version: 1.0
Python Version: 

Changelog:
- 2024-08-11: Initial creation.

Usage: Access a module and its I/O interfaces by calling the relevant create function

"""

from sat_sim.sat_sim import SatSim
from sat_sim.sat_sim import SatSimInput

from algorithm import Algorithm
from algorithm_config import AlgorithmConfig
from algorithm_input import AlgorithmInput

from federated_learning import FederatedLearning
from fl_config import FLConfig
from fl_input import FLInput

# Instantiate relevant classes

# Instantiate Inputs, Configs and Concretes
# (Outputs get instantiated within Concretes)

def create_sat_sim_module():
    sat_sim = SatSim()
    sat_sim_input = SatSimInput

def create_algorithm_module():
    algorithm = Algorithm()
    algorithm_config = AlgorithmConfig(algorithm)
    algorithm_input = AlgorithmInput(algorithm)
    
    return algorithm_config, algorithm_input, algorithm.output

def create_fl_module():
    fl = FederatedLearning()
    fl_config = FLConfig(fl)
    fl_input = FLInput(fl)

    return fl_input, fl_config, fl.output

