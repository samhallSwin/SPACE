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
from typing import NamedTuple

from sat_sim.sat_sim import SatSim
from sat_sim.sat_sim_config import SatSimConfig
from sat_sim.sat_sim_handler import SatSimHandler
from sat_sim.sat_sim_output import SatSimOutput

from algorithm import Algorithm
from algorithm_config import AlgorithmConfig
from algorithm_handler import AlgorithmHandler
from algorithm_output import AlgorithmOutput

from fl_core import FederatedLearning
from fl_config import FLConfig
from fl_handler import FLHandler
from fl_output import FLOutput

# Instantiate relevant classes

# Instantiate Handlers, Configs and Concretes
# (Outputs get instantiated within Concretes)
class SatSimModule(NamedTuple):
    config: SatSimConfig
    handler: SatSimHandler
    output: SatSimOutput

class AlgorithmModule(NamedTuple):
    config: AlgorithmConfig
    handler: AlgorithmHandler
    output: AlgorithmOutput

class FLModule(NamedTuple):
    config: FLConfig
    handler: FLHandler
    output: FLOutput

def create_sat_sim_module() -> SatSimModule:
    sat_sim = SatSim()
    sat_sim_config = SatSimConfig(sat_sim)
    sat_sim_handler = SatSimHandler(sat_sim)

    return SatSimModule(sat_sim_config, sat_sim_handler, sat_sim.output)

def create_algorithm_module() -> AlgorithmModule:
    algorithm = Algorithm()
    algorithm_config = AlgorithmConfig(algorithm)
    algorithm_handler = AlgorithmHandler(algorithm)
    
    return AlgorithmModule(algorithm_config, algorithm_handler, algorithm.output)

def create_fl_module() -> FLModule:
    fl = FederatedLearning()
    fl_config = FLConfig(fl)
    fl_handler = FLHandler(fl)

    return FLModule(fl_handler, fl_config, fl.output)

