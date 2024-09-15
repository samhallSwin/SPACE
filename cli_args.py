import argparse
from enum import Enum

class ModuleKeys(Enum):
    SAT_SIM = "sat_sim"
    ALGORITHM = "algorithm"
    FL = "federated_learning"

def add_sat_sim_args(parser):
    sat_sim_group = parser.add_argument_group(ModuleKeys.SAT_SIM)

    # Add args here
    sat_sim_group.add_argument('--start-time', type=int, help='The start date/time for the satellite simulation')
    sat_sim_group.add_argument('--end-time', type=int, help='The end date/time for the satellite simulation')
    sat_sim_group.add_argument('--duration', type=int, help='Duration of the simulation (in hours)')
    sat_sim_group.add_argument('--timestep', type=int, help='Timestep for the simulation')

def add_algorithm_args(parser):
    algorithm_group = parser.add_argument_group(ModuleKeys.ALGORITHM)

    # Add args here

def add_fl_args(parser):
    fl_group = parser.add_argument_group(ModuleKeys.FL)

    # Add args here
    fl_group.add_argument('--num-clients', type=int, help='Number of Federated Learning clients for the simulation')
