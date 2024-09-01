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

def add_algorithm_args(parser):
    algorithm_group = parser.add_argument_group(ModuleKeys.ALGORITHM)

    # Add args here

def add_fl_args(parser):
    fl_group = parser.add_argument_group(ModuleKeys.FL)

    # Add args here
    fl_group.add_argument('--num-clients', type=int, help='Number of Federated Learning clients for the simulation')
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

def add_algorithm_args(parser):
    algorithm_group = parser.add_argument_group(ModuleKeys.ALGORITHM)

    # Add args here

def add_fl_args(parser):
    fl_group = parser.add_argument_group(ModuleKeys.FL)

    # Add args here
    fl_group.add_argument('--num-clients', type=int, help='Number of Federated Learning clients for the simulation')
    fl_group.add_argument('--model', type=str, help='The Model to use for the simulation (mnist/resnet)') 
    # Currently implemented models include MNIST & resnet, will have to adapt this once resnet trained with bigearthnet dataset is implemented
    fl_group.add_argument('--epochs', type=int, help='Number of Epochs used for simulation')
    fl_group.add_argument('--batch-size', type=int, help='The Batch Size of the model to use for simulation')

