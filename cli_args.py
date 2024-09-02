import argparse
from enum import Enum
from module_factory import ModuleKey

# class ModuleKeys(Enum):
#     SAT_SIM = "sat_sim"
#     ALGORITHM = "algorithm"
#     FL = "federated_learning"
#     MODEL = "model"

def add_positional_args(parser):
    parser.add_argument('input_file', type=str, help="Provide relative path to input file")

def add_options_args(parser):
    parser.add_argument('--show-config', type=str, help="Display JSON options for module configuration")

    # Standalone module execution
    parser.add_argument('--sat-sim-only', action='store_true', help="Run the Satellite Simulator standalone. Requires a TLE file.")
    parser.add_argument('--algorithm-only', action='store_true', help="Run the Algorithm standalone. Requires an Adjacency Matrices (.am) file.")
    parser.add_argument('--fl-only', action='store_true', help="Run the Federated Learning standalone. Requires a Federated Learning Adjacency Matrices (.flam) file.")
    parser.add_argument('--model-only', action='store_true', help="Run the ML model standalone.")


# def add_general_args(parser):
#     general_group = parser.add_argument_group("general")

#     # Add args here that are not module specific, nor required in JSON
#     general_group.add_argument('--sat-sim-only', action='store_true', help="Run the Satellite Simulator standalone. Requires a TLE file.")
#     general_group.add_argument('--algorithm-only', action='store_true', help="Run the Algorithm standalone. Requires an Adjacency Matrices (.am) file.")
#     general_group.add_argument('--fl-only', action='store_true', help="Run the Federated Learning standalone. Requires a Federated Learning Adjacency Matrices (.flam) file.")
#     general_group.add_argument('--model-only', action='store_true', help="Run the ML model standalone.")

def add_sat_sim_args(parser):
    sat_sim_group = parser.add_argument_group(ModuleKey.SAT_SIM)

    # Add args here
    sat_sim_group.add_argument('--start-time', type=int, help='The start date/time for the satellite simulation')
    sat_sim_group.add_argument('--end-time', type=int, help='The end date/time for the satellite simulation')

def add_algorithm_args(parser):
    algorithm_group = parser.add_argument_group(ModuleKey.ALGORITHM)

    # Add args here

def add_fl_args(parser):
    fl_group = parser.add_argument_group(ModuleKey.FL)

    # Add args here
    fl_group.add_argument('--num-clients', type=int, help='Number of Federated Learning clients for the simulation')
