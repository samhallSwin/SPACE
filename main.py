from dataclasses import dataclass
import json

from algorithm import Algorithm
from algorithm_config import AlgorithmConfig
from algorithm_input import AlgorithmInput

from federated_learning import FederatedLearning
from fl_config import FLConfig
from fl_input import FLInput

# Instantiate relevant classes

# Instantiate Inputs, Configs and Concretes
# (Outputs get instantiated within Concretes)

def create_algorithm_module():
    algorithm = Algorithm()
    algorithm_config = AlgorithmConfig(algorithm)
    algorithm_input = AlgorithmInput(algorithm)
    
    return algorithm_config, algorithm_input

def create_fl_module():
    fl = FederatedLearning()
    fl_config = FLConfig(fl)
    fl_input = FLInput(fl)

    return fl_input, fl_config

def read_options_file():
    with open('options.json') as f:
        options = json.load(f)
        
    return options

if __name__ == "__main__":
    # Run helper function
    options = read_options_file()
    
    algorithm_config, algorithm_input = create_algorithm_module()
    algorithm_config.read_options(options["algorithm"])

    fl_config, fl_input = create_fl_module()

    # fl_input.federated_learning.start_server()

    # Run Sat Sim

    # Pass input to Algorithm

   

