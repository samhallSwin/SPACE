from dataclasses import dataclass
import json

import module_factory

def read_options_file():
    with open('options.json') as f:
        options = json.load(f)
        
    return options

if __name__ == "__main__":
    # Run helper function
    options = read_options_file()
    print(options)    

    # Argparse

    # 1) End-to-End Simulation

    # 2) SatSim only (can execute either from this file or run independent sat_sim.py file)

    # 3) Algorithm only (requires adjacency matrices file)

    # 4) FL only (requires parametric matrices file)

    # 5) Model only (for testing purposes, 
    #                run either MNIST or ResNet
    #                on its own without FL overhead)

    

    # Create Modules
    sat_sim_module = module_factory.create_sat_sim_module()
    sat_sim_module.config.read_options(options["sat_sim"])

    algorithm_module = module_factory.create_algorithm_module()
    algorithm_module.config.read_options(options["algorithm"])

    fl_module = module_factory.create_fl_module()
    fl_module.config.read_options(options["federated_learning"])

    # Simulation Process
    sat_sim_module.input.parse_input('TLEs/leoSatelliteConstellation4.tle')
    matrices = sat_sim_module.input.run_module()
    print(matrices)

    algorithm_module.input.parse_input(matrices)

    # Algorithm Output -> FL Input
    # algorithm_output.set_fl_input(fl_input)

    # fl_input.federated_learning.start_server()

    # Run Sat Sim
    

    # Pass input to Algorithm

   

