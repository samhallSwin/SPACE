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
    # print(options)    # Commented by Yuganya Perumal on 24/08/2024 for testing

    # Argparse

    # 1) End-to-End Simulation

    # 2) SatSim only (can execute either from this file or run independent sat_sim.py file)

    # 3) Algorithm only (requires adjacency matrices file)

    # 4) FL only (requires parametric matrices file)

    # 5) Model only (for testing purposes, 
    #                run either MNIST or ResNet
    #                on its own without FL overhead)

    

    # Create Modules
    '''
    Commented by Yuganya Perumal on 24/08/2024 for testing
    sat_sim_module = module_factory.create_sat_sim_module()
    sat_sim_module.config.read_options(options["sat_sim"])
    '''
    algorithm_module = module_factory.create_algorithm_module()
    algorithm_module.config.read_options(options["algorithm"])
    '''
    Commented by Yuganya Perumal on 24/08/2024 for testing
    fl_module = module_factory.create_fl_module()
    fl_module.config.read_options(options["federated_learning"])
    '''
    # Simulation Process
    '''
    Commented by Yuganya Perumal on 24/08/2024 for testing
    sat_sim_module.input.parse_input('TLEs/leoSatelliteConstellation4.tle')
    matrices = sat_sim_module.input.run_module()
    print(matrices)
    '''

    # Commented by Yuganya Perumal on 24/08/2024 for testing algorithm_module.input.parse_input(matrices)
    # Process Algorithm Component Block 
    print("Number of nodes read from JSON file")
    print (algorithm_module.config.get_algorithm_no_of_nodes()) # Successfully reads the no of nodes from JSON
    print("Node Names read from JSON file")
    print (algorithm_module.config.get_algorithm_node_names()) # Successfully reads the Node Names from JSON

    # Pass Adjacency Matrices text file into Algorithm input module
    algorithm_module.input.parse_input('adjacency_matrices.txt')

    # Algorithm Output -> FL Input
    # algorithm_output.set_fl_input(fl_input)

    # fl_input.federated_learning.start_server()

    # Run Sat Sim
    
    # Pass input to Algorithm

   

