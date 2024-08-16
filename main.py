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
    pass
    
    # algorithm_config, algorithm_input, algorithm_output = module_factory.create_algorithm_module()
    # algorithm_config.read_options(options["algorithm"])

    # fl_config, fl_input, fl_output = module_factory.create_fl_module()

    # Connect modules
    # Sat Sim -> Algorithm Output
    sat_sim_module = module_factory.create_sat_sim_module()
    sat_sim_module.config.read_options(options["sat_sim"])
    matrices = sat_sim_module.input.parse_input('TLEs/leoSatelliteConstellation4.tle')
    print(matrices)

    # Algorithm Output -> FL Input
    # algorithm_output.set_fl_input(fl_input)

    # fl_input.federated_learning.start_server()

    # Run Sat Sim
    

    # Pass input to Algorithm

   

