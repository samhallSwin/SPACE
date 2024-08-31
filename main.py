import sys
import argparse
import json

import module_factory
from module_factory import ModuleKey
import cli_args

def read_options_file():
    with open('options.json') as f:
        options = json.load(f)
        
    return options

def write_options_file(options):
    with open('options.json', 'w') as f:
        json.dump(options, f, indent=4)

def setup_parser():
    parser = argparse.ArgumentParser(description='Run FLOMPS Simulation Suite')

    # Add general args to parser group
    cli_args.add_general_args(parser)

    # Add args to respective parser groups for each module, such that group titles match JSON module keys for configuration
    cli_args.add_sat_sim_args(parser)
    cli_args.add_algorithm_args(parser)
    cli_args.add_fl_args(parser)

    return parser

def separate_args(parser, args_for_config):
    general_group = next(group for group in parser._action_groups if group.title == "general")
    general_arg_keys = [a.dest for a in general_group._group_actions]
    general_args = {k: v for k, v in args_for_config.items() if k in general_arg_keys}
    module_args = {k: v for k, v in args_for_config.items() if k not in general_arg_keys}

    return general_args, module_args

def check_standalone_module_flag(args):
    flags = [k[:-5] for k in args.keys() if 'only' in k]

    if len(flags) > 1:
        raise ValueError("Cannot set more than one module to run standalone")

    if len(flags) == 0:
        return None
    
    single_module = flags[0]

    try:
        return ModuleKey(single_module)
    except ValueError:
        raise ValueError(f"Invalid module specified for standalone operation: {single_module}")

def check_args_match_module(parser, args_for_config, module_key: ModuleKey):
    print(module_key.value)
    group = next(g for g in parser._action_groups if g.title == module_key)

    mismatch = any(a not in group._group_actions for a in args_for_config)

    if mismatch:
        raise ValueError(f"Invalid command found for module: {module_key.value}")


def check_standalone_config(parser, args_for_config):
    try:
        general_args, module_args = separate_args(parser, args_for_config)
        single_module_key = check_standalone_module_flag(general_args)
        # Ensure only options for that module were selected
        if single_module_key:
            check_args_match_module(parser, module_args, single_module_key)
            return single_module_key
        else:
            return None
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        

def update_options_with_args(parser, args_for_config, options):
    # Apply incoming args to update options.json
    for group in parser._action_groups:
        # Skip default Argparser groups
        if group.title == "positional arguments" or group.title == "options" or group.title == "general":
            continue

        # Select args pertaining to group
        print(f'ARG GROUP: {group.title}')
        group_arg_keys = [a.dest for a in group._group_actions]
        options_to_update = {k: v for k, v in args_for_config.items() if k in group_arg_keys}

        # Apply new settings to JSON options of relevant module
        options[group.title.value].update(options_to_update)

    return options

def read_cli(options):
    parser = setup_parser()

    # Only get arg keys with specified values
    args = parser.parse_args()

    args_for_config = {k: v for k, v in args.__dict__.items() if v is not None and v is not False}
    print(args_for_config)

    # Check whether user wishes to run a module standalone
    single_module_key = check_standalone_config(parser, args_for_config)

    updated_options = update_options_with_args(parser, args_for_config, options)
    write_options_file(updated_options)

    # Check which module was selected
    if single_module_key:
        print("made it here!")
        module = module_factory.create_single_instance(single_module_key)
        module.config.read_options(options[single_module_key.value])
        module.handler.parse_input('TLEs/leoSatelliteConstellation4.tle')
        matrices = module.handler.run_module()
        print(matrices)
    else:
        # Run as simulation pipeline
        pass
    

if __name__ == "__main__":
    options = read_options_file()
    read_cli(options)

    # Argparse

    # 1) End-to-End Simulation

    # 2) SatSim only (can execute either from this file or run independent sat_sim.py file)

    # 3) Algorithm only (requires adjacency matrices file)

    # 4) FL only (requires parametric matrices file)

    # 5) Model only (for testing purposes, 
    #                run either MNIST or ResNet
    #                on its own without FL overhead)

    # Create Modules
    # sat_sim_module = module_factory.create_sat_sim_module()
    # sat_sim_module.config.read_options(options["sat_sim"])
    

    # algorithm_module = module_factory.create_algorithm_module()
    # algorithm_module.config.read_options(options["algorithm"])

    # fl_module = module_factory.create_fl_module()
    # fl_module.config.read_options(options["federated_learning"])

    # Simulation Process
    # sat_sim_module.handler.parse_input('TLEs/leoSatelliteConstellation4.tle')
    # matrices = sat_sim_module.handler.run_module()
    # print(matrices)

    # algorithm_module.handler.parse_input(matrices)

    # Algorithm Output -> FL Input
    # algorithm_output.set_fl_input(fl_handler)

    # fl_handler.federated_learning.start_server()

    # Run Sat Sim
    

    # Pass input to Algorithm

   

