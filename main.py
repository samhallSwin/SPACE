import sys
import argparse
import json

import module_factory
from module_factory import ModuleKey
import cli_args

def set_config_file(file):
    with open('settings.json') as f:
        options = json.load(f)
        options['config_file'] = file
        json.dump(options, f, indent=4)
    
def get_config_file():
    config_file = ''
    with open('settings.json') as f:
        options = json.load(f)
        config_file = options['config_file']

    return config_file

def read_options_file():
    with open('options.json') as f:
        options = json.load(f)
        
    return options

def write_options_file(options):
    with open('options.json', 'w') as f:
        json.dump(options, f, indent=4)

def setup_parser():
    parser = argparse.ArgumentParser(description='Run FLOMPS Simulation Suite')

    subparsers = parser.add_subparsers(dest='command', help="Available commands")

    settings_parser = subparsers.add_parser('settings', help="System settings")
    cli_args.setup_settings_parser(settings_parser)

    flomps_parser = subparsers.add_parser('flomps', help="Run a FLOMPS simulation")
    cli_args.setup_flomps_parser(flomps_parser)

    return parser

def separate_args(parser, args_for_config):
    for group in parser._action_groups:
        print(f"Group: {group.title}")
        for action in group._group_actions:
            print(f" - {action.dest}")

    def get_group_arg_keys(parser, group_title):
        group = next(group for group in parser._action_groups if group.title == group_title)
        keys = [a.dest for a in group._group_actions]
        return keys
    
    options_arg_keys = get_group_arg_keys(parser, "options")
    positional_arg_keys = get_group_arg_keys(parser, "positional arguments")

    options_args = {k: v for k, v in args_for_config.items() if k in options_arg_keys}
    module_args = {k: v for k, v in args_for_config.items() if k not in options_arg_keys and k not in positional_arg_keys}

    return options_args, module_args

def check_standalone_module_flag(args):
    flags = [k[:-5] for k in args.keys() if 'only' in k]

    if len(flags) > 1:
        raise ValueError("Cannot set more than one module to run standalone")

    if len(flags) == 0:
        return None
    
    single_module = flags[0]

    # Module mapping for fl -> federated_learning, cli takes "fl" rather than "federated_learning"
    module_mapping = {
        'sat_sim': 'sat_sim',
        'algorithm': 'algorithm',
        'fl': 'federated_learning',
        'model': 'model'
    }
    # error for mapping failure
    if single_module not in module_mapping:
        raise ValueError(f"Invalid module specified for standalone operation: {single_module}")

    try:
        return ModuleKey(module_mapping[single_module])
    except ValueError:
        raise ValueError(f"Invalid module specified for standalone operation: {single_module}")

def check_args_match_module(parser, args_for_config, module_key: ModuleKey):
    group = next(g for g in parser._action_groups if g.title == module_key)

    # Get dest (names) for all actions for comparison
    group_actions_keys = []
    for action in group._group_actions:
        group_actions_keys.append(action.dest)

    mismatch = any(a not in group_actions_keys for a in args_for_config.keys())

    if mismatch:
        raise ValueError(f"Invalid command found for module: {module_key.value}")


def check_standalone_config(parser, args_for_config):
    try:
        options_args, module_args = separate_args(parser, args_for_config)
        print("OPTIONS ARGS")
        print(options_args)
        print("MODULE ARGS")
        print(module_args)

        single_module_key = check_standalone_module_flag(options_args)
        # Ensure only options for that module were selected
        if single_module_key:
            if module_args:
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
        if group.title == "positional arguments" or group.title == "options":
            continue

        # Select args pertaining to group
        print(f'ARG GROUP: {group.title}')
        group_arg_keys = [a.dest for a in group._group_actions]
        options_to_update = {k: v for k, v in args_for_config.items() if k in group_arg_keys}

        # Apply new settings to JSON options of relevant module
        options[group.title.value].update(options_to_update)

    return options

def run_standalone_module(single_module_key, input_file):
        # Temporary structure
        module = module_factory.create_single_instance(single_module_key)
        module.config.read_options(options[single_module_key.value])
        module.handler.parse_file(input_file)
        module.handler.run_module()
        output = module.output.get_result()
        print(output)

def log_options(options):
    # TODO: Make this pretty later...
    print(options)

def read_settings_cli(options, args):
    # Check if user would like to read JSON options
    # TODO: IN PROGRESS - We need a root command e.g. "run" for all simulation related stuff
    # Anything without "run" would be for meta functions, e.g. show JSON options for a particular module
    show_options_flag = args.show_options
    if show_options_flag:
        log_options(options)
        return

def read_flomps_cli(parser, args, options):
    args_for_config = {k: v for k, v in args.__dict__.items() if v is not None and v is not False}
    args_for_config.pop("command")
    print(args_for_config)

    # Check whether user wishes to run a module standalone
    single_module_key = check_standalone_config(parser, args_for_config)
    print(single_module_key)

    updated_options = update_options_with_args(parser, args_for_config, options)
    write_options_file(updated_options)

    # Get input file
    input_file = args.input_file

    # Input file ready to be used by entry module
    return single_module_key, input_file
    
def build_modules(options):
    sat_sim_module = module_factory.create_sat_sim_module()
    sat_sim_module.config.read_options(options["sat_sim"])
        
    algorithm_module = module_factory.create_algorithm_module()
    algorithm_module.config.read_options(options["algorithm"])

    fl_module = module_factory.create_fl_module()
    fl_module.config.read_options(options["federated_learning"])

    return sat_sim_module, algorithm_module, fl_module

if __name__ == "__main__":
    options = read_options_file()
    parser = setup_parser()

    # Only get arg keys with specified values
    args = parser.parse_args()

    subparsers_action = parser._subparsers._group_actions[0]
    
    # Access the 'flomps' subparser
    flomps_parser = subparsers_action.choices['flomps']
    # print(parser._subparsers._group_actions.)

    if args.command == 'settings':
        read_settings_cli(options, args)
    elif args.command == 'flomps':
        flomps_parser = subparsers_action.choices['flomps']
        single_module_key, input_file = read_flomps_cli(flomps_parser, args, options)

        # Check which module was selected
        if single_module_key is not None:
            # Run standalone module
            # print("going to run standalone module")
            run_standalone_module(single_module_key, input_file)
        else:
            # Run as simulation pipeline
            
            # Check if user would like to run an ML performance test or use existing settings
            # (Model runtime accounted for in algorithm process)

            # Create Modules
            sat_sim_module, algorithm_module, fl_module = build_modules(options)
            
            # # Simulation Process
            sat_sim_module.handler.parse_file(input_file)
            sat_sim_module.handler.run_module()
            matrices = sat_sim_module.output.get_result()
            print(matrices)
                
            algorithm_module.handler.parse_data(matrices)
            algorithm_module.handler.run_module()
            flam = algorithm_module.output.get_result()
            print(flam)

            fl_module.handler.parse_data(flam)
            fl_module.handler.run_module()

   