from module_factory import ModuleKey

def setup_settings_parser(parser):
    parser.add_argument('--options-file', type=str, help="Set JSON Options File")
    parser.add_argument('--show-options', action='store_true', help="Display JSON options for module configuration")

def add_sat_sim_args(parser):
        print("Adding SAT_SIM args")
        sat_sim_group = parser.add_argument_group(ModuleKey.SAT_SIM)

        # Add args here
        sat_sim_group.add_argument('--start-time', type=str, help='The start date/time for the satellite simulation')
        sat_sim_group.add_argument('--end-time', type=str, help='The end date/time for the satellite simulation')
        sat_sim_group.add_argument('--timestep', type=int, help='Timestep for the simulation')
        sat_sim_group.add_argument('--gui', action='store_true', help='Enable GUI mode for satellite simulation')

def add_algorithm_args(parser):
    print("Adding ALGORITHM args")
    algorithm_group = parser.add_argument_group(ModuleKey.ALGORITHM)

    # Add args here

def add_fl_args(parser):
    print("Adding FL args")
    fl_group = parser.add_argument_group(ModuleKey.FL)

    # Add args here
    fl_group.add_argument('--num-rounds', type=int, help='Number of Federated Learning rounds for the simulation')
    fl_group.add_argument('--num-clients', type=int, help='Number of Federated Learning clients for the simulation')
    fl_group.add_argument('--model-type', type=str, help='FL Model Type (currently 2 options): ResNet50, SimpleCNN')
    fl_group.add_argument('--data-set', type=str, help='FL Data Set (currently 1 option): MNIST')

def setup_flomps_parser(parser):
    def add_positional_args(parser):
        parser.add_argument('input_file', type=str, help="Provide relative path to input file")

    def add_options_args(parser):
        # Standalone module execution
        print("Adding optional args")
        # options_group = parser.add_argument_group("optional")

        parser.add_argument('--sat-sim-only', action='store_true', help="Run the Satellite Simulator standalone. Requires a TLE file.")
        parser.add_argument('--algorithm-only', action='store_true', help="Run the Algorithm standalone. Requires an Adjacency Matrices (.am) file.")
        parser.add_argument('--fl-only', action='store_true', help="Run the Federated Learning standalone. Requires a Federated Learning Adjacency Matrices (.flam) file.")
        parser.add_argument('--model-only', action='store_true', help="Run the ML model standalone.")

    add_positional_args(parser)
    add_options_args(parser)
    add_sat_sim_args(parser)
    add_algorithm_args(parser)
    add_fl_args(parser)

    

# def create_subparsers(parser):
#     subparsers = parser.add_subparsers(dest='command', help="Available commands")

#     # Create subparsers
#     flomps_parser = subparsers.add_parser('flomps', help="Run a FLOMPS simulation")

#     add_positional_args(flomps_parser)
#     add_options_args(flomps_parser)

#     add_sat_sim_args(flomps_parser)
#     add_algorithm_args(flomps_parser)
#     add_fl_args(flomps_parser)