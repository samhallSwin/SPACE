from skyfield.api import load, Topos

class SatSimConfig:
    """ Configuration handler for the satellite simulation environment. """
    def __init__(self, sat_sim=None):
        self.ts = load.timescale()
        self.sat_sim = sat_sim  # Core SatSim module, if applicable
        self.ground_stations = {
            "Station 1": Topos(latitude_degrees=35.0, longitude_degrees=-120.0),
            "Station 2": Topos(latitude_degrees=-33.9, longitude_degrees=18.4)
        }
        self.options = None  # Options to be configured via JSON or similar structure

    def set_sat_sim(self, sat_sim):
        """Sets the satellite simulation core module."""
        self.sat_sim = sat_sim

    def read_options(self, options):
        """Reads and applies options from a structured dictionary (simulating JSON)."""
        self.options = options
        # Apply the settings to the SatSim module if it's initialized
        if self.sat_sim:
            if 'duration' in options:
                self.sat_sim.set_duration(options['duration'])
            if 'timestep' in options:
                self.sat_sim.set_timestep(options['timestep'])

    def read_options_from_file(self, file_path):
        """Simulates reading options from a file and applying them."""
        # Example structure: could be enhanced to actually parse a JSON file
        import json
        with open(file_path, 'r') as file:
            options = json.load(file)
        self.read_options(options)
