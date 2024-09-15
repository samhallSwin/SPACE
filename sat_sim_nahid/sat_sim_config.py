"""
Filename: sat_sim_config.py
Author: Md Nahid Tanjum"""

from skyfield.api import load, Topos

class SatSimConfig:
    def __init__(self, sat_sim=None):
        self.ts = load.timescale()
        self.sat_sim = sat_sim
        # Commenting out the ground stations as per your request.
        # self.ground_stations = {
        #     "Station 1": Topos(latitude_degrees=35.0, longitude_degrees=-120.0),
        #     "Station 2": Topos(latitude_degrees=-33.9, longitude_degrees=18.4)
        # }
        self.options = None

    def set_sat_sim(self, sat_sim):
        """Associate this configuration with a specific satellite simulation instance."""
        self.sat_sim = sat_sim

    def read_options(self, options):
        """Apply externally provided configuration options."""
        self.options = options
        if self.sat_sim:
            if 'start_time' in options:
                self.sat_sim.set_start_time(options['start_time'])
            if 'end_time' in options:
                self.sat_sim.set_end_time(options['end_time'])
            if 'duration' in options:
                self.sat_sim.set_duration(options['duration'])
            if 'timestep' in options:
                self.sat_sim.set_timestep(options['timestep'])

    def read_options_from_file(self, file_path):
        """Read and apply options from a JSON configuration file."""
        import json
        with open(file_path, 'r') as file:
            options = json.load(file)
        self.read_options(options)
