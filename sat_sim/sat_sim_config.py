"""
Filename: sat_sim_config.py
Author: Md Nahid Tanjum
"""

from skyfield.api import load, Topos
from datetime import datetime

class SatSimConfig:
    def __init__(self, sat_sim=None):
        # Initialize the configuration with an optional SatSim instance.
        self.ts = load.timescale()
        self.sat_sim = sat_sim
        self.options = None
        self.config_loaded = False  # Prevent recursive loading of configuration

    def set_sat_sim(self, sat_sim):
        """Associate this configuration with a specific satellite simulation instance."""
        self.sat_sim = sat_sim

    def _convert_to_timestamp(self, time_str):
        """Helper function to convert a DateTime string to a Skyfield timestamp."""
        dt_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return self.ts.utc(dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, dt_obj.minute, dt_obj.second)

    def read_options(self, options):
        """Reads and applies options to the SatSim instance."""
        # Ensure the configuration is loaded only once
        if self.config_loaded:
            return
        self.config_loaded = True  # Prevent recursive loading

        self.options = options

        # Ensure SatSim instance exists
        if not self.sat_sim:
            print("Error: SatSim instance is not set.")
            return

        # Set start and end times
        if 'start_time' in options and 'end_time' in options:
            start_time = self._convert_to_timestamp(options['start_time'])
            end_time = self._convert_to_timestamp(options['end_time'])
            self.sat_sim.set_start_end_times(start=start_time, end=end_time)

        # Set timestep if provided
        if 'timestep' in options:
            self.sat_sim.set_timestep(int(options['timestep']))  # Cast to int

        # Set ground station if provided
        if 'ground_station' in options and 'location' in options['ground_station']:
            ground_station_location = options['ground_station']['location']
            self.sat_sim.ground_station = Topos(latitude_degrees=ground_station_location['lat'],
                                                longitude_degrees=ground_station_location['long'])

        # Set output file type (txt or csv) if provided
        if 'output_file_type' in options:
            self.sat_sim.set_output_file_type(options['output_file_type'])

    def read_options_from_file(self, file_path):
        """Read and apply options from a JSON configuration file."""
        if self.config_loaded:  # Prevent recursive loading
            return
        self.config_loaded = True  # Set flag before loading to avoid recursive triggers

        try:
            import json
            with open(file_path, 'r') as file:
                options = json.load(file)
            self.read_options(options)
        except Exception as e:
            print(f"Error reading options from file: {e}")
            self.config_loaded = False  # Allow re-attempt if it fails
