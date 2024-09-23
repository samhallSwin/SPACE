"""
Filename: sat_sim_config.py
Author: Md Nahid Tanjum
"""

from skyfield.api import load, Topos
from datetime import datetime

class SatSimConfig:
    def __init__(self, sat_sim=None):
        """
        Initialize the configuration with an optional SatSim instance.
        If not provided, a new SatSim instance will be created.
        """
        self.ts = load.timescale()
        self.sat_sim = sat_sim
        self.options = None

    def set_sat_sim(self, sat_sim):
        """Associate this configuration with a specific satellite simulation instance."""
        self.sat_sim = sat_sim

    def _convert_to_timestamp(self, time_str):
        """
        Helper function to convert a DateTime string to a Skyfield timestamp.
        Accepts the time string in the format "YYYY-MM-DD HH:MM:SS".
        """
        dt_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return self.ts.utc(dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, dt_obj.minute, dt_obj.second)

    def read_options(self, options):
        """
        Apply externally provided configuration options to the satellite simulation.
        This includes start and end times, timestep, satellite count, ground station info, etc.
        """
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
            self.sat_sim.set_timestep(options['timestep'])

        # Set satellite count if provided
        if 'satellite_count' in options:
            self.sat_sim.satellite_count = options['satellite_count']

        # Set constellation type if provided
        if 'constellation_type' in options:
            self.sat_sim.constellation_type = options['constellation_type']

        # Set ground station if provided
        if 'ground_station' in options:
            ground_station_location = options['ground_station']['location']
            self.sat_sim.ground_station = Topos(latitude_degrees=ground_station_location['lat'],
                                                longitude_degrees=ground_station_location['long'])

        # Set output file type (txt or csv) if provided
        if 'output_file_type' in options:
            self.sat_sim.set_output_file_type(options['output_file_type'])

        # TODO: Check keys exist
        self.sat_sim.set_output_to_file(self.options["module_settings"]["output_to_file"])

    def read_options_from_file(self, file_path):
        """
        Read and apply options from a JSON configuration file.
        This method loads the JSON file and passes the options to the `read_options` method.
        """
        import json
        with open(file_path, 'r') as file:
            options = json.load(file)
        self.read_options(options)