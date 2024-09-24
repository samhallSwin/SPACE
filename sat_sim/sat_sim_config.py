"""
Filename: sat_sim_config.py
Author: Nahid Tanjum

"""

import json
from datetime import datetime
from skyfield.api import load, Topos

class SatSimConfig:
    def __init__(self, sat_sim=None):
        # Initialize the configuration with an optional SatSim instance.
        self.ts = load.timescale()
        self.sat_sim = sat_sim
        self.config_loaded = False

    def _convert_to_timestamp(self, time_str):
        #Helper function to convert a DateTime string to a Skyfield timestamp.
        dt_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return self.ts.utc(dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, dt_obj.minute, dt_obj.second)

    def set_gui_enabled(self, enabled):
        #Sets whether the GUI is enabled or not.
        self.sat_sim.set_gui_enabled(enabled)

    def read_options(self, options):
        #Reads and applies options to the SatSim instance.
        if self.config_loaded:
            return

        self.config_loaded = True

        # Call set_gui_enabled() to enable or disable the GUI
        self.set_gui_enabled(options.get('gui', False))

        # Set start and end times
        if 'start_time' in options and 'end_time' in options:
            start_time = self._convert_to_timestamp(options['start_time'])
            end_time = self._convert_to_timestamp(options['end_time'])
            self.sat_sim.set_start_end_times(start=start_time, end=end_time)

        # Set timestep
        if 'timestep' in options:
            self.sat_sim.set_timestep(int(options['timestep']))

        # Set ground station if available
        if 'ground_station' in options and 'location' in options['ground_station']:
            ground_station_location = options['ground_station']['location']
            self.sat_sim.ground_station = Topos(latitude_degrees=ground_station_location['lat'],
                                                longitude_degrees=ground_station_location['long'])

        # Set output file type (txt or csv)
        if 'output_file_type' in options:
            self.sat_sim.set_output_file_type(options['output_file_type'])

    def read_options_from_file(self, file_path):
        #Read and apply options from a JSON configuration file.
        if self.config_loaded:
            return

        self.config_loaded = True
        try:
            with open(file_path, 'r') as file:
                options = json.load(file)
            self.read_options(options)
        except Exception as e:
            print(f"Error reading options from file: {e}")
            self.config_loaded = False
