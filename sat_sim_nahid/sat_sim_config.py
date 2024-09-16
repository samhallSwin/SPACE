"""
Filename: sat_sim_config.py
Author: Md Nahid Tanjum
"""

from skyfield.api import load, Topos

class SatSimConfig:
    def __init__(self, sat_sim=None):
        self.ts = load.timescale()
        self.sat_sim = sat_sim
        self.options = None

    def set_sat_sim(self, sat_sim):
        """Associate this configuration with a specific satellite simulation instance."""
        self.sat_sim = sat_sim

    def _convert_to_timestamp(self, hours):
        """Helper function to convert hours to a timestamp."""
        from datetime import datetime, timedelta
        current_time = datetime.now()
        return self.ts.utc(current_time.year, current_time.month, current_time.day, current_time.hour + hours)

    def read_options(self, options):
        """Apply externally provided configuration options."""
        self.options = options
        if self.sat_sim:
            if 'start_time' in options:
                start_time = self._convert_to_timestamp(options['start_time'])
                self.sat_sim.set_start_end_times(start=start_time)
            if 'end_time' in options:
                end_time = self._convert_to_timestamp(options['end_time'])
                self.sat_sim.set_start_end_times(end=end_time)
            if 'duration' in options:
                self.sat_sim.set_duration(options['duration'])
            if 'timestep' in options:
                self.sat_sim.set_timestep(options['timestep'])

            # Handling satellite count, constellation type, and ground station (if needed later)
            if 'satellite_count' in options:
                self.sat_sim.satellite_count = options['satellite_count']  # Example if satellite count is needed
            if 'constellation_type' in options:
                self.sat_sim.constellation_type = options['constellation_type']  # Store the constellation type if required
            if 'ground_station' in options:
                ground_station_location = options['ground_station']['location']
                self.sat_sim.ground_station = Topos(latitude_degrees=ground_station_location['lat'],
                                                    longitude_degrees=ground_station_location['long'])

    def read_options_from_file(self, file_path):
        """Read and apply options from a JSON configuration file."""
        import json
        with open(file_path, 'r') as file:
            options = json.load(file)
        self.read_options(options)
