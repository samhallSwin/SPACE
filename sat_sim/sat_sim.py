"""
Filename: sat_sim.py
Author: Md Nahid Tanjum
"""

import argparse
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import numpy as np

from sat_sim.sat_sim_config import SatSimConfig
from sat_sim.sat_sim_handler import SatSimHandler
from sat_sim.sat_sim_output import SatSimOutput

class SatSim:
    """Main class handling operations and simulations of satellite orbits."""

    def __init__(self):
        self.tle_data = None
        self.timestep = None
        self.sf_timescale = load.timescale()
        self.output = SatSimOutput()
        self.output_file_flag = True
        self.start_time = None
        self.end_time = None
        self.output_file_type = "txt"  # Default output file type

    def set_tle_data(self, tle_data):
        """Sets TLE data for the simulation."""
        self.tle_data = tle_data

    def set_timestep(self, timestep):
        """Sets the timestep for simulation updates."""
        self.timestep = timestep

    def set_output_file_flag(self, output_to_file):
        self.output_file_flag = output_to_file
    
    # CORE METHODS
    def set_start_end_times(self, start, end):
        """Sets the start and end times for the simulation."""
        self.start_time = start
        self.end_time = end


    def set_output_file_type(self, file_type):
        """Sets the output file type, either txt or csv."""
        self.output_file_type = file_type

    def get_satellite_positions(self, tle_data, time):
        """Retrieve satellite positions at a given time."""
        positions = {}
        for name, tle in tle_data.items():
            satellite = EarthSatellite(tle[0], tle[1], name)
            geocentric = satellite.at(time)
            positions[name] = geocentric.position.km
        return positions

    def calculate_distance(self, pos1, pos2):
        """Calculate the distance between two positions."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def run_with_adj_matrix(self):
        """Generates adjacency matrices over the set duration and timestep."""
        current_time = self.start_time
        matrices = []

        while current_time < self.end_time:
            positions = self.get_satellite_positions(self.tle_data, current_time)
            keys = list(positions.keys())
            size = len(keys)
            adj_matrix = np.zeros((size, size), dtype=int)

            for i in range(size):
                for j in range(i + 1, size):
                    dist = self.calculate_distance(positions[keys[i]], positions[keys[j]])
                    adj_matrix[i, j] = adj_matrix[j, i] = 1 if dist < 10000 else 0

            matrices.append((current_time.utc_strftime('%Y-%m-%d %H:%M:%S'), adj_matrix))
            current_time += timedelta(seconds=self.timestep)

        # Determine output format and save
        if self.output_file_type == "txt":
            self.output.write_to_file("output.txt", matrices)
        elif self.output_file_type == "csv":
            self.output.write_to_csv("output.csv", matrices)

        return matrices  # Return the generated matrices

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the satellite simulation.")
    parser.add_argument("--tle_file", required=True, help="Path to the TLE file.")
    parser.add_argument("--start", required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), help="Start time (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--end", required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), help="End time (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--timeframe", required=True, type=int, help="Timeframe in seconds between steps.")
    return parser.parse_args()

def main():
    args = parse_args()
    simulation = SatSim()
    handler = SatSimHandler(simulation)

    tle_data = handler.read_tle_file(args.tle_file)
    if not tle_data:
        print("Failed to read TLE data or TLE data is empty.")
        return

    simulation.set_tle_data(tle_data)
    simulation.set_timestep(args.timeframe)
    simulation.set_start_end_times(args.start, args.end)

    result = simulation.run_with_adj_matrix()

    # Print or handle the result if needed
    print(result)

if __name__ == "__main__":
    main()