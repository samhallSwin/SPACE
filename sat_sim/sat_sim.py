"""
Filename: sat_sim.py
Author: Md Nahid Tanjum
"""

import argparse
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import numpy as np
import os

from sat_sim.sat_sim_config import SatSimConfig
from sat_sim.sat_sim_output import SatSimOutput


class SatSim:
    # Main class handling operations and simulations of satellite orbits.

    def __init__(self):
        self.tle_data = None
        self.timestep = None
        self.sf_timescale = load.timescale()
        self.output = SatSimOutput()
        self.start_time = None
        self.end_time = None
        self.output_file_type = "txt"

    def set_tle_data(self, tle_data):
        # Sets TLE data for the simulation.
        if tle_data:
            self.tle_data = tle_data
        else:
            self.tle_data = None

    def set_timestep(self, timestep):
        # Sets the timestep for simulation updates.
        self.timestep = timedelta(minutes=timestep)

    def set_start_end_times(self, start, end):
        # Sets the start and end times for the simulation.
        self.start_time = start
        self.end_time = end

    def set_output_file_type(self, file_type):
        # Sets the output file type, either txt or csv.
        self.output_file_type = file_type

    def read_tle_file(self, file_path):
        # Reads TLE data from the specified file path.
        tle_data = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                name = lines[i].strip()
                tle_line1 = lines[i + 1].strip()
                tle_line2 = lines[i + 2].strip()
                tle_data[name] = [tle_line1, tle_line2]
        return tle_data

    def get_satellite_positions(self, time):
        # Retrieve satellite positions at a given time.
        positions = {}
        for name, tle in self.tle_data.items():
            satellite = EarthSatellite(tle[0], tle[1], name)
            geocentric = satellite.at(time)
            positions[name] = geocentric.position.km
        return positions

    def calculate_distance(self, pos1, pos2):
        # Calculate the distance between two positions.
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def generate_adjacency_matrix(self, positions, distance_threshold=10000):
        # Generates an adjacency matrix based on distances between satellites.
        keys = list(positions.keys())
        size = len(keys)
        adj_matrix = np.zeros((size, size), dtype=int)

        for i in range(size):
            for j in range(i + 1, size):
                dist = self.calculate_distance(positions[keys[i]], positions[keys[j]])
                adj_matrix[i, j] = adj_matrix[j, i] = 1 if dist < distance_threshold else 0

        return adj_matrix, keys

    def run_with_adj_matrix(self):
        # Generates adjacency matrices over the set duration and timestep.
        current_time = self.start_time
        matrices = []

        while current_time < self.end_time:
            # If current_time is a datetime, convert it to Skyfield time.
            if isinstance(current_time, datetime):
                t = self.sf_timescale.utc(current_time.year, current_time.month, current_time.day,
                                          current_time.hour, current_time.minute, current_time.second)
            else:
                t = current_time

            # Fetch satellite positions
            positions = self.get_satellite_positions(t)

            # Generate the adjacency matrix
            adj_matrix, keys = self.generate_adjacency_matrix(positions)

            # Append the result for the current timestep
            matrices.append((t.utc_strftime('%Y-%m-%d %H:%M:%S'), adj_matrix))

            # Increment the current time by the timestep
            current_time += self.timestep

        # Save the results in the specified file format
        if self.output_file_type == "txt":
            self.output.write_to_file("output.txt", matrices)
        elif self.output_file_type == "csv":
            self.output.write_to_csv("output.csv", matrices)

        return matrices


def validate_tle_file(file_path):
    # Validate that the TLE file exists and is readable
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"The file {file_path} does not exist.")
    # Ensure the file is not empty
    if os.stat(file_path).st_size == 0:
        raise argparse.ArgumentTypeError(f"The file {file_path} is empty.")
    return file_path


def validate_time(time_str):
    # Validate the date and time format
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Time '{time_str}' is not in the correct format (YYYY-MM-DD HH:MM:SS).")


def validate_timestep(timestep):
    # Validate that the timestep is a positive integer and reasonable for simulation
    if timestep <= 0 or timestep > 1440:
        raise argparse.ArgumentTypeError(f"Timestep {timestep} must be a positive integer between 1 and 1440.")
    return timestep


def parse_args():
    # Parse command line arguments with validation
    parser = argparse.ArgumentParser(description="Run the satellite simulation.")
    parser.add_argument("--tle_file", required=True, type=validate_tle_file, help="Path to the TLE file.")
    parser.add_argument("--start", required=True, type=validate_time, help="Start time (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--end", required=True, type=validate_time, help="End time (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--timestep", required=True, type=validate_timestep, help="Timeframe in minutes between steps.")

    args = parser.parse_args()

    # Validate that the start time is before the end time
    if args.start >= args.end:
        parser.error("The start time must be before the end time.")

    return args


def main():
    args = parse_args()
    simulation = SatSim()

    # Read and set TLE data
    tle_data = simulation.read_tle_file(args.tle_file)
    if not tle_data:
        print("Failed to read TLE data or TLE data is empty.")
        return

    simulation.set_tle_data(tle_data)
    simulation.set_timestep(args.timestep)
    simulation.set_start_end_times(args.start, args.end)

    # Run the simulation and get the adjacency matrices
    result = simulation.run_with_adj_matrix()

    # Print or handle the result if needed
    print("Satellite Simulation Results:")
    for timestamp, matrix in result:
        print(f"Time: {timestamp}")
        print(matrix)
        print()


if __name__ == "__main__":
    main()
