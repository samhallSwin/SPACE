import argparse
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import numpy as np
import os
from .sat_sim_output import SatSimOutput
from .sat_sim_config import SatSimConfig  # Importing SatSimConfig
from .sat_sim_handler import SatSimHandler
from PyQt5.QtWidgets import QApplication

class SatSim:
    def __init__(self, start_time=None, end_time=None, timestep=30, output_file_type="csv", gui_enabled=False):
        self.tle_data = None
        self.timestep = timedelta(minutes=timestep)  # Set default timestep in minutes
        self.sf_timescale = load.timescale()
        self.output = SatSimOutput()
        self.start_time = start_time
        self.end_time = end_time
        self.output_file_type = output_file_type  # Default to csv output instead of txt
        self.output_path = "output"  # Default output path
        self.gui_enabled = gui_enabled  # GUI mode flag
        self.config_loaded = False  # Flag to prevent recursive loading

        # Initialize configuration without relying on JSON
        self.config = SatSimConfig(sat_sim=self)

    def set_tle_data(self, tle_data):
        # Sets TLE data for the simulation.
        self.tle_data = tle_data if tle_data else None

    def set_timestep(self, timestep):
        # Sets the timestep for simulation updates.
        self.timestep = timedelta(minutes=timestep)

    def set_start_end_times(self, start, end):
        # Sets the start and end times for the simulation, ensuring they are datetime objects.
        if isinstance(start, str):
            self.start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        else:
            self.start_time = start

        if isinstance(end, str):
            self.end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        else:
            self.end_time = end

    def set_output_file_type(self, file_type):
        # Sets the output file type, either txt or csv.
        self.output_file_type = file_type

    def run_gui(self):
        """Launch the GUI for satellite simulation."""
        from .sat_sim_gui import SatSimGUI  # Import here to avoid circular import
        import sys
        app = QApplication(sys.argv)  # Create the application instance
        gui = SatSimGUI(tle_file=None, config_file=None)  # No config file passed
        gui.show()  # Show the GUI window
        sys.exit(app.exec_())  # Start the event loop for the GUI

    def read_tle_file(self, file_path):
        # Reads TLE data from the specified file path.
        try:
            tle_data = {}
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 3):
                    name = lines[i].strip()
                    tle_line1 = lines[i + 1].strip()
                    tle_line2 = lines[i + 2].strip()
                    tle_data[name] = [tle_line1, tle_line2]
            return tle_data
        except Exception as e:
            print(f"Error reading TLE file: {e}")
            return None

    def get_satellite_positions(self, time):
        # Retrieve satellite positions at a given time.
        if isinstance(time, datetime):
            # Convert to Skyfield time
            time = self.sf_timescale.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)

        positions = {}
        for name, tle in self.tle_data.items():
            # Ensure tle and tle are strings
            tle_line1 = str(tle)
            tle_line2 = str(tle)
            satellite = EarthSatellite(tle_line1, tle_line2, name)
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
        if not self.tle_data:
            print("No TLE data loaded. Please provide valid TLE data.")
            return []

        current_time = self.start_time
        matrices = []

        while current_time < self.end_time:
            try:
                # Convert current_time to Skyfield time.
                if isinstance(current_time, datetime):
                    t = self.sf_timescale.utc(current_time.year, current_time.month, current_time.day,
                                              current_time.hour, current_time.minute, current_time.second)
                else:
                    t = current_time  # Handle Skyfield time directly

                # Fetch satellite positions
                positions = self.get_satellite_positions(t)

                if not positions:
                    print(f"Skipping timestep {current_time}: No positions calculated.")
                    current_time += self.timestep
                    continue

                # Generate the adjacency matrix
                adj_matrix, keys = self.generate_adjacency_matrix(positions)

                # Skip empty matrices
                if adj_matrix.size == 0:
                    print(f"Skipping timestep {current_time}: Empty adjacency matrix.")
                    current_time += self.timestep
                    continue

                # Append the result for the current timestep
                matrices.append((t.utc_strftime('%Y-%m-%d %H:%M:%S'), adj_matrix))

                # Increment the current time by the timestep
                current_time += self.timestep
            except Exception as e:
                print(f"Error during simulation: {e}")
                break

        # Save the results in the specified file format (default is CSV)
        output_file = os.path.join(self.output_path, f"output.{self.output_file_type}")
        if self.output_file_type == "txt":
            self.output.write_to_file(output_file, matrices)
        elif self.output_file_type == "csv":
            self.output.write_to_csv(output_file, matrices)

        return matrices

    def run(self):
        """Run the simulation, either in GUI or CLI mode."""
        if self.gui_enabled:
            self.run_gui()
        else:
            self.run_with_adj_matrix()
