import argparse
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from .sat_sim_config import SatSimConfig
from .sat_sim_handler import SatSimHandler
from .sat_sim_output import SatSimOutput

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
        self.tle_data = tle_data

    def set_timestep(self, timestep):
        self.timestep = timedelta(minutes=timestep)

    def set_start_end_times(self, start, end):
        self.start_time = start
        self.end_time = end

    def set_output_file_type(self, file_type):
        self.output_file_type = file_type

    def get_satellite_positions(self, tle_data, current_time):

        try:
            sky_time = current_time
            _ = sky_time.tt

        except AttributeError:
            # Fallback if current_time is not a Skyfield Time object, convert from datetime
            sky_time = self.sf_timescale.utc(current_time.year, current_time.month, current_time.day,
                                         current_time.hour, current_time.minute, current_time.second)
        positions = {}
        for name, tle in tle_data.items():
            satellite = EarthSatellite(tle[0], tle[1], name)
            geocentric = satellite.at(sky_time)
            positions[name] = geocentric.position.km
        return positions

    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def run_with_adj_matrix(self):
        current_time = self.start_time
        matrices = []

        while current_time < self.end_time:
            positions = self.get_satellite_positions(self.tle_data, current_time)
            if positions:
                keys = list(positions.keys())
                size = len(keys)
                adj_matrix = np.zeros((size, size), dtype=int)

            for i in range(size):
                for j in range(i + 1, size):
                    dist = self.calculate_distance(positions[keys[i]], positions[keys[j]])
                    adj_matrix[i, j] = adj_matrix[j, i] = 1 if dist < 10000 else 0

            # Handle current_time formatting based on its type
            if hasattr(current_time, 'utc_strftime'):  # Skyfield Time object
                formatted_time = current_time.utc_strftime('%Y-%m-%d %H:%M:%S')
            else:  # Python datetime object
                formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

            matrices.append((current_time.utc_strftime('%Y-%m-%d %H:%M:%S'), adj_matrix))
            current_time += self.timestep

        if not matrices:
            print("No matrices generated. Exiting.")  # Handle no data generated
            return []
        
        if matrices:
            max_size = max(len(matrix) for _, matrix in matrices)  # Safe to call max
            print("Max matrix size:", max_size)

        # Determine output format and save
        if self.output_file_type == "txt":
            self.output.write_to_file("output.txt", matrices)
        elif self.output_file_type == "csv":
            self.output.write_to_csv("output.csv", matrices)
        return matrices

# GUI class
class SatSimGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Satellite Simulation")
        self.geometry("500x400")
        self.simulation = SatSim()
        # Create input fields
        self.create_widgets()

    def create_widgets(self):
        # TLE File
        self.tle_label = tk.Label(self, text="TLE File:")
        self.tle_label.pack()
        self.tle_entry = tk.Entry(self, width=50)
        self.tle_entry.pack()
        self.tle_button = tk.Button(self, text="Browse", command=self.browse_tle_file)
        self.tle_button.pack()

        # Start Time
        self.start_label = tk.Label(self, text="Start Time (YYYY-MM-DD HH:MM:SS):")
        self.start_label.pack()
        self.start_entry = tk.Entry(self, width=50)
        self.start_entry.pack()

        # End Time
        self.end_label = tk.Label(self, text="End Time (YYYY-MM-DD HH:MM:SS):")
        self.end_label.pack()
        self.end_entry = tk.Entry(self, width=50)
        self.end_entry.pack()

        # Timestep
        self.timestep_label = tk.Label(self, text="Timestep (in minutes):")
        self.timestep_label.pack()
        self.timestep_entry = tk.Entry(self, width=50)
        self.timestep_entry.pack()

        # Run button
        self.run_button = tk.Button(self, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack()

        # Output display
        self.output_text = tk.Text(self, height=10, width=60)
        self.output_text.pack()

    def browse_tle_file(self):
        tle_file = filedialog.askopenfilename(filetypes=[("TLE files", "*.tle"), ("All files", "*.*")])
        self.tle_entry.delete(0, tk.END)
        self.tle_entry.insert(0, tle_file)

    def run_simulation(self):
        tle_file = self.tle_entry.get()
        start_time_str = self.start_entry.get()
        end_time_str = self.end_entry.get()
        timestep_str = self.timestep_entry.get()

        start_time = None
        end_time = None

        if not tle_file or not start_time or not end_time or not timestep:
            messagebox.showerror("Error", "All fields must be filled.")
            return

        # Validation and parsing
        try:
            timestep = int(timestep)
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            start_time = self.simulation.sf_timescale.utc(start_time.year, start_time.month, start_time.day,
                                                      start_time.hour, start_time.minute, start_time.second)
            end_time = self.simulation.sf_timescale.utc(end_time.year, end_time.month, end_time.day,
                                                    end_time.hour, end_time.minute, end_time.second)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        self.simulation.set_output_file_type("csv")
        tle_data = SatSimHandler(self.simulation).read_tle_file(tle_file)

        self.simulation.set_tle_data(SatSimHandler(self.simulation).read_tle_file(tle_file))
        self.simulation.set_timestep(timestep)
        self.simulation.set_start_end_times(start_time, end_time)

        result = self.simulation.run_with_adj_matrix()

        if result:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Simulation completed successfully. Check the CSV file.")
        else:
            messagebox.showerror("Error", "No data generated from simulation. Check input parameters and TLE data.")


def main():
    # Launch the GUI
    app = SatSimGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
