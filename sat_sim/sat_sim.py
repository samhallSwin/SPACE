"""
Filename: sat_sim.py
Author: Md Nahid Tanjum
"""

import argparse
import os
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx

from sat_sim.sat_sim_config import SatSimConfig
from sat_sim.sat_sim_handler import SatSimHandler
from sat_sim.sat_sim_output import SatSimOutput


class SatSim:
    #Main class handling operations and simulations of satellite orbits.

    def __init__(self):
        self.tle_data = None
        self.timestep = None
        self.sf_timescale = load.timescale()
        self.output = SatSimOutput()
        self.start_time = None
        self.end_time = None
        self.output_file_type = "txt"

    def set_tle_data(self, tle_data):
        #Sets TLE data for the simulation.
        self.tle_data = tle_data

    def set_timestep(self, timestep):
        #Sets the timestep for simulation updates.
        self.timestep = timedelta(minutes=timestep)

    def set_start_end_times(self, start, end):
        #Sets the start and end times for the simulation.
        self.start_time = start
        self.end_time = end


    def set_output_file_type(self, file_type):
        #Sets the output file type, either txt or csv.
        self.output_file_type = file_type

    def get_satellite_positions(self, tle_data, time):
        #Retrieve satellite positions at a given time.
        positions = {}
        for name, tle in tle_data.items():
            satellite = EarthSatellite(tle[0], tle[1], name)
            geocentric = satellite.at(time)
            positions[name] = geocentric.position.km
        return positions

    def calculate_distance(self, pos1, pos2):
        #Calculate the distance between two positions.
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def run_with_adj_matrix(self):
        #Generates adjacency matrices over the set duration and timestep.
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
            current_time += self.timestep

        # Determine output format and save
        if self.output_file_type == "txt":
            self.output.write_to_file("output.txt", matrices)
        elif self.output_file_type == "csv":
            self.output.write_to_csv("output.csv", matrices)
        return matrices  


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Satellite Visualization Over Time')
        self.setGeometry(100, 100, 1200, 900)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)
        self.simulation = SatSim()

        self.initUI()

    def initUI(self):
        self.tle_path_input = QLineEdit()
        self.tle_path_input.setPlaceholderText("Enter path to TLE file")
        self.layout.addWidget(self.tle_path_input)

        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("Enter start time (YYYY-MM-DD HH:MM:SS)")
        self.layout.addWidget(self.start_time_input)

        self.end_time_input = QLineEdit()
        self.end_time_input.setPlaceholderText("Enter end time (YYYY-MM-DD HH:MM:SS)")
        self.layout.addWidget(self.end_time_input)

        self.timestep_input = QLineEdit()
        self.timestep_input.setPlaceholderText("Enter timestep in minutes")
        self.layout.addWidget(self.timestep_input)

        btn_run_simulation = QPushButton("Run Simulation and Generate Files")
        btn_run_simulation.clicked.connect(self.run_simulation)
        self.layout.addWidget(btn_run_simulation)

        self.figure, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def run_simulation(self):
        tle_file = self.tle_path_input.text()
        start_time_str = self.start_time_input.text()
        end_time_str = self.end_time_input.text()
        timestep = int(self.timestep_input.text())

        # Read TLE file
        try:
            with open(tle_file, 'r') as f:
                tle_data = {}
                lines = f.readlines()
                for i in range(0, len(lines), 3):
                    name = lines[i].strip()
                    tle_line1 = lines[i + 1].strip()
                    tle_line2 = lines[i + 2].strip()
                    tle_data[name] = [tle_line1, tle_line2]
            self.simulation.set_tle_data(tle_data)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "TLE file not found.")
            return

        # Set simulation parameters
        try:
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid date format. Please use YYYY-MM-DD HH:MM:SS.")
            return

        self.simulation.set_start_end_times(start_time, end_time)
        self.simulation.set_timestep(timestep)

        # Run simulation
        matrices = self.simulation.run_with_adj_matrix()

        # Display results
        self.update_graphs(matrices)

    def update_graphs(self, matrices):
        keys = list(self.simulation.tle_data.keys())
        size = len(keys)

        for timestamp, adj_matrix in matrices:
            self.axes[0].cla()
            self.axes[1].cla()

            # Plot adjacency matrix
            self.axes[0].imshow(adj_matrix, cmap='Blues', interpolation='none')
            self.axes[0].set_title(f'Adjacency Matrix at {timestamp}')
            self.axes[0].set_xticks(np.arange(size))
            self.axes[0].set_yticks(np.arange(size))
            self.axes[0].set_xticklabels(keys)
            self.axes[0].set_yticklabels(keys)

            # Plot network graph
            G = nx.from_numpy_array(adj_matrix)
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=self.axes[1], with_labels=True, node_color='skyblue')
            self.axes[1].set_title(f'Network Graph at {timestamp}')

            # Draw on canvas
            self.canvas.draw()



# --- CLI Implementation ---
def parse_args():
    #Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run the satellite simulation.")
    parser.add_argument("--tle_file", required=True, help="Path to the TLE file.")
    parser.add_argument("--start", required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), help="Start time (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--end", required=True, type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), help="End time (YYYY-MM-DD HH:MM:SS).")
    parser.add_argument("--timestep", required=True, type=int, help="Timeframe in minutes between steps.")
    return parser.parse_args()

def run_cli_simulation():
    """Run the satellite simulation via CLI."""
    args = parse_args()
    simulation = SatSim()

    # Read the TLE file
    try:
        with open(args.tle_file, 'r') as f:
            tle_data = {}
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                name = lines[i].strip()
                tle_line1 = lines[i + 1].strip()
                tle_line2 = lines[i + 2].strip()
                tle_data[name] = [tle_line1, tle_line2]
    except FileNotFoundError:
        print("Error: TLE file not found.")
        return

    simulation.set_tle_data(tle_data)
    simulation.set_timestep(args.timestep)
    simulation.set_start_end_times(args.start, args.end)

    result = simulation.run_with_adj_matrix()
    print(result)


# --- MAIN SWITCH BETWEEN CLI AND GUI ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run CLI version
        run_cli_simulation()
    else:
        # Run GUI version
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
