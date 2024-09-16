"""
Filename: sat_sim.py
Description: Run satellite orbital simulations with TLE file data and output adjacency matrices
Author: Md Nahid Tanjum
Date: 2024-08-09
Version: 1.0
Python Version: 

Changelog:
- 2024-08-09: Initial creation. // get actual date

Usage: 

"""
import sys
import os
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QTimer
import vtk
# from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from skyfield.api import load, EarthSatellite, Topos
import networkx as nx

from sat_sim.sat_sim_output import SatSimOutput


class SatSim():

    def __init__(self):
        self.tle_data = None
        self.duration = None
        self.timestep = None
        self.sf_timescale = load.timescale()
        self.output = SatSimOutput()
        self.output_file_flag = True

    # INPUT
    def set_tle_data(self, tle_data):
        self.tle_data = tle_data
    
    # CONFIG
    def set_duration(self, duration):
        self.duration = duration

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_output_file_flag(self, output_to_file):
        self.output_file_flag = output_to_file
    
    # CORE METHODS
    def get_satellite_positions(self, tle_data, time):
        positions = {}
        for name, tle in tle_data.items():
            satellite = EarthSatellite(tle[0], tle[1], name)
            geocentric = satellite.at(time)
            positions[name] = geocentric.position.km
        return positions

    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def run_with_adj_matrix(self):
        # Setup timesteps
        try:
            duration = int(self.duration)
            timestep = int(self.timestep)
            if duration <= 0 or timestep <= 0:
                raise ValueError("Duration and timestep must be positive integers.")
        except ValueError as e:
            print(f"Invalid input: {e}")
            return
        
        t = self.sf_timescale.now()
        end_time = t + timedelta(hours=duration)
        current_time = t
        matrices = []
        
        # Generate adjacency matrices
        while current_time < end_time:
            positions = get_satellite_positions(self.tle_data, current_time)
            keys = list(positions.keys())
            size = len(keys)
            adj_matrix = np.zeros((size, size))
            for i in range(size):
                for j in range(i + 1, size):
                    dist = calculate_distance(positions[keys[i]], positions[keys[j]])
                    adj_matrix[i, j] = adj_matrix[j, i] = 1 if dist < 10000 else 0
            matrices.append((current_time.utc_strftime('%Y-%m-%d %H:%M:%S'), adj_matrix))
            current_time += timedelta(seconds=timestep)
        
        if self.output_file_flag:
            # Write to file - TO GO IN OUTPUT CLASS - get matrices there though
            self.output.write_to_file("adjacency_matrices.txt", matrices)

        # Set adjacency matrices to output
        self.output.set_result(matrices)
        # return matrices
    
    # Pass TLE data to MainWindow
    # SatSim as composite of MainWindow and OrbitManager ??


# Ground stations
GROUND_STATIONS = {
    "Station 1": Topos(latitude_degrees=35.0, longitude_degrees=-120.0),
    "Station 2": Topos(latitude_degrees=-33.9, longitude_degrees=18.4)
}

def read_tle_file(file_path):
    tle_data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            name = lines[i].strip()
            tle_line1 = lines[i + 1].strip()
            tle_line2 = lines[i + 2].strip()
            tle_data[name] = [tle_line1, tle_line2]
    return tle_data

def get_satellite_positions(tle_data, time):
    positions = {}
    for name, tle in tle_data.items():
        satellite = EarthSatellite(tle[0], tle[1], name)
        geocentric = satellite.at(time)
        positions[name] = geocentric.position.km
    return positions

def calculate_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Satellite Visualization Over Time')
        self.setGeometry(100, 100, 1200, 900)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)
        
        self.initUI()
        
        # Use an absolute path to the TLE file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tle_file_path = os.path.join(script_dir, "threeSatelliteConstellationE.tle")
        self.tle_data = read_tle_file(tle_file_path)
        
        self.ts = load.timescale()
        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # Update every second for faster testing
        self.timer.timeout.connect(self.update_orbit)
        self.timer.start()

    def initUI(self):
        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.layout.addWidget(self.vtkWidget)
        
        self.position_label = QLabel("Current Positions:")
        self.layout.addWidget(self.position_label)
        
        self.distance_label = QLabel("Distance Calculations:")
        self.layout.addWidget(self.distance_label)
        
        self.figure, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        btn_update = QPushButton("Update Orbit")
        btn_update.clicked.connect(self.update_orbit)
        self.layout.addWidget(btn_update)
        
        self.hours_input = QLineEdit()
        self.hours_input.setPlaceholderText("Enter number of hours")
        self.layout.addWidget(self.hours_input)

        self.timeframe_input = QLineEdit()
        self.timeframe_input.setPlaceholderText("Enter timeframe in seconds")
        self.layout.addWidget(self.timeframe_input)

        btn_save_adj_matrix = QPushButton("Save Adjacency Matrix for Specified Timeframe")
        btn_save_adj_matrix.clicked.connect(self.save_adj_matrix_for_specified_timeframe)
        self.layout.addWidget(btn_save_adj_matrix)

    def update_orbit(self):
        t = self.ts.now()
        positions = get_satellite_positions(self.tle_data, t)
        self.renderer.Clear()
        self.plot_orbits(positions)
        self.vtkWidget.GetRenderWindow().Render()
        self.update_positions(positions)
        self.update_distances(positions)
        self.update_graphs(positions)

    def plot_orbits(self, positions):
        for name, pos in positions.items():
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(pos[0], pos[1], pos[2])
            sphereSource.SetRadius(50)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.renderer.AddActor(actor)

    def update_positions(self, positions):
        pos_text = " | ".join([f"{name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) km" for name, pos in positions.items()])
        self.position_label.setText("Current Positions: " + pos_text)

    def update_distances(self, positions):
        distances = []
        keys = list(positions.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dist = calculate_distance(positions[keys[i]], positions[keys[j]])
                distances.append(f"{keys[i]} to {keys[j]}: {dist:.2f} km")
            for station_name, station in GROUND_STATIONS.items():
                station_pos = station.itrf_xyz().km
                dist_to_station = calculate_distance(positions[keys[i]], station_pos)
                distances.append(f"{keys[i]} to {station_name}: {dist_to_station:.2f} km")
        self.distance_label.setText("Distances: " + " | ".join(distances))

    def update_graphs(self, positions):
        keys = list(positions.keys())
        size = len(keys)
        self.adj_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                dist = calculate_distance(positions[keys[i]], positions[keys[j]])
                self.adj_matrix[i, j] = self.adj_matrix[j, i] = 1 if dist < 10000 else 0
        self.axes[0].cla()
        self.axes[1].cla()
        self.axes[0].imshow(self.adj_matrix, cmap='Blues', interpolation='none')
        self.axes[0].set_title('Adjacency Matrix')
        self.axes[0].set_xticks(np.arange(size))
        self.axes[0].set_yticks(np.arange(size))
        self.axes[0].set_xticklabels(keys)
        self.axes[0].set_yticklabels(keys)
        G = nx.from_numpy_array(self.adj_matrix)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=self.axes[1], with_labels=True, node_color='skyblue')
        self.axes[1].set_title('Network Graph')
        self.canvas.draw()
        
    def save_adj_matrix_for_specified_timeframe(self):
        try:
            hours = int(self.hours_input.text())
            timeframe = int(self.timeframe_input.text())
            if hours <= 0 or timeframe <= 0:
                raise ValueError("Hours and timeframe must be positive integers.")
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")
            return
        
        t = self.ts.now()
        end_time = t + timedelta(hours=hours)
        current_time = t
        matrices = []
        
        while current_time < end_time:
            positions = get_satellite_positions(self.tle_data, current_time)
            keys = list(positions.keys())
            size = len(keys)
            adj_matrix = np.zeros((size, size))
            for i in range(size):
                for j in range(i + 1, size):
                    dist = calculate_distance(positions[keys[i]], positions[keys[j]])
                    adj_matrix[i, j] = adj_matrix[j, i] = 1 if dist < 10000 else 0
            matrices.append((current_time.utc_strftime('%Y-%m-%d %H:%M:%S'), adj_matrix))
            current_time += timedelta(seconds=timeframe)
        
        with open("adjacency_matrices.txt", "w") as f:
            for timestamp, matrix in matrices:
                f.write(f"Time: {timestamp}\n")
                np.savetxt(f, matrix, fmt='%d')
                f.write("\n")
        
        QMessageBox.information(self, "Success", "Adjacency matrices saved to adjacency_matrices.txt")

'''
GUI Version
'''
# def main():
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     window.iren.Initialize()
#     sys.exit(app.exec_())

# if __name__ == '__main__':
#     main()
