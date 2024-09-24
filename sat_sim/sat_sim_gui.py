"""
Filename: sat_sim_gui.py
Author: Nahid Tanjum

"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QFileDialog, QSlider
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import networkx as nx
from skyfield.api import load, utc
from datetime import datetime, timedelta, timezone
from .sat_sim import SatSim

class SatSimGUI(QMainWindow):
    gui_initialized = False

    def __init__(self, tle_file=None):
        super().__init__()

        if SatSimGUI.gui_initialized:
            return 

        self.setWindowTitle('Satellite Visualization Over Time')
        self.setGeometry(100, 100, 1200, 900)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)
        self.simulation = SatSim()
        self.tle_file = tle_file

        self.start_time = None
        self.end_time = None

        self.initUI()

        SatSimGUI.gui_initialized = True

    def initUI(self):
        #Sets up the user interface including buttons, input fields, and labels.
        # Setup VTK widget for 3D visualization
        self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.layout.addWidget(self.vtkWidget)

        # Labels to show satellite positions and distances
        self.position_label = QLabel("Current Positions:")
        self.layout.addWidget(self.position_label)

        self.distance_label = QLabel("Distance Calculations:")
        self.layout.addWidget(self.distance_label)

        # Matplotlib for adjacency matrix and network graph
        self.figure, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Input fields for user-defined TLE file, start and end times, and timestep
        self.tle_file_input = QLineEdit()
        self.tle_file_input.setPlaceholderText("Select TLE file")
        self.layout.addWidget(self.tle_file_input)

        btn_tle_file = QPushButton("Browse TLE File")
        btn_tle_file.clicked.connect(self.browse_tle_file)
        self.layout.addWidget(btn_tle_file)

        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("Enter start time (YYYY-MM-DD HH:MM:SS)")
        self.layout.addWidget(self.start_time_input)

        self.end_time_input = QLineEdit()
        self.end_time_input.setPlaceholderText("Enter end time (YYYY-MM-DD HH:MM:SS)")
        self.layout.addWidget(self.end_time_input)

        self.timeframe_input = QLineEdit()
        self.timeframe_input.setPlaceholderText("Enter timestep in minutes")
        self.layout.addWidget(self.timeframe_input)

        # Slider to control time between start and end time
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.valueChanged.connect(self.on_slider_change)
        self.layout.addWidget(self.time_slider)

        # Button to trigger orbit update
        btn_update = QPushButton("Update Orbit")
        btn_update.clicked.connect(self.update_orbit)
        self.layout.addWidget(btn_update)

        # Button to save adjacency matrix and network graphs for the specified timeframe
        btn_save_adj_matrix = QPushButton("Save Adjacency Matrix & Network Graphs for Specified Timeframe")
        btn_save_adj_matrix.clicked.connect(self.save_adj_matrix_for_specified_timeframe)
        self.layout.addWidget(btn_save_adj_matrix)

    def browse_tle_file(self):
        #Opens a file dialog to select the TLE file."""
        tle_file, _ = QFileDialog.getOpenFileName(self, "Select TLE File", "", "TLE Files (*.tle);;All Files (*)")
        if tle_file:
            self.tle_file_input.setText(tle_file)
            tle_data = self.simulation.read_tle_file(tle_file)
            self.simulation.set_tle_data(tle_data)

    def on_slider_change(self, value):
        #Handle the slider change event to update visualization time.
        if self.start_time is not None and self.end_time is not None:
            # Calculate time based on slider value
            slider_time = self.start_time + (value * self.simulation.timestep)
            self.update_visualization(slider_time.utc_datetime())

    def update_orbit(self):
        #Fetches satellite positions, updates the visualization, and recalculates distances.
        try:
            if self.simulation.tle_data is None:
                QMessageBox.warning(self, "TLE Data Missing", "Please load a TLE file first.")
                return

            start_time_str = self.start_time_input.text()
            end_time_str = self.end_time_input.text()
            timestep_str = self.timeframe_input.text()

            if not start_time_str or not end_time_str or not timestep_str:
                QMessageBox.warning(self, "Input Missing", "Please provide start time, end time, and timestep.")
                return

            # Ensure the times are passed as Skyfield Time objects
            ts = load.timescale()
            start_time = ts.utc(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc))
            end_time = ts.utc(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc))
            timestep = int(timestep_str)

            self.simulation.set_timestep(timestep)
            self.simulation.set_start_end_times(start_time, end_time)

            self.start_time = start_time
            self.end_time = end_time

            total_steps = int((end_time.utc_datetime() - start_time.utc_datetime()) / timedelta(minutes=timestep))
            self.time_slider.setMaximum(total_steps)

            # Update the visualization for the first time
            self.update_visualization(start_time.utc_datetime())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def update_visualization(self, time):
        #Updates satellite positions, distances, adjacency matrix, and network graph based on the given time.S
        positions = self.simulation.get_satellite_positions(time)

        if not positions:
            QMessageBox.warning(self, "No Positions", "No satellite positions were calculated.")
            return

        self.plot_orbits(positions)
        self.vtkWidget.GetRenderWindow().Render()

        self.update_positions(positions)
        self.update_distances(positions)
        self.update_graphs(positions)

        self.position_label.setText(f"Visualization Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def plot_orbits(self, positions):
        #Plots the satellites as spheres at their current positions.
        self.renderer.Clear()

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
        #Updates the position label with the current positions of the satellites.
        pos_text = " | ".join([f"{name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) km" for name, pos in positions.items()])
        self.position_label.setText("Current Positions: " + pos_text)

    def update_distances(self, positions):
        #Updates the distance calculations between satellites.
        distances = []
        keys = list(positions.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dist = self.simulation.calculate_distance(positions[keys[i]], positions[keys[j]])
                distances.append(f"{keys[i]} to {keys[j]}: {dist:.2f} km")

        self.distance_label.setText("Distance Calculations: " + " | ".join(distances))

    def update_graphs(self, positions):
        #Generates and updates the adjacency matrix and network graph.
        adj_matrix, keys = self.simulation.generate_adjacency_matrix(positions)

        self.axes[0].cla()
        self.axes[1].cla()

        self.axes[0].imshow(adj_matrix, cmap='Blues', interpolation='none')
        self.axes[0].set_title('Adjacency Matrix')
        self.axes[0].set_xticks(np.arange(len(keys)))
        self.axes[0].set_yticks(np.arange(len(keys)))
        self.axes[0].set_xticklabels(keys)
        self.axes[0].set_yticklabels(keys)

        G = nx.from_numpy_array(adj_matrix)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=self.axes[1], with_labels=True, node_color='skyblue')
        self.axes[1].set_title('Network Graph')

        self.canvas.draw()

    def save_adj_matrix_for_specified_timeframe(self):
        #Saves adjacency matrices and network graphs for a user-specified timeframe.
        try:
            output_file, _ = QFileDialog.getSaveFileName(self, "Save Adjacency Matrix", "", "Text Files (*.txt);;All Files (*)")
            pdf_file, _ = QFileDialog.getSaveFileName(self, "Save Network Graphs PDF", "", "PDF Files (*.pdf);;All Files (*)")

            if output_file and pdf_file:
                # Loop over the range of the timeframe and save the matrices and graphs
                matrices = self.simulation.run_with_adj_matrix()
                if not matrices:
                    QMessageBox.warning(self, "No Data", "No data was generated to save.")
                    return

                with open(output_file, 'w') as f:
                    for timestamp, matrix in matrices:
                        formatted_timestamp = timestamp if isinstance(timestamp, str) else timestamp.utc_iso()
                        f.write(f"Time: {formatted_timestamp}\n")
                        np.savetxt(f, matrix, fmt='%d')
                        f.write("\n")

                with PdfPages(pdf_file) as pdf:
                    for timestamp, matrix in matrices:
                        formatted_timestamp = timestamp if isinstance(timestamp, str) else timestamp.utc_iso()
                        G = nx.from_numpy_array(matrix)
                        pos = nx.spring_layout(G)
                        plt.figure()
                        nx.draw(G, pos, with_labels=True, node_color='skyblue')
                        plt.title(f"Network Graph at {formatted_timestamp}")
                        pdf.savefig()
                        plt.close()

                QMessageBox.information(self, "Success", "Adjacency matrix and network graphs saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving the matrix: {e}")

    def show_gui(self):
        #Display the GUI.
        self.show()
        self.iren.Initialize()
