"""
Filename: sat_sim_gui.py
Author: Md Nahid Tanjum
"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import networkx as nx
from datetime import datetime, timedelta
from .sat_sim import SatSim
from skyfield.api import Topos


class SatSimGUI(QMainWindow):
    def __init__(self, tle_file=None):
        super().__init__()
        self.setWindowTitle('Satellite Visualization Over Time')
        self.setGeometry(100, 100, 1200, 900)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)

        # Initialize the SatSim instance
        self.simulation = SatSim()
        self.initUI()

        if tle_file:
            tle_data = self.simulation.read_tle_file(tle_file)
            self.simulation.set_tle_data(tle_data)

        # Timer to update satellite positions periodically
        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # Update every second (can be adjusted)
        self.timer.timeout.connect(self.update_orbit)
        self.timer.start()

    def initUI(self):
        """Sets up the user interface including buttons, input fields, and labels."""
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

        # Button to trigger orbit update
        btn_update = QPushButton("Update Orbit")
        btn_update.clicked.connect(self.update_orbit)
        self.layout.addWidget(btn_update)

        # Button to save adjacency matrix and network graphs for the specified timeframe
        btn_save_adj_matrix = QPushButton("Save Adjacency Matrix & Network Graphs for Specified Timeframe")
        btn_save_adj_matrix.clicked.connect(self.save_adj_matrix_for_specified_timeframe)
        self.layout.addWidget(btn_save_adj_matrix)

    def browse_tle_file(self):
        """Opens a file dialog to select the TLE file."""
        tle_file, _ = QFileDialog.getOpenFileName(self, "Select TLE File", "", "TLE Files (*.tle);;All Files (*)")
        if tle_file:
            self.tle_file_input.setText(tle_file)
            tle_data = self.simulation.read_tle_file(tle_file)
            self.simulation.set_tle_data(tle_data)

    def update_orbit(self):
        """Fetches satellite positions, updates the visualization, and recalculates distances."""
        try:
            # Ensure the TLE data is loaded
            if self.simulation.tle_data is None:
                QMessageBox.warning(self, "TLE Data Missing", "Please load a TLE file first.")
                return

            # Convert timestep to minutes (from GUI input)
            if self.timeframe_input.text():
                try:
                    timestep = int(self.timeframe_input.text())
                    self.simulation.set_timestep(timestep)  # Set timestep in minutes
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Timestep must be a valid number.")
                    return

            # Update simulation with current time
            t = self.simulation.sf_timescale.now()  # Use Skyfield timescale
            positions = self.simulation.get_satellite_positions(t)  # Get satellite positions

            # Clear previous visualization
            self.renderer.Clear()

            # Plot the orbits and update the display
            self.plot_orbits(positions)
            self.vtkWidget.GetRenderWindow().Render()

            # Update positions, distances, and graphs in the UI
            self.update_positions(positions)
            self.update_distances(positions)  # This will now show distances
            self.update_graphs(positions)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def plot_orbits(self, positions):
        """Plots the satellites as spheres at their current positions."""
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
        """Updates the position label with the current positions of the satellites."""
        pos_text = " | ".join([f"{name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) km" for name, pos in positions.items()])
        self.position_label.setText("Current Positions: " + pos_text)

    def update_distances(self, positions):
        """Updates the distance calculations between satellites."""
        distances = []
        keys = list(positions.keys())

        # Calculate distances between satellites
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dist = self.simulation.calculate_distance(positions[keys[i]], positions[keys[j]])
                distances.append(f"{keys[i]} to {keys[j]}: {dist:.2f} km")

        # Update the distance label with distances
        self.distance_label.setText("Distance Calculations: " + " | ".join(distances))

    def update_graphs(self, positions):
        """Generates and updates the adjacency matrix and network graph."""
        adj_matrix, keys = self.simulation.generate_adjacency_matrix(positions)

        # Clear previous plots
        self.axes[0].cla()
        self.axes[1].cla()

        # Plot adjacency matrix
        self.axes[0].imshow(adj_matrix, cmap='Blues', interpolation='none')
        self.axes[0].set_title('Adjacency Matrix')
        self.axes[0].set_xticks(np.arange(len(keys)))
        self.axes[0].set_yticks(np.arange(len(keys)))
        self.axes[0].set_xticklabels(keys)
        self.axes[0].set_yticklabels(keys)

        # Plot network graph
        G = nx.from_numpy_array(adj_matrix)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=self.axes[1], with_labels=True, node_color='skyblue')
        self.axes[1].set_title('Network Graph')

        # Refresh the canvas
        self.canvas.draw()

    def save_adj_matrix_for_specified_timeframe(self):
        """Saves adjacency matrices and network graphs for a user-specified timeframe."""
        try:
            # Ensure start, end times, and timestep are provided
            start_time_str = self.start_time_input.text()
            end_time_str = self.end_time_input.text()
            timestep_str = self.timeframe_input.text()

            if not start_time_str or not end_time_str or not timestep_str:
                QMessageBox.warning(self, "Input Missing", "Please provide start time, end time, and timestep.")
                return

            # Convert inputs to appropriate data types
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
            timestep = int(timestep_str)

            # Set up simulation parameters
            self.simulation.set_start_end_times(start_time, end_time)
            self.simulation.set_timestep(timestep)

            # Run the simulation and get the adjacency matrices
            result = self.simulation.run_with_adj_matrix()

            # Prompt the user to choose where to save the output file
            output_file, _ = QFileDialog.getSaveFileName(self, "Save Adjacency Matrix", "", "Text Files (*.txt);;All Files (*)")
            pdf_file, _ = QFileDialog.getSaveFileName(self, "Save Network Graphs PDF", "", "PDF Files (*.pdf);;All Files (*)")

            if output_file and pdf_file:
                # Write the result to a file
                with open(output_file, 'w') as f:
                    for timestamp, matrix in result:
                        f.write(f"Time: {timestamp}\n")
                        np.savetxt(f, matrix, fmt='%d')
                        f.write("\n")

                # Save network graphs to a PDF
                with PdfPages(pdf_file) as pdf:
                    for timestamp, matrix in result:
                        G = nx.from_numpy_array(matrix)
                        pos = nx.spring_layout(G)
                        plt.figure()
                        nx.draw(G, pos, with_labels=True, node_color='skyblue')
                        plt.title(f"Network Graph at {timestamp}")
                        pdf.savefig()
                        plt.close()

                QMessageBox.information(self, "Success", "Adjacency matrix and network graphs saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving the matrix: {e}")


def main(tle_file=None):
    """Main entry point for the GUI."""
    app = QApplication(sys.argv)
    window = SatSimGUI(tle_file=tle_file)
    window.show()
    window.iren.Initialize()  # For VTK rendering
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
