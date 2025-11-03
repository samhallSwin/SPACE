"""
Filename: GraphDisplay.py
Author: Luke Magnuson, Nikola Markoski
Description: This script derives from collapsibleOverlay, and serves as a display for the connection and adjacency graphs.
These graphs are drawn from the Sat_Sim instance from user supplied data.
"""

from PyQt5.QtWidgets import QFileDialog, QSizePolicy, QPushButton, QVBoxLayout, QWidget, QMessageBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import networkx as nx

from Components.CollapsibleOverlay import CollapsibleOverlay, OverlaySide

class GraphDisplay(CollapsibleOverlay):

    def __init__(self, parent=None, side= OverlaySide.RIGHT):
        self.backend = parent.backend

        super().__init__(parent,side)

        self.control_panel_height_reduction = 260
        self.set_up_adjacency_graph()
        self.set_up_connection_graph()

        self.output_button = QPushButton("Generate Output File")
        self.output_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.output_button.clicked.connect(self.save_adj_matrix_for_specified_timeframe)
        self.graphLayout.addWidget(self.output_button)

        self.backend.subscribe(self.update_graphs)

        self.adj_img = None
        self.conGraph = None
        self.conGraphPos = None
        self.conGraphNodes = None
        self.conGraphEdges = None
        self.conGraphLabels = None

        self.oldMatrix = None
        self.oldKeys = []

    def set_up_adjacency_graph(self):
        # Layout for graphs and controls
        self.graphLayout = QVBoxLayout()
        self.content_layout.addLayout(self.graphLayout, stretch=5)

        # --- Adjacency graph ---
        self.figure_adj, self.ax_adj = plt.subplots(figsize=(6, 4))
        self.canvas_adj = FigureCanvas(self.figure_adj)
        #self.canvas_adj.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphLayout.addWidget(self.canvas_adj, stretch=5)


    def set_up_connection_graph(self):
        # --- Connection graph ---
        self.figure_conn, self.ax_conn = plt.subplots(figsize=(6, 4))
        self.canvas_conn = FigureCanvas(self.figure_conn)
        self.canvas_conn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphLayout.addWidget(self.canvas_conn, stretch=5)

    def update_graphs(self):
        adj_matrix = self.backend.adjacencyMatrix
        keys = self.backend.adjacencyMatrixKeys

        if adj_matrix is None: return
        # if self.oldMatrix is not None and np.array_equal(self.oldMatrix, adj_matrix): return
        try:
            # --- update adjacency graph ---
            if self.adj_img is None:
                self.adj_img = self.ax_adj.imshow(adj_matrix, cmap='Blues', interpolation='none', aspect='equal')
            else:
                self.ax_adj.clear()
                self.ax_adj.imshow(adj_matrix, cmap='Blues', interpolation='none', aspect='equal')
                self.ax_adj.set_title('Adjacency Matrix')
                self.ax_adj.set_xticks(np.arange(len(keys)))
                self.ax_adj.set_yticks(np.arange(len(keys)))
                self.ax_adj.set_xticklabels(keys, rotation=90)
                self.ax_adj.set_yticklabels(keys)
            
            # --- update connection graph ---
            if self.conGraph is None:
                self.conGraph = nx.from_numpy_array(adj_matrix)
                self.conGraphPos = nx.spring_layout(self.conGraph)
                self.conGraphNodes = nx.draw_networkx_nodes(self.conGraph, self.conGraphPos, ax=self.ax_conn, node_color='skyblue')
                self.conGraphEdges = nx.draw_networkx_edges(self.conGraph, self.conGraphPos, ax=self.ax_conn)
                self.conGraphLabels = nx.draw_networkx_labels(self.conGraph, self.conGraphPos, ax=self.ax_conn)
                self.ax_conn.set_title('Connection Graph')
            else:
                self.conGraph = nx.from_numpy_array(adj_matrix)
                # if len(self.conGraphPos) != len(self.conGraph):#Will only change graph if nodes are changed, if labels are then it will break
                if self.conGraphNodes:
                    self.conGraphNodes.remove()
                if self.conGraphLabels:
                    for label in self.conGraphLabels.values():
                        label.remove()
                self.conGraphPos = nx.spring_layout(self.conGraph, pos=self.conGraphPos, iterations=1)
                self.conGraphNodes = nx.draw_networkx_nodes(self.conGraph, self.conGraphPos, ax=self.ax_conn, node_color='skyblue')
                self.conGraphLabels = nx.draw_networkx_labels(self.conGraph, self.conGraphPos, ax=self.ax_conn)

                if self.conGraphEdges:
                    self.conGraphEdges.remove()
                self.conGraphEdges = nx.draw_networkx_edges(self.conGraph, self.conGraphPos, ax=self.ax_conn)

            if self.canvas_conn.width() > 0 and self.canvas_adj.height() > 0:
                self.canvas_conn.draw()
                self.canvas_adj.draw()

            self.oldMatrix = adj_matrix.copy()

        except ValueError as e:
            QMessageBox.critical(self, "Error", f"An error occurred during graph update: {e}")
        except Exception as e:
            print(f'Exception Occurred: {e.with_traceback}')

    def save_adj_matrix_for_specified_timeframe(self):
        #Save the adjacency matrices and network graphs to files.
        try:
            output_file, _ = QFileDialog.getSaveFileName(self, "Save Adjacency Matrix", "", "Text Files (*.txt);;All Files (*)")
            pdf_file, _ = QFileDialog.getSaveFileName(self, "Save Network Graphs PDF", "", "PDF Files (*.pdf);;All Files (*)")

            print("GraphDisplay -----------------------------------------------------------------1")

            if output_file and pdf_file:
                self.backend.instance.set_tle_data(self.backend.get_data_from_enabled_tle_slots())
                matrices = self.backend.instance.run_with_adj_matrix()

                print("GraphDisplay -----------------------------------------------------------------2")

                if not matrices:
                    QMessageBox.warning(self, "No Data", "No data was generated to save.")
                    return
                
                with open(output_file, 'w') as f:
                    print("GraphDisplay - Working on output file")
                    for timestamp, matrix in matrices:
                        formatted_timestamp = timestamp if isinstance(timestamp, str) else timestamp.utc_iso()
                        f.write(f"Time: {formatted_timestamp}\n")
                        np.savetxt(f, matrix, fmt='%d')
                        f.write("\n")

                with PdfPages(pdf_file) as pdf:
                    print("GraphDisplay - Working on PDF")
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
