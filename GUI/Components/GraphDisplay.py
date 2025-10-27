from PyQt5.QtWidgets import QGraphicsOpacityEffect, QSizePolicy, QPushButton, QVBoxLayout, QWidget, QMessageBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
        self.canvas_adj.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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