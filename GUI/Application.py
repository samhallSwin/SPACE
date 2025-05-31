import sys
import numpy as np
from PyQt5.QtCore import Qt, QTime, QTimer
from PyQt5.QtGui import QFontDatabase, QFont, QSurfaceFormat, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QOpenGLWidget
)
from OpenGL.GL import *
from OpenGL.GLU import *

class MainWindow(QMainWindow):
    
    def __init__(self): 
        super().__init__()
        #Initialize window and place OpengGL inside 
        self.setWindowTitle("S.P.A.C.E")
        self.setGeometry(100, 100, 640, 480)
        self.sphere = Sphere(self) 
        self.setCentralWidget(self.sphere)



class  Sphere(QOpenGLWidget):
    def initializeGL(self):
        glClearColor(0.1,0.2,0.2,1.0)

    def resizeGL(self,w,h):
        glViewport(0,0,w,h)       



    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # self.verticies  = (
        #     -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
        #     0.5, -0.5, 0.0, 0.0, 1.0, 0.0,
        #     0.0, 0.5, 0.0, 0.0, 0.0, 1.0
        # )

        # #Numpy arrary datatype converts verticies into a format readable by gpu
        # self.verticies = np.arrary(self.vertices, dtype=np.float32)

        # self.vertex_count = 3

        # self.veo = glGenVertexArrays(1)
        # glBindVertexArray(self.veo)
        # #Vertex buffer object
        # self.vbo = glGenBuffers(1)
        # glBinBuffer(GL_ARRAY_BUFFER, self.vbo)
        # glBufferData(GL_ARRAY_BUFFER, self,verticies,nbytes, self,verticies, GLSTATIC_DRAW)

        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

