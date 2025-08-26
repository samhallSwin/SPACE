import sys
import numpy as np
from PyQt5.QtCore import Qt, QTime, QTimer
from PyQt5.QtGui import QFontDatabase, QFont, QSurfaceFormat, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QOpenGLWidget, QTimeEdit, QLineEdit
)
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFrame, QGraphicsOpacityEffect, QScrollArea
from PyQt5.QtCore import pyqtSignal
import math
import datetime
import os
import skyfield
from skyfield.api import load
from skyfield.api import Timescale
from skyfield.api import EarthSatellite
import time
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve

class Window(QWidget):

    def __init__(self):
        super().__init__()

        # Window geometry
        self.setGeometry(100, 100, 800, 400)
        self.setWindowTitle("Clock Display")

        # Layout
        layout = QVBoxLayout()

        # Font
        font = QFont('Arial', 120, QFont.Bold)


        # QLineEdit as display-only clock
        self.time_display = QLineEdit(self)
        self.time_display.setReadOnly(True)  # Prevent edits
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setFont(font)
        self.time_display.setStyleSheet("border: none; background: transparent;")  # look like QLabel

        # Accessibility for automation
        self.time_display.setAccessibleName("clock_Time")
        self.time_display.setAccessibleDescription(QTime.currentTime().toString("hh:mm:ss"))

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        #self.slider.setAccessibleName("slider_Time")

        self.slider.setRange(0, 360)
        self.slider.setValue(90)  # Start with Earth facing front
        self.slider.setTickInterval(30)
        #self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.onSlider)
        # Animations
        self.anim = QPropertyAnimation(self.slider, b"value", self)
        self.anim.setEasingCurve(QEasingCurve.Linear)
        self.anim.setDuration(4000) 

        # Play button
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.scroll)
        # Stop Button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)

        # Add to layout
        layout.addWidget(self.time_display)
        layout.addWidget(self.slider)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        # Timer updates every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        # Initial display
        self.update_time()
        self.midnight

     

# what to do on slider interaction
    def onSlider(self, value):
        print(value)
# auto scroll slider
    def scroll(self):
        self.anim.stop()
        self.anim.setStartValue(self.slider.value())
        self.anim.setEndValue(self.slider.maximum())
        self.anim.start()
        

    def stop(self):
        self.anim.stop()

    def update_time(self):
        current_time = QTime.currentTime().toString("hh:mm:ss")
        self.time_display.setText(current_time)
        self.time_display.setAccessibleDescription(current_time)

    def midnight(self):
        current_time = QTime.fromMSecsSinceStartOfDay(0).toString("hh:mm:ss")
        self.time_display.setText(current_time)
        self.time_display.setAccessibleDescription(current_time)

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())