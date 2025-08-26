import sys
import numpy as np
from PyQt5.QtCore import Qt, QTime, QTimer
from PyQt5.QtGui import QFontDatabase, QFont, QSurfaceFormat, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QOpenGLWidget, QLineEdit
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


#----------------------------------------------------------------
# Constants


Seconds_Per_Day = 24 * 60 *60
Earth_Degrees = 360
Seconds_Per_Degree = Seconds_Per_Day / 360

#---------------------------------------------------------------
# Convert seconds since midnight to degrees of rotation

def degrees_to_sec(deg: int):
    return int(round(deg * Seconds_Per_Degree)) % Seconds_Per_Day

def seconds_to_degrees(s: int):
    s = s % Seconds_Per_Day
    return int(round(s / Seconds_Per_Degree))

#----------------------------------------------------------------
# Time and Slider Widget

class TimeSliderWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Clock Display")
        self.setGeometry(100, 100, 800, 400)

        font = QFont('Arial', 120, QFont.Bold)

        layout = QVBoxLayout()

        # Clock Display
        self.time_display = QLineEdit(self)
        self.time_display.setReadOnly(True)  # Prevent edits
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setFont(font)
        self.time_display.setStyleSheet("border: none; background: transparent;")  # look like QLabel

        # Accessibility for testing automation
        self.time_display.setAccessibleName("clock_Time")
        self.time_display.setAccessibleDescription(QTime.currentTime().toString("hh:mm:ss"))

        # Set time to current time
        #self.time = QTime.currentTime()
        # Timer that updates clock
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.now()
        self.start()

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 360)
        self.slider.setValue(90)  
        self.slider.setTickInterval(30)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.onSlider)

        # Play button
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.scroll)
        # Stop Button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        # Start Button
        self.start_btn    = QPushButton("Start")
        self.start_btn.clicked.connect(self.start)
        # Current Time Button
        self.now_btn      = QPushButton("Now")
        self.now_btn.clicked.connect(self.now)
        # Reset Button
        self.midnight_btn = QPushButton("Midnight")
        self.midnight_btn.clicked.connect(self.midnight)


        # Layout
        layout.addWidget(self.time_display)
        layout.addWidget(self.slider)
        self.setLayout(layout)


    # Time Controls
    # Updates display clock every second
    def tick(self):
        self.time = self.time.addSecs(1)
        self.time_display.setText(self.time.toString("hh:mm:ss"))
        #self.time_display.setAccessibleDescription(self.time)

    def start(self):
        self.timer.start()

    def now(self):
        self.timer.stop()
        self.time = QTime.currentTime()
        self.time_display.setText(self.time.toString("hh:mm:ss"))
        self.timer.start()

    def midnight(self):
        self.timer.stop()
        self.time = QTime(0, 0, 0)
        self.time_display.setText(self.time.toString("hh:mm:ss"))

    def stop(self):
        self.timer.stop()

    # Slider Controls
    def onSlider(self, value):
        print(value)
        self.timer.stop()
        s =  value * 240
        total = int(round(s)) % 86400
        self.time = QTime(0,0).addSecs(total)
        self.time_display.setText(self.time.toString("hh:mm:ss"))
        self.timer.start()




# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TimeSliderWidget()
    window.show()
    sys.exit(app.exec_())