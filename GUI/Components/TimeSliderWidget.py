from PyQt5.QtCore import Qt, QTime, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QFontDatabase, QFont, QSurfaceFormat, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QRadioButton, QHBoxLayout, QPushButton, QSlider, QTimeEdit, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel
from OpenGL.GL import *
from OpenGL.GLU import *

from skyfield.api import load
from skyfield.api import Timescale
from skyfield.api import EarthSatellite
from GlobeWidget import GlobeWidget
import datetime
from Backend import Backend
import matplotlib

#TEMP#
import random #To temporarily select a couple satellites from the 3le file sam posted
from skyfield.constants import ERAD

Seconds_Per_Day = 24 * 60 *60
Earth_Degrees = 360
Seconds_Per_Degree = Seconds_Per_Day / 360

def degrees_to_sec(deg: int):
    return int(deg * Seconds_Per_Degree) % Seconds_Per_Day

def seconds_to_degrees(s: int):
    s = s % Seconds_Per_Day
    return int(s / Seconds_Per_Degree)





class TimeSliderWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Clock Display")
        self.setGeometry(100, 100, 800, 400)

        #Layout Boxes       
        outer = QVBoxLayout()
        globe_layout = QVBoxLayout()
        controlpanel = QVBoxLayout()
        time_controls = QHBoxLayout()
        settings = QVBoxLayout()
        self.globe = GlobeWidget()
        globe_layout.addWidget(self.globe)


#------ Clock Display
        #font_id = QFontDatabase.addApplicationFont('Assets/A-Space Regular Demo.otf')
        #self.font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        #self.space_font = QFont(self.font_family, 50)
        self.space_font = QFont("Arial", 50)

        self.time_display = QLineEdit(self)
        self.time_display.setReadOnly(True)  # Prevent edits
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setFont(self.space_font)
        self.time_display.setStyleSheet("border: none; background: transparent;")  # look like QLabel

        # Accessibility for testing automation
        self.time_display.setAccessibleName("clock_Time")
        self.time_display.setAccessibleDescription(QTime.currentTime().toString("hh:mm:ss"))

#-------Slider
        # Time Window
        start_time = 0
        end_time = Seconds_Per_Day

        self.timeedit_lbl = QLabel("Set Time Frame (24hr)")
        self.start_tf = QTimeEdit(self)
        self.end_tf   = QTimeEdit(self)
        self.timeedit_btn = self.style_button(QPushButton("Change Time Frame"))
        self.timeedit_btn.clicked.connect(self.edit_time)

        for w in (self.start_tf, self.end_tf):
            w.setDisplayFormat("HH:mm")             
            w.setMinimumTime(QTime(0, 0))
            w.setMaximumTime(QTime(23, 59))
            w.setAccelerated(True)    
        

        self.slider = self.style_slider(QSlider(Qt.Horizontal))
        self.slider.setRange(start_time, end_time)
        self.slider.setValue(90)  
        self.slider.setTickInterval(30)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.onSlider)

        # Timer that updates clock
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.now()
        self.start()


        

# ----- Time Controls
        # Play button
        self.play_btn = self.style_button(QPushButton("▶"))
        self.play_btn.clicked.connect(self.scroll)
        # Speed Controls
        self.onex_lbl = QLabel("1x")
        self.onex_radiobtn = QRadioButton()
        self.onex_radiobtn.setGeometry(QtCore.QRect(180, 120, 95, 20))
        self.twox_lbl = QLabel("2x")
        self.twox_radiobtn = QRadioButton()
        self.twox_radiobtn.setGeometry(QtCore. QRect(180, 120, 95, 20))
        # Stop Button
        self.stop_btn = self.style_button(QPushButton("Stop"))
        self.stop_btn.clicked.connect(self.stop)
     
        # Start Button
        self.start_btn = self.style_button(QPushButton("Start"))
        self.start_btn.clicked.connect(self.start)

        # Current Time Button
        self.now_btn = self.style_button(QPushButton("Now"))
        self.now_btn.clicked.connect(self.now)
      
        # Reset Button
        self.midnight_btn = self.style_button(QPushButton("Midnight"))
        self.midnight_btn.clicked.connect(self.midnight)
       


        # Layout
        lower = QHBoxLayout()

        for w in (self.timeedit_lbl, self.start_tf, self.end_tf, self.timeedit_btn, self.onex_lbl,  self.onex_radiobtn, self.twox_lbl, self.twox_radiobtn):
            settings.addWidget(w)
        for w in (self.time_display, self.slider):
            controlpanel.addWidget(w)
        for w in (self.play_btn, self.stop_btn, self.start_btn, self.now_btn, self.midnight_btn):
            time_controls.addWidget(w)
        controlpanel.addLayout(time_controls)
        for l in (settings, controlpanel):
            lower.addLayout(l)
        for l in (globe_layout, lower):
            outer.addLayout(l)


        
        self.setLayout(outer)
        
      
        

    # Animations Controls
        self.anim = QPropertyAnimation(self.slider, b"value", self)
        self.anim.setEasingCurve(QEasingCurve.Linear)
        self.anim.setDuration(Rot_Dur) #Orbit duration

    # Time Controls
    # Updates display clock every second
    def tick(self):
        self.time = self.time.addSecs(1)
        self.time_display.setText(self.time.toString("hh:mm:ss"))
        self.time_display.setAccessibleDescription(self.time.toString("hh:mm:ss"))

    def start(self):
        self.timer.start()

    def now(self):
        self.timer.stop()
        self.time = QTime.currentTime()
        self.time_display.setText(self.time.toString("hh:mm:ss"))
        self.slider.blockSignals(True)
        self.slider.setValue((QTime(0, 0).secsTo(self.time)))
        self.slider.blockSignals(False)
        self.timer.start()

    def midnight(self):
        self.timer.stop()
        self.time = QTime(0, 0, 0)
        self.slider.setValue(0)
        self.time_display.setText(self.time.toString("hh:mm:ss"))

    def stop(self):
        self.timer.stop()
        self.anim.stop()

    # Slider Controls
    def onSlider(self, value):
        print(seconds_to_degrees(value))
        self.timer.stop()
        self.time = QTime(0,0).addSecs(value)
        self.time_display.setText(self.time.toString("hh:mm:ss"))
        self.timer.start()
        self.globe.yRot = float(value/240)
        self.globe.update()

    def update_display_and_slider(self):
        self.time_display.setText(self.time.toString("hh:mm:ss"))
        self.time_display.setAccessibleDescription(self.time.toString("hh:mm:ss"))
        self.slider.blockSignals(True)
        self.slider.setValue(degrees_to_sec(QTime(0,0).secsTo(self.time)))
        self.slider.blockSignals(True)

    def scroll(self):
        self.anim.stop()
        self.anim.setStartValue(self.slider.value())
        self.anim.setEndValue(self.slider.maximum())
        self.anim.start()
    
    def edit_time(self):
        self.slider.setRange((self.start_tf.time().msecsSinceStartOfDay() // 1000),(self.end_tf.time().msecsSinceStartOfDay() // 1000))
        print(self.start_tf.time(), self.end_tf.time())
        print((self.start_tf.time().msecsSinceStartOfDay() // 1000),(self.end_tf.time().msecsSinceStartOfDay() // 1000))

        

#---------------------------------------------------------------
# Styling

    def style_button(self, btn):
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(20)
        self.btn_font = QFont("Arial",20)
        btn.setFont(self.btn_font)
        btn.setStyleSheet("""
            QPushButton {
                background: #3A86FF;
                color: #FFFFFF;
                border: 1px solid #2A6FD6;
                border-radius: 8px;
                padding: 6px 12px;
               
            }
            QPushButton:hover   { background: #4F95FF; }
            QPushButton:pressed { background: #2A6FD6; }
            QPushButton:disabled{
                background: #D9E6FF;
                color: #8AA9E6;
                border-color: #BFD2FF;
            }
        """)
        return btn

    def style_slider(self, slider):
        slider.setCursor(Qt.PointingHandCursor)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #E6EAF2;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3A86FF;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #E6EAF2;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #2A6FD6;
                width: 16px;
                margin: -5px 0;   /* centers thumb on 6px track */
                border-radius: 8px;
            }
        """)
        return slider

        