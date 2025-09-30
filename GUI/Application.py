import sys
import numpy as np
from PyQt5.QtCore import Qt, QTime, QTimer
from PyQt5.QtGui import QFontDatabase, QFont, QSurfaceFormat, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QOpenGLWidget, QLineEdit, QSpinBox, QTimeEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, QPropertyAnimation, QEasingCurve
from OpenGL.GL import *
from OpenGL.GLU import *


#----------------------------------------------------------------
# Constants

Seconds_Per_Day = 24 * 60 *60
Earth_Degrees = 360
Seconds_Per_Degree = Seconds_Per_Day / 360
Rot_Dur = 4000 #Aninmation speed in miliseconds
#---------------------------------------------------------------
# Convert seconds since midnight to degrees of rotation

def degrees_to_sec(deg: int):
    return float(deg * Seconds_Per_Degree) % Seconds_Per_Day

def seconds_to_degrees(s: int):
    s = s % Seconds_Per_Day
    return float(s / Seconds_Per_Degree)







# ————————————————————————————————
#  1) OpenGL Globe
# ————————————————————————————————
class GlobeWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.xRot = 0.0
        self.yRot = 90.0  # Start with prime meridian facing front
        self.zRot = 0.0
        self.quadric = None
        self.textureID = 0

    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        # Load Textures
        self.textureID = self.loadTexture("Assets/Earth.png")
        self.bgtextureID = self.loadTexture("Assets/Background.jpg")
        if not self.textureID:
            glDisable(GL_TEXTURE_2D)
        else:
            print(f"[OK] Texture loaded. ID: {self.textureID}")

        # Flip texture vertically
        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glScalef(1.0, -1.0, 1.0)  # Flip vertically
        glMatrixMode(GL_MODELVIEW)

        self.quadric = gluNewQuadric()
        gluQuadricTexture(self.quadric, GL_TRUE)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width() / (self.height() or 1), 1.0, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def loadTexture(self, path):
        img = QImage(path)
        if img.isNull():
            print(f"[Warning] Failed to load texture: {path}")
            return 0

        img = img.convertToFormat(QImage.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr, dtype=np.uint8).reshape(h, w, 4)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex

    def drawCone(self, radius, height, num_slices):
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(1.0, 0.5, 0.0, 0.5)  # Transparent orange
        glVertex3f(0, height / 2, 0)
        for i in range(num_slices + 1):
            angle = i * (2.0 * np.pi) / num_slices
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            glVertex3f(x, -height / 2, z)
        glEnd()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Position the camera
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

        # Align poles vertically (Z → Y)
        glRotatef(-90, 1, 0, 0)  # Rotate globe forward so poles face up/down

        # Rotate east–west around vertical (Z now acts as vertical)
        glRotatef(self.yRot, 0, 0, 1)

        # Draw Earth (opaque)
        if self.textureID:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.textureID)

        glColor4f(1.0, 1.0, 1.0, 1.0)  # Ensure Earth is fully opaque
        gluSphere(self.quadric, 1.0, 40, 40)

        if self.textureID:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)

        # Draw transparent cone
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        self.drawCone(radius=1.5, height=2.0, num_slices=40)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)


# ————————————————————————————————
#  2) Time + Globe + Slider UI
# ————————————————————————————————
class MainMenu(QWidget):
    def __init__(self):
        super().__init__()

        #Layout Boxes       
        outer = QVBoxLayout()
        globe_layout = QVBoxLayout()
        controlpanel = QVBoxLayout()
        time_controls = QHBoxLayout()
        self.globe = GlobeWidget()
        globe_layout.addWidget(self.globe)


#------ Clock Display
        font_id = QFontDatabase.addApplicationFont('Assets\A-Space Regular Demo.otf')
        self.font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        self.space_font = QFont(self.font_family, 50)

        self.time_display = QLineEdit(self)
        self.time_display.setReadOnly(True)  # Prevent edits
        self.time_display.setAlignment(Qt.AlignCenter)
        self.time_display.setFont(self.space_font)
        self.time_display.setStyleSheet("border: none; background: transparent;")  # look like QLabel

        # Accessibility for testing automation
        self.time_display.setAccessibleName("clock_Time")
        self.time_display.setAccessibleDescription(QTime.currentTime().toString("hh:mm:ss"))
        controlpanel.addWidget(self.time_display)

#-------Slider
        self.slider = self.style_slider(QSlider(Qt.Horizontal))
        self.slider.setRange(0, Seconds_Per_Day)
        self.slider.setValue(90)  
        self.slider.setTickInterval(30)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.onSlider)
        controlpanel.addWidget(self.slider)

        # Timer that updates clock
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.now()
        self.start()


        # Time Window
        #start_time = 
        #end_time = 


        #start_edit = QTimeEdit(start_time or QTime.currentTime())
        #end_edit   = QTimeEdit(end_time   or QTime.currentTime().addSecs(3600))

        #self.slider.setRange(start_time, end_time)


# ----- Time Controls
        # Play button
        self.play_btn = self.style_button(QPushButton("▶"))
        self.play_btn.clicked.connect(self.scroll)
        time_controls.addWidget(self.play_btn)
        # Stop Button
        self.stop_btn = self.style_button(QPushButton("Stop"))
        self.stop_btn.clicked.connect(self.stop)
        time_controls.addWidget(self.stop_btn)
        # Start Button
        self.start_btn = self.style_button(QPushButton("Start"))
        self.start_btn.clicked.connect(self.start)
        time_controls.addWidget(self.start_btn)
        # Current Time Button
        self.now_btn = self.style_button(QPushButton("Now"))
        self.now_btn.clicked.connect(self.now)
        time_controls.addWidget(self.now_btn)
        # Reset Button
        self.midnight_btn = self.style_button(QPushButton("Midnight"))
        self.midnight_btn.clicked.connect(self.midnight)
        time_controls.addWidget(self.midnight_btn)


        # Layout
        controlpanel.addLayout(time_controls)
        outer.addLayout(globe_layout)
        outer.addLayout(controlpanel)
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

#---------------------------------------------------------------
# Styling

    def style_button(self, btn):
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(36)
        self.btn_font = QFont(self.font_family, 20)
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

# ————————————————————————————————
#  3) Main Window
# ————————————————————————————————
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.P.A.C.E.")
        self.resize(1280, 720)
        self.setCentralWidget(MainMenu())

# ————————————————————————————————
#  4) Application Entry
# ————————————————————————————————
if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())