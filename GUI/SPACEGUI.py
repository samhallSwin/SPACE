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

# ————————————————————————————————
#  1) OpenGL Globe
# ————————————————————————————————
class GlobeWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #Sets angle of rotation
        self.xRot = self.yRot = self.zRot = 0.0

       # self.lastPos = None
        self.quadric = None
        self.textureID = 0

    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        self.textureID = self.loadTexture("Earth.png")

        if not self.textureID:
            glDisable(GL_TEXTURE_2D)

        self.quadric = gluNewQuadric()
        gluQuadricTexture(self.quadric, GL_TRUE)

        # Set up perspective manually (since no resizeGL)
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
        ptr.setsize(img.byteCount())  # <-- THIS LINE WAS MISSING
        arr = np.array(ptr, dtype=np.uint8).reshape(h, w, 4)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex


    def paintGL(self):
        # Clear Screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        # GL Camera starting position eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
        #glRotatef(-90, 1, 0, 0)
        glRotatef(self.xRot, 1, 0, 0)
        glRotatef(self.yRot, 0, 1, 0)
        glRotatef(self.zRot, 0, 0, 1)

        if self.textureID:
            glBindTexture(GL_TEXTURE_2D, self.textureID)

        gluSphere(self.quadric, 1.0, 40, 40)

        if self.textureID:
            glBindTexture(GL_TEXTURE_2D, 0)


# ————————————————————————————————
#  2) Combined Clock + Globe + Slider Widget
# ————————————————————————————————
class TimeGlobeWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Time state
        self.time = QTime.currentTime()

        # Widgets
        self.welc_lbl     = QLabel("Welcome to S.P.A.C.E.")
        self.time_lbl     = QLabel(self.time.toString("hh:mm:ss"))
        self.start_btn    = QPushButton("Start")
        self.stop_btn     = QPushButton("Stop")
        self.now_btn      = QPushButton("Now")
        self.midnight_btn = QPushButton("Midnight")
        self.globe        = GlobeWidget()
        self.slider       = QSlider(Qt.Horizontal)

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        self._buildUI()

    def _buildUI(self):
        # Fonts
        font_id = QFontDatabase.addApplicationFont("A-Space Regular Demo.otf")
        fams = QFontDatabase.applicationFontFamilies(font_id)
        base = QFont(fams[0], 24) if fams else QFont("Arial", 24)
        self.welc_lbl.setFont(base)
        self.time_lbl.setFont(base)

        # Configure slider: 0–360°
        self.slider.setRange(0, 360)
        self.slider.setValue(0)
        self.slider.setTickInterval(30)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.onSlider)

        # Layout
        v = QVBoxLayout(self)
        v.addWidget(self.welc_lbl, alignment=Qt.AlignHCenter)
        v.addWidget(self.time_lbl, alignment=Qt.AlignHCenter)

        h = QHBoxLayout()
        for btn in (self.start_btn, self.stop_btn, self.now_btn, self.midnight_btn):
            h.addWidget(btn)
        v.addLayout(h)

        v.addWidget(self.globe, stretch=1)
        v.addWidget(self.slider)

        # Connect buttons
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.now_btn.clicked.connect(self.now)
        self.midnight_btn.clicked.connect(self.midnight)

        #Set clock to start on start up
        self.now()
        self.start()

    def _tick(self):
        self.time = self.time.addSecs(1)
        self.time_lbl.setText(self.time.toString("hh:mm:ss"))

    def onSlider(self, value):
        # Map slider (0–360) to globe rotation
        self.globe.yRot = -float(value)
        self.globe.update()

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def now(self):
        self.timer.stop()
        self.time = QTime.currentTime()
        self.time_lbl.setText(self.time.toString("hh:mm:ss"))
        self.timer.start()

    def midnight(self):
        self.timer.stop()
        self.time = QTime(0, 0, 0)
        self.time_lbl.setText(self.time.toString("hh:mm:ss"))


# ————————————————————————————————
#  3) Main Application Window
# ————————————————————————————————
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.P.A.C.E.")
        self.resize(1280, 720)
        self.setCentralWidget(TimeGlobeWidget())


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
