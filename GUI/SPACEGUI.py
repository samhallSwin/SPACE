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
        self.xRot = 0.0
        self.yRot = 90.0  # Start with prime meridian facing front
        self.zRot = 0.0
        self.quadric = None
        self.textureID = 0

    def initializeGL(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        self.textureID = self.loadTexture("Earth.png")
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

        # FIX 1: Align poles vertically (Z → Y)
        glRotatef(-90, 1, 0, 0)  # Rotate globe forward so poles face up/down

        # FIX 2: Rotate east–west around vertical (Z now acts as vertical)
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
class TimeGlobeWidget(QWidget):
    def __init__(self):
        super().__init__()
        # self.time = QTime.currentTime()

        # self.welc_lbl     = QLabel("Welcome to S.P.A.C.E.")
        # self.time_lbl     = QLabel(self.time.toString("hh:mm:ss"))
        # self.start_btn    = QPushButton("Start")
        # self.stop_btn     = QPushButton("Stop")
        # self.now_btn      = QPushButton("Now")
        # self.midnight_btn = QPushButton("Midnight")
        # self.globe        = GlobeWidget()
        # self.slider       = QSlider(Qt.Horizontal)

        # self.timer = QTimer(self)
        # self.timer.setInterval(1000)
        # self.timer.timeout.connect(self._tick)

        self._buildUI()

    def _buildUI(self):
        font_id = QFontDatabase.addApplicationFont("A-Space Regular Demo.otf")
        fams = QFontDatabase.applicationFontFamilies(font_id)
        base = QFont(fams[0], 24) if fams else QFont("Arial", 24)
        self.welc_lbl.setFont(base)
        self.time_lbl.setFont(base)

        tle = TLEDisplay()



        self.slider.setRange(0, 360)
        self.slider.setValue(90)  # Start with Earth facing front
        self.slider.setTickInterval(30)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.onSlider)

        # v = QVBoxLayout(self)
        # v.addWidget(self.welc_lbl, alignment=Qt.AlignHCenter)
        # v.addWidget(self.time_lbl, alignment=Qt.AlignHCenter)

        # h = QHBoxLayout()
        # for btn in (self.start_btn, self.stop_btn, self.now_btn, self.midnight_btn):
        #     h.addWidget(btn)
        # v.addLayout(h)

        # v.addWidget(self.globe, stretch=1)
        # v.addWidget(self.slider)

    #     self.start_btn.clicked.connect(self.start)
    #     self.stop_btn.clicked.connect(self.stop)
    #     self.now_btn.clicked.connect(self.now)
    #     self.midnight_btn.clicked.connect(self.midnight)

    #     self.now()
    #     self.start()

    # def _tick(self):
    #     self.time = self.time.addSecs(1)
    #     self.time_lbl.setText(self.time.toString("hh:mm:ss"))

    # def onSlider(self, value):
    #     self.globe.yRot = float(value)
    #     self.globe.update()

    # def start(self):
    #     self.timer.start()

    # def stop(self):
    #     self.timer.stop()

    # def now(self):
    #     self.timer.stop()
    #     self.time = QTime.currentTime()
    #     self.time_lbl.setText(self.time.toString("hh:mm:ss"))
    #     self.timer.start()

    # def midnight(self):
    #     self.timer.stop()
    #     self.time = QTime(0, 0, 0)
    #     self.time_lbl.setText(self.time.toString("hh:mm:ss"))

# ————————————————————————————————
#  3) Main Window
# ————————————————————————————————
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.P.A.C.E.")
        self.resize(1280, 720)
        self.setCentralWidget(TimeGlobeWidget())

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


from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFrame, QGraphicsOpacityEffect, QScrollArea
from PyQt5.QtCore import pyqtSignal

class TLEDisplay(QFrame):
    update_signal = pyqtSignal()

    def __init__(self, parent=None):

        self.full_width = 300  # Total width when expanded
        self.collapsed_width = 20

        self.above_height_amount = 0
        self.control_panel_height_reduction = 92 #TODO: find a better way to access this information
        
        self.expanded = True
        self.tle_dict = {}
        
        super().__init__(parent)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid gray;")
        self.setParent(parent)
        self.setVisible(True)

        self.set_up_content_window()
        
        self.set_up_drop_label()
        self.set_up_scroll_box()

        self.set_up_collapse_button()
        
        self.update_height()
        self.toggle_width()

    def set_up_content_window(self):
        self.overlay = QWidget(self)
        self.overlay.setGeometry(0, 0, self.full_width if self.expanded else self.collapsed_width, self.parent().height() - self.overlay.height() - self.control_panel_height_reduction)
        self.overlay.setStyleSheet("background-color: #dddddd;")
        self.overlay.setVisible(True)

        self.content_window = QVBoxLayout()
        self.overlay.setLayout(self.content_window)

    def set_up_drop_label(self):
        # Use a horizontal layout
        self.drop_layout = QHBoxLayout()
        self.drop_layout.setContentsMargins(10, 10, 10, 10)
        self.drop_layout.setSpacing(15)

        # Add DropLabel on the left
        self.drop_label = DropLabel(self)
        self.drop_layout.addWidget(self.drop_label)

        # Add your existing content on the right
        self.right_content = QLabel("Collapsible content goes here.")
        self.right_content.setWordWrap(True)
        self.drop_layout.addWidget(self.right_content)

        self.content_window.addLayout(self.drop_layout)

        #set up event listener
        self.drop_label.fileDropped.connect(self.on_file_dropped)
        
    def set_up_scroll_box(self):
    # Use a vertical layout
        scroll_layout = QVBoxLayout()
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        self.inner_layout = QVBoxLayout(scroll_widget)
        
        scroll_area.setWidget(scroll_widget)
        scroll_layout.addWidget(scroll_area)

        self.content_window.addLayout(scroll_layout)

    def populate_scroll_items(self):
        #clear all existing items
        #TODO: update existing items, then Add/Clear additional items as needed.
        for i in reversed(range(self.inner_layout.count())):
            widget_to_remove = self.inner_layout.itemAt(i).widget()
            
            #self.inner_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
            widget_to_remove.deleteLater()

        for i in self.tle_dict:
            name = str(i)
            line_1 = self.tle_dict[name][0]
            line_2 = self.tle_dict[name][1]

            tle_slot = TLESlot(self, name, line_1, line_2)
            tle_slot.checkbox.stateChanged.connect(self.get_data_from_enabled_tle_slots)
            tle_slot.checkbox.stateChanged.connect(self.emit_update_signal)

            self.inner_layout.addWidget(tle_slot)

    def get_data_from_enabled_tle_slots(self):
        return_dict = {}

        for i in range(self.inner_layout.count()):
            slot = self.inner_layout.itemAt(i).widget()
            if(isinstance(slot, TLESlot)):
                if slot.checkbox.isChecked():
                    return_dict[slot.tle_name] = [slot.tle_line_1, slot.tle_line_2]
        
        print(f"Return_dict: {return_dict}")
        return return_dict

    def add_elements(self, tle_data):
        for i in tle_data:
            name = str(i)
            line_1 = tle_data[name][0]
            line_2 = tle_data[name][1]

            self.tle_dict[name] = [line_1, line_2]

        self.populate_scroll_items()

    def delete_element(self, element_name):
        del self.tle_dict[element_name]

        self.populate_scroll_items()


    def set_up_collapse_button(self):
        self.toggle_button = QPushButton("▼\nE\nx\np\na\nn\nd", self)
        self.toggle_button.setGeometry(0, self.above_height_amount, 20, self.height())

        self.opacity_effect = QGraphicsOpacityEffect(self.toggle_button)
        self.toggle_button.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)

        # Set up enter/leave events
        self.toggle_button.installEventFilter(self)

        self.toggle_button.clicked.connect(self.toggle_width)

    def toggle_width(self):
        self.expanded = not self.expanded

        #start_rect = self.geometry()
        end_width = self.full_width if self.expanded else self.collapsed_width
        #end_rect = QRect(0, 0, self.parent().width(), end_height)
        self.setGeometry(0, 0, end_width, self.height())  # Start with just the button height

        self.overlay.setVisible(self.expanded)
        self.toggle_button.setText("▲\nC\no\nl\nl\na\np\ns\ne" if self.expanded else "▼\nE\nx\np\na\nn\nd")
        
    def update_height(self):
        target_width = self.full_width if self.expanded else self.collapsed_width
        target_height = self.parent().height() - self.above_height_amount - self.control_panel_height_reduction

        self.setGeometry(0, self.above_height_amount, target_width, target_height)  # Start with just the button height
        self.toggle_button.setGeometry(0, self.above_height_amount, 20, target_height)

    def emit_update_signal(self):
        self.update_signal.emit()
        print("Sending update signal")

    def on_file_dropped(self, tle_file_path):
        print(tle_file_path)
        tle_data = self.read_tle_file(tle_file_path)

        self.add_elements(tle_data)

    #yoinked from sat_sim_handler.
    #TODO: figure out how to reference the method in the sat_sim_handler, to reduce duplicated code.
    def read_tle_file(self, file_path):
        # Reads TLE data from the specified file path.
        tle_data = {}
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            # Handle files with or without a title line (3LE or 2LE)
            i = 0
            while i < len(lines):
                if lines[i].startswith('1') or lines[i].startswith('2'):
                    # This is the 2LE format (no title line)
                    tle_line1 = lines[i].strip()
                    tle_line2 = lines[i + 1].strip()
                    tle_data[f"Satellite_{i // 2 + 1}"] = [tle_line1, tle_line2]
                    i += 2
                else:
                    # 3LE format (title line included)
                    name = lines[i].strip()
                    tle_line1 = lines[i + 1].strip()
                    tle_line2 = lines[i + 2].strip()
                    tle_data[name] = [tle_line1, tle_line2]
                    i += 3

            if not tle_data:
                return None
            return tle_data
        
        except OSError:
            raise
        except Exception as e:
            print(f"Error reading TLE file at {file_path}: {e}")
            raise ValueError("Error reading TLE file.")
#endregion
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#region - TLE Slot
from PyQt5.QtWidgets import QApplication,QPushButton, QCheckBox, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import QEvent, QObject, QSize

class TLESlot(QWidget):
    def __init__(self, tle_display, name, line_1, line_2):
        self.enabled = True
        self.delete_me = False

        self.tle_display = tle_display

        self.tle_name = name
        self.tle_line_1 = line_1
        self.tle_line_2 = line_2

        super().__init__()
        self.setMinimumHeight(150)

        self.layout = QHBoxLayout(self)
        
        self.set_up_display_layer()

        # Install event filter on the entire window
        #self.installEventFilter(self)
        QApplication.instance().installEventFilter(self)

    def set_up_display_layer(self):
        #checkbox
        self.checkbox = QCheckBox("")
        self.checkbox.setToolTip("Enable" if self.enabled else "Disable")

        indicator_size = 15 # Default checkbox indicator size in most styles
        self.checkbox.setFixedSize(QSize(indicator_size, indicator_size))
        
        # Optional: Remove padding/margin (works in many styles)
        self.checkbox.setStyleSheet("QCheckBox::indicator { margin: 0px; }")
        self.checkbox.setChecked(True)

        self.checkbox.stateChanged.connect(self.checkbox_changed)

        self.layout.addWidget(self.checkbox)

        #label
        item = QLabel(self.tle_name)
        self.layout.addWidget(item)

        #text
        self.toggle_button = QPushButton("▼" if self.enabled else "▲", self)
        self.toggle_button.setGeometry(0, 0, 20, 20)
        self.toggle_button.clicked.connect(self.toggle)
        self.layout.addWidget(self.toggle_button)

        self.delete_button = QPushButton("🗑")
        self.delete_button.setGeometry(0, 0, 20, 20)
        self.delete_button.clicked.connect(self.delete)
        self.layout.addWidget(self.delete_button)

    def checkbox_changed(self, state):
        if state == 2:  # 2 = Checked, 0 = Unchecked, 1 = Partially checked
            self.enabled = True
        else:
            self.enabled = False

        self.checkbox.setToolTip("Enable" if self.enabled else "Disable")    

    def delete(self):
        if(self.delete_me == False):
            self.delete_me = True
            self.delete_button.setStyleSheet("background-color: red;")
        else:
            self.tle_display.delete_element(self.tle_name)
        
    def toggle(self):
        self.toggle_button = QPushButton("▼" if self.enabled else "▲", self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.MouseButtonPress:
            # Check if the click is outside the button
            if not self.delete_button.geometry().contains(event.pos()):
                if self.delete_me:                
                    self.delete
                else:
                    self.delete_me = False
                    self.delete_button.setStyleSheet("")

        return super().eventFilter(obj, event)
#endregion
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#region - DropLabel
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal

class DropLabel(QLabel):
    fileDropped = pyqtSignal(str)  # Emits TLE content string
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop\nHere")
        self.setFixedSize(100, 100)
        self.setStyleSheet("""
            border: 2px dashed #888;
            border-radius: 6px;
            background-color: #f8f8f8;
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #00aaff;
                    background-color: #e0f7ff;
                    padding: 20px;
                    font-size: 14px;
                }
            """)
        else:
            event.ignore()

    def dropEvent(self, event):
        self.resetStyle()
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(".txt") or file_path.endswith(".tle"):
                with open(file_path, "r") as file:
                    self.fileDropped.emit(file_path)
                break

    def dragLeaveEvent(self, event):
        self.resetStyle()

    def resetStyle(self):
        self.setStyleSheet("""
            border: 2px dashed #888;
            border-radius: 6px;
            background-color: #f8f8f8;
        """)
#endregion