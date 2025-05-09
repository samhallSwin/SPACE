from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import sys


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # set the title
        self.setWindowTitle("S.P.A.C.E.")

        # setting  the geometry of window
        self.setGeometry(0, 0, 1920, 1080)

        self.UI()

        # show all the widgets
        self.show()

    def UI(self):

        # set background image using stylesheet
        self.setStyleSheet(
            "QMainWindow { background-image: url('digital-earth.jpg'); "
            "background-repeat: no-repeat; "
            "background-position: center; "
            "background-size: cover;"
            "background-color: black; }")

        #Enter button
        self.btn = QPushButton("ENTER S.P.A.C.E.", self)
        self.btn.setGeometry(200, 150, 200, 50)
        self.btn.clicked.connect(self.changewindow)


    def changewindow(self):
        print(self.geometry())



# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# start the app
sys.exit(App.exec())