import unittest
import sys
from PyQt5.QtWidgets import QApplication

from GUI import Application

# python -m unittest GUI/Tests/gui_tests.py in CMD to run tests

# Ensure a single QApplication is created
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class TestGUIWindow(unittest.TestCase):
    def setUp(self):
        self.window = Application.MainWindow()

    def tearDown(self):
        self.window.close()

    def test_open_window(self):
        self.assertFalse(self.window.isVisible())

        self.window.show()
        self.assertTrue(self.window.isVisible())
    
    