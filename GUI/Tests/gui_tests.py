import unittest
import sys
import time
import subprocess

from pywinauto import Application
from pathlib import Path

# To run tests:
# python -m unittest -v GUI.Tests.gui_tests

class TestGUIWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        python_exe = Path(".venv") / "Scripts" / "python.exe"
        script_path = Path("GUI") / "Application.py"
        cls.proc = subprocess.Popen([str(python_exe), str(script_path)])

        time.sleep(2)

        cls.app = Application(backend="uia").connect(title="S.P.A.C.E")
        cls.window = cls.app.window(title="S.P.A.C.E")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()

    def test_window_exists(self):
        self.assertTrue(self.window.exists())
        
    def test_midnight_button_exists(self):
        button = self.window.child_window(title="Midnight2", control_type="Button")
        self.assertTrue(button.exists())

    def test_print_all_elements(self):
        self.window.print_control_identifiers()

if __name__ == "__main__":
    unittest.main()