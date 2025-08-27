import unittest
import sys
import time
import subprocess

from pywinauto import Application
from pathlib import Path

# To run ALL tests in this file:
# python -m unittest -v GUI.Tests.gui_tests

# To run specific tests:
# python -m unittest GUI.Tests.gui_tests.TestGUIWindow.test_window_exists

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
        button = self.window.child_window(title="Midnight", control_type="Button")
        self.assertTrue(button.exists())
    
    # Implement after clock is fixed (currently not fixed)
    def test_clock_exists(self):
        # # (Name) text_Clock
        # clock = self.window.child_window(title="edit_Clock", control_type="Edit")
        
        # # (Legacy|Accessible.Value or Value.Value) Time shown on clock
        # print(clock.get_value())
        pass
    
    # Need to update later as timer's name is the actual time.
    def test_click_midnight_button(self):
        button = self.window.child_window(title="Midnight", control_type="Button")
        self.assertTrue(button.exists(), "Midnight button should exist")

        # Click the button
        button.click_input()

        # After clicking, check that the clock label shows midnight
        clock_label = self.window.child_window(title="00:00:00", control_type="Text")
        self.assertTrue(clock_label.exists(), "Clock should reset to 00:00:00 after Midnight is clicked")
    
    def test_slider_time_exists(self):
        slider = self.window.child_window(title="slider_Time", control_type="Slider")
        self.assertTrue(slider.exists(), "Time slider should exist")
        
    # Tests for Emelee:
    def test_start_button_exists(self):
        # (title: Start)
        pass
    
    def test_stop_button_exists(self):
        # (title: Stop)
        pass
    
    def test_now_button_exists(self):
        # (title: Now)
        pass

    # def test_print_all_elements(self):
    #     self.window.print_control_identifiers()

if __name__ == "__main__":
    unittest.main() 