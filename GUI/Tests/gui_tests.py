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
        cls.repo_root = Path(__file__).parent.parent
        cls.proc = subprocess.Popen([str(python_exe), str(script_path)])
        
        cls.app = Application(backend="uia")
        for _ in range(30):
            try:
                cls.app.connect(title_re="S.P.A.C.E")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Could not find window with title 'S.P.A.C.E' after 15 seconds")

        cls.window = cls.app.window(title_re="S.P.A.C.E")
        cls.window.wait("visible", timeout=10)        

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()

    def test_window_exists(self):
        self.assertTrue(self.window.exists())
        
    def test_midnight_button_exists(self):
        button = self.window.child_window(title="Midnight", control_type="Button")
        self.assertTrue(button.exists())
    
    def test_clock_exists(self):
        clock_container = self.window.child_window(title="clock_Time", control_type="Group")
        clock_edit = clock_container.child_window(control_type="Edit")
        self.assertTrue(clock_edit.exists())
    
    def test_click_midnight_button(self):
        midnight_button = self.window.child_window(title="Midnight", control_type="Button")
        midnight_button.click_input()

        stop_button = self.window.child_window(title="Stop", control_type="Button")
        stop_button.click_input()

        clock_container = self.window.child_window(title="clock_Time", control_type="Group")
        clock_edit = clock_container.child_window(control_type="Edit")
        
        self.assertTrue(clock_container.exists())
        self.assertTrue(clock_edit.exists())

        clock_text = clock_edit.iface_value.CurrentValue
        print(f"Clock shows: {clock_text}")
        self.assertEqual(clock_text, "00:00:00", "Clock should reset to 00:00:00 after Midnight is clicked")
            
    def test_slider_time_exists(self):
        slider = self.window.child_window(title="slider_Time", control_type="Slider")
        self.assertTrue(slider.exists(), "Time slider should exist")
        
    # Tests for Emelee:
    def test_start_button_exists(self):
        button = self.window.child_window(title="Start", control_type="Button")
        self.assertTrue(button.exists(), "Start button should exist")

    def test_stop_button_exists(self):
        button = self.window.child_window(title="Stop", control_type="Button")
        self.assertTrue(button.exists(), "Stop button should exist")

    def test_now_button_exists(self):
        button = self.window.child_window(title="Now", control_type="Button")
        self.assertTrue(button.exists(), "Now button should exist")
        
    def test_tle_expand_collapse_button_exists(self):
        button = self.window.child_window(title="button_ExpandCollapse", control_type="Button")
        self.assertTrue(button.exists(), "TLE Expand/Collapse button should exist")
        
    # Overlay works by physically collapsing and hiding via opacity
    # Could be worth looking into setVisible
    # def test_click_tle_expand_collapse_button(self):
    #     button = self.window.child_window(title="button_ExpandCollapse", control_type="Button")
    #     label = self.window.child_window(title="Drop Label", control_type="Text")
    #     initial_visible = label.is_visible()
        
    #     tle_file = self.repo_root / "TLEs" / "SatCount1.tle"
    #     label.fileDropped.emit(tle_file)

    #     button.click_input()
    #     time.sleep(0.5)
    #     new_visible = label.is_visible()
    #     self.assertNotEqual(initial_visible, new_visible, "Clicking Expand/Collapse should toggle visibility of DropHere label")

    # def test_print_all_elements(self):
    #     self.window.print_control_identifiers()

if __name__ == "__main__":
    unittest.main() 