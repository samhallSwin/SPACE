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
        
    def test_expand_collapse_button_exists(self):
        
        # Verify that the Expand/Collapse button exists within the TLEDisplay panel.
        # The test checks if the overlay component (TLEDisplay) is present and contains the expected button element.

        # Locate the TLEDisplay container by its name
        tle_display_overlay = self.window.child_window(title="TLEDisplayOverlay")
        self.assertTrue(tle_display_overlay.exists(), "TLEDisplay panel should exist")

        # Locate the expand/collapse button inside the TLEDisplay
        button = self.window.child_window(class_name="TLEDisplay")
        self.assertTrue(button.exists(), "Expand/Collapse button should exist within TLEDisplay")

        initial_visible = tle_display_overlay.is_enabled()

        # Click the button to toggle expand/collapse
        button.click_input()
        time.sleep(0.5)  # Give GUI time to update

        # Verify that the visibility has changed (toggled)
        new_visible = tle_display_overlay.is_enabled()
        self.assertNotEqual(
            initial_visible,
            new_visible,
            "Clicking the Expand/Collapse button should toggle TLEDisplay visibility"
        )

    def test_graph_display_expand_button_exists(self):
        
        # Verify that the Expand/Collapse button exists within the GraphDisplay panel.
        
        graph_display_overlay = self.window.child_window(title="GraphDisplayOverlay")
        self.assertTrue(graph_display_overlay.exists(), "GraphDisplay panel should exist")

        button = self.window.child_window(class_name="GraphDisplay")
        self.assertTrue(button.exists(), "Expand/Collapse button should exist within GraphDisplay")

    # def test_print_all_elements(self):
    #     self.window.print_control_identifiers()

if __name__ == "__main__":
    unittest.main() 