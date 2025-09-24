"""
path manager - manage all paths in the project
ensure the project runs smoothly on any user and operating system
Author: stephen zeng
Date: 2025-06-04
Version: 1.0
Python Version: 3.10

Changelog:
- 2025-06-04: Initial creation.
- 2025-06-04: Added convenience functions to get the project root, synth_FLAMs directory, algorithm output directory, TLE directory, satellite simulation directory, and federated learning directory.
- 2025-06-04: Added a method to get the latest csv file.
"""
import os
from pathlib import Path

class ProjectPathManager:
    """project path manager class"""
    
    def __init__(self):
        # get the project root (by finding specific identifier files)
        self.project_root = self._find_project_root()
        
    def _find_project_root(self):
        """automatically find the project root"""
        current_path = Path(__file__).resolve()
        
        # search up until finding the directory containing main.py
        for parent in current_path.parents:
            if (parent / "main.py").exists() and (parent / "requirements.txt").exists():
                return parent
                
        # if not found, use the current working directory
        return Path.cwd()
    
    @property
    def root(self):
        """project root directory"""
        return self.project_root
    
    @property
    def synth_flams_dir(self):
        """synthetic FLAMs output directory"""
        path = self.project_root / "synth_FLAMs"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def algorithm_output_dir(self):
        """algorithm output directory"""
        path = self.project_root / "flomps_algorithm" / "output"
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def tle_dir(self):
        """TLE file directory"""
        return self.project_root / "TLEs"
    
    @property
    def sat_sim_dir(self):
        """satellite simulation directory"""
        return self.project_root / "sat_sim"
    
    @property
    def federated_learning_dir(self):
        """federated learning directory"""
        return self.project_root / "federated_learning"
    
    def get_output_file_path(self, filename, subdirectory=None):
        """get the full path of the output file"""
        if subdirectory:
            base_path = self.project_root / subdirectory
        else:
            base_path = self.project_root
        
        base_path.mkdir(exist_ok=True)
        return base_path / filename
    
    def get_latest_csv_file(self, pattern="flam_*.csv"):
        """get the latest csv file"""
        import glob
        csv_files = list(self.synth_flams_dir.glob(pattern))
        if csv_files:
            return max(csv_files, key=lambda x: x.stat().st_ctime)
        return None

# global instance
path_manager = ProjectPathManager()

# convenience function
def get_project_root():
    """get the project root"""
    return path_manager.root

def get_synth_flams_dir():
    """get the synth_FLAMs directory"""
    return path_manager.synth_flams_dir

def get_algorithm_output_dir():
    """get the algorithm output directory"""
    return path_manager.algorithm_output_dir 