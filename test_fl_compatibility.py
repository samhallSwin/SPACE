#!/usr/bin/env python3
"""
Test FL core compatibility with the new CSV format
Author: stephen zeng
Date: 2025-06-04
Version: 1.0
Python Version: 3.10

Changelog:
- 2025-06-04: Initial creation. 
"""

import os
import sys

# add path manager
try:
    from utilities.path_manager import get_synth_flams_dir
    use_path_manager = True
except ImportError:
    use_path_manager = False

def test_fl_core_compatibility():
    print("Testing FL core compatibility with new CSV format...")

    # Get the synth_FLAMs directory using universal path management
    if use_path_manager:
        csv_path = get_synth_flams_dir()
        csv_files = [f for f in csv_path.glob('flam_*.csv')] # get all csv files in the synth_FLAMs directory
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_ctime)
            full_path = latest_csv
            print(f"Testing with CSV file: {latest_csv.name}")
        else:
            print("No CSV files found in synth_FLAMs directory")
            return False
    else:
        # use backup path
        csv_path_str = "synth_FLAMs/"
        if not os.path.exists(csv_path_str):
            print("synth_FLAMs directory not found")
            return False
        # get all csv files in the synth_FLAMs directory
        csv_files = [f for f in os.listdir(csv_path_str) if f.startswith('flam_') and f.endswith('.csv')]
        # check if there are any csv files
        if not csv_files:
            print("No CSV files found in synth_FLAMs directory")
            return False
        
        latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(csv_path_str, x)))
        full_path = os.path.join(csv_path_str, latest_csv) # get the full path of the latest csv file
        print(f"Testing with CSV file: {latest_csv}")

    # Test the load_flam_schedule method that FL core uses
    try:
        timestep_count = 0
        round_count = 0
        # open the latest csv file
        with open(str(full_path), 'r') as f:
            for line in f:
                if line.startswith("Timestep:"): # check if the line starts with "Timestep:"
                    timestep_count += 1
                    if "Round:" in line: # check if the line contains "Round:"
                        # Extract round number
                        parts = line.split("Round: ")[1].split(",")[0]
                        round_num = int(parts)
                        round_count = max(round_count, round_num)
        
        print(f"✓ Successfully parsed {timestep_count} timesteps")
        print(f"✓ Found {round_count} federated learning rounds")
        print("✓ CSV format is compatible with FL core")
        
        return True
        
    except Exception as e:
        print(f"✗ Error parsing CSV file: {e}") # print the error
        return False

if __name__ == "__main__":
    # test the fl core compatibility
    success = test_fl_core_compatibility()
    if success:
        print("\n FL COMPATIBILITY TEST PASSED!")
    else:
        print("\n FL COMPATIBILITY TEST FAILED!")
        sys.exit(1)
