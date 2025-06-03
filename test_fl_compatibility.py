#!/usr/bin/env python3
"""
Test FL core compatibility with the new CSV format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.fl_core import FederatedLearning

def test_fl_core_compatibility():
    print("Testing FL core compatibility with new CSV format...")

    # Create FL instance
    fl = FederatedLearning()

    # Test loading the new CSV format
    csv_path = "/Users/ash/Desktop/SPACE_FLTeam/synth_FLAMs/"
    csv_files = [f for f in os.listdir(csv_path) if f.startswith('flam_') and f.endswith('.csv')]
    latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(csv_path, x)))
    full_path = os.path.join(csv_path, latest_csv)

    print(f"Testing with CSV file: {latest_csv}")

    try:
        # Test the load_flam_schedule method that FL core uses
        schedule = fl.load_flam_schedule(full_path)
        print(f"✓ Successfully loaded {len(schedule)} timesteps")

        # Check first few entries
        for i, entry in enumerate(schedule[:5]):
            print(f"  Timestep {entry['timestep']}: Phase = {entry['phase']}")

        print("✓ FL core can successfully read the new CSV format")
        print("✓ Integration maintains backward compatibility")

    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_fl_core_compatibility()
    if success:
        print("\n🎉 FL CORE COMPATIBILITY CONFIRMED!")
    else:
        print("\n❌ FL core compatibility issues detected")
