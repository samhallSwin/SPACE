#!/usr/bin/env python3
"""
FLAM CSV Generator
Main utility to generate FLAM CSV files using SatSim → Algorithm workflow.
Uses direct core classes (bypasses handlers) for better reliability.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from sat_sim.sat_sim import SatSim
from flomps_algorithm.algorithm_core import Algorithm
import json
from skyfield.api import load

def generate_flam_csv(tle_file="TLEs/SatCount4.tle"):
    """
    Generate FLAM CSV using SatSim + Algorithm direct approach.

    This uses core classes directly instead of handlers for better reliability.
    The handler-based approach had compatibility issues and was removed.
    """

    print("🚀 Starting FLAM CSV Generation...")
    print(f"📡 Using TLE file: {tle_file}")

    # Load configuration
    with open('options.json', 'r') as f:
        config = json.load(f)

    # Calculate expected timesteps
    from datetime import datetime
    start_time = datetime.strptime(config['sat_sim']['start_time'], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(config['sat_sim']['end_time'], "%Y-%m-%d %H:%M:%S")
    timestep_minutes = config['sat_sim']['timestep']
    total_minutes = int((end_time - start_time).total_seconds() / 60)
    expected_timesteps = total_minutes // timestep_minutes

    print(f"⏱️  Expected timesteps: {expected_timesteps}")
    print(f"    Start: {start_time}")
    print(f"    End: {end_time}")
    print(f"    Interval: {timestep_minutes} minutes")

    try:
        # Step 1: Run SatSim directly
        print("\n📡 Setting up SatSim...")

        # Convert to Skyfield time objects
        ts = load.timescale()
        start_skyfield = ts.utc(start_time.year, start_time.month, start_time.day,
                               start_time.hour, start_time.minute, start_time.second)
        end_skyfield = ts.utc(end_time.year, end_time.month, end_time.day,
                             end_time.hour, end_time.minute, end_time.second)

        # Create SatSim instance
        sat_sim = SatSim(
            start_time=start_skyfield,
            end_time=end_skyfield,
            timestep=timestep_minutes,
            output_file_type='txt',  # We'll handle CSV ourselves
            gui_enabled=False,
            output_to_file=False  # We don't need SatSim's file output
        )

        # Read TLE data
        print(f"📡 Loading TLE file: {tle_file}")
        tle_data = {}
        with open(tle_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # Parse TLE data (3LE format)
        i = 0
        while i < len(lines):
            if i + 2 < len(lines):
                name = lines[i].strip()
                tle_line1 = lines[i + 1].strip()
                tle_line2 = lines[i + 2].strip()
                tle_data[name] = [tle_line1, tle_line2]
                i += 3
            else:
                break

        sat_sim.set_tle_data(tle_data)
        print(f"✅ Loaded TLE data for {len(tle_data)} satellites")

        # Run SatSim to get adjacency matrices
        print("📡 Running SatSim simulation...")
        adjacency_matrices = sat_sim.run_with_adj_matrix()
        print(f"✅ SatSim completed: {len(adjacency_matrices)} timesteps generated")

        # Step 2: Run Algorithm with SatSim data
        print("\n🧮 Setting up Algorithm...")
        algorithm = Algorithm()

        # Set up algorithm with SatSim output
        algorithm.set_adjacency_matrices(adjacency_matrices)

        # Set Sam's algorithm parameters
        algorithm.set_algorithm_parameters(
            toggle_chance=0.1,
            training_time=3,
            down_bias=2.0
        )

        # Set satellite names from TLE data
        satellite_names = list(tle_data.keys())
        algorithm.set_satellite_names(satellite_names)

        print(f"🧮 Running algorithm with {len(satellite_names)} satellites...")

        # Run the algorithm
        algorithm.start_algorithm_steps()
        print("✅ Algorithm completed!")

        # Step 3: Check output files
        print("\n📄 Checking output files...")

        # Find the generated CSV file
        import glob
        csv_files = glob.glob("/Users/ash/Desktop/SPACE_FLTeam/synth_FLAMs/*.csv")
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getctime)
            print(f"✅ CSV file generated: {os.path.basename(latest_csv)}")

            # Count lines to verify timesteps
            with open(latest_csv, 'r') as f:
                content = f.read()
                timestep_count = content.count('Timestep:')
                print(f"📊 CSV contains {timestep_count} timesteps")

            return latest_csv
        else:
            print("❌ No CSV file found in synth_FLAMs/")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Allow TLE file to be specified as command line argument
    tle_file = sys.argv[1] if len(sys.argv) > 1 else "TLEs/SatCount4.tle"

    result = generate_flam_csv(tle_file)

    if result:
        print(f"\n🎉 SUCCESS! FLAM CSV generated: {os.path.basename(result)}")
        print(f"📁 Full path: {result}")
        print("\n🔗 This CSV can now be used by FL core!")
    else:
        print("\n💥 FAILED to generate FLAM CSV")
