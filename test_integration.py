#!/usr/bin/env python3
"""
Simple integration test for Sam's algorithm implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from flomps_algorithm.algorithm_core import Algorithm
from flomps_algorithm.algorithm_output import AlgorithmOutput

# add path manager
try:
    from utilities.path_manager import get_synth_flams_dir
    use_path_manager = True
except ImportError:
    use_path_manager = False

def test_sam_algorithm():
    print("Testing Sam's algorithm integration...")

    # Create algorithm instance
    algorithm = Algorithm()

    # Set up basic test data
    satellite_names = ["Satellite_1", "Satellite_2", "Satellite_3", "Satellite_4", "Satellite_5"]
    algorithm.set_satellite_names(satellite_names)

    # Set algorithm parameters (Sam's defaults)
    algorithm.set_algorithm_parameters(toggle_chance=0.1, training_time=3, down_bias=2.0)

    # Create some test adjacency matrices (simulating timesteps)
    test_matrices = []
    for i in range(10):  # 10 timesteps
        # Create a simple test matrix with some connectivity
        matrix = np.random.choice([0, 1], size=(5, 5), p=[0.7, 0.3])
        # Make it symmetric and zero diagonal
        matrix = (matrix + matrix.T) // 2
        np.fill_diagonal(matrix, 0)
        timestamp = f"2025-06-03 12:{i:02d}:00"
        test_matrices.append((timestamp, matrix))

    algorithm.set_adjacency_matrices(test_matrices)

    # Run the algorithm
    print("Running algorithm steps...")
    algorithm.start_algorithm_steps()

    # Get results
    output = algorithm.output.get_result()
    print(f"Algorithm completed. Generated {len(output)} timestep results.")

    # Check first few results
    if isinstance(output, dict):
        # Handle direct dictionary output
        for i, (timestamp, result) in enumerate(list(output.items())[:3]):
            print(f"\nTimestep {i+1} ({timestamp}):")
            print(f"  Round: {result.get('round_number', 'N/A')}")
            print(f"  Phase: {result.get('phase', 'N/A')}")
            print(f"  Target Node: {result.get('target_node', 'N/A')}")
            print(f"  Selected Satellite: {result.get('selected_satellite', 'N/A')}")
            print(f"  Aggregator Flag: {result.get('aggregator_flag', 'N/A')}")
            print(f"  Matrix shape: {result['federatedlearning_adjacencymatrix'].shape}")
    else:
        # Handle DataFrame output
        for i in range(min(3, len(output))):
            result = output.iloc[i]
            print(f"\nTimestep {i+1}:")
            print(f"  Timestamp: {result.get('time_stamp', 'N/A')}")
            print(f"  Selected Satellite: {result.get('satellite_name', 'N/A')}")
            print(f"  Aggregator Flag: {result.get('aggregator_flag', 'N/A')}")
            print(f"  Matrix shape: {result['federatedlearning_adjacencymatrix'].shape if 'federatedlearning_adjacencymatrix' in result else 'N/A'}")

    print("\nTest completed successfully!")

    # Check outputs using universal paths
    if use_path_manager:
        csv_path = get_synth_flams_dir()
        csv_files = [f.name for f in csv_path.glob('flam_*.csv')]
    else:
        # use backup path: synth_FLAMs/
        csv_path = "synth_FLAMs/"
        if os.path.exists(csv_path):
            csv_files = [f for f in os.listdir(csv_path) if f.startswith('flam_') and f.endswith('.csv')]
        else:
            csv_files = []
    
    if csv_files:
        print(f"Sam's CSV format files created: {csv_files}")
    else:
        print("No CSV files found in synth_FLAMs directory")

if __name__ == "__main__":
    test_sam_algorithm()
