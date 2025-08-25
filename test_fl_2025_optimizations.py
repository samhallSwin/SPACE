#!/usr/bin/env python3
"""
Test script for 2025 FL optimizations
Tests the enhanced federated learning functionality with FLAM integration
"""

import os
import sys
import json
import torch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.fl_core import FederatedLearning
from federated_learning.fl_output import FLOutput
from federated_learning.fl_visualization import FLVisualization

def test_fl_2025_features():
    """Test the 2025 FL optimizations and features"""
    print("Testing 2025 FL Optimizations...")
    
    # Find the latest FLAM file
    flam_dir = "synth_FLAMs"
    flam_files = [f for f in os.listdir(flam_dir) if f.endswith('.csv')]
    if not flam_files:
        print("❌ No FLAM files found for testing")
        return False
    
    latest_flam = max(flam_files, key=lambda x: os.path.getctime(os.path.join(flam_dir, x)))
    flam_path = os.path.join(flam_dir, latest_flam)
    print(f"Using FLAM file: {latest_flam}")
    
    # Test 1: FLAM Schedule Loading
    print("\nTest 1: FLAM Schedule Loading")
    fl = FederatedLearning()
    try:
        schedule = fl.load_flam_schedule(flam_path)
        print(f"✅ Successfully loaded {len(schedule)} timesteps")
        print(f"   Sample schedule: {schedule[:2]}")
    except Exception as e:
        print(f"❌ FLAM loading failed: {e}")
        return False
    
    # Test 2: FL Core Initialization
    print("\nTest 2: FL Core Initialization")
    try:
        fl.set_num_rounds(3)
        fl.set_num_clients(4)
        print("✅ FL parameters set successfully")
    except Exception as e:
        print(f"❌ FL initialization failed: {e}")
        return False
    
    # Test 3: Data and Model Initialization
    print("\nTest 3: Data and Model Initialization")
    try:
        fl.initialize_data()
        fl.initialize_model()
        print("✅ Data and model initialized")
        print(f"   Client data loaders: {len(fl.client_data)}")
        print(f"   Model type: {type(fl.global_model).__name__}")
    except Exception as e:
        print(f"❌ Data/Model initialization failed: {e}")
        return False
    
    # Test 4: FLAM-based Round Execution
    print("\nTest 4: FLAM-based Round Execution")
    try:
        # Test TRAINING phase
        training_entry = {"phase": "TRAINING"}
        fl.run_flam_round(training_entry)
        print("✅ TRAINING phase executed")
        
        # Test TRANSMITTING phase  
        transmitting_entry = {"phase": "TRANSMITTING"}
        fl.run_flam_round(transmitting_entry)
        print("✅ TRANSMITTING phase executed")
    except Exception as e:
        print(f"❌ FLAM round execution failed: {e}")
        return False
    
    # Test 5: Metrics Collection
    print("\nTest 5: Metrics Collection")
    try:
        metrics = fl.get_round_metrics()
        print("✅ Metrics collected successfully")
        print(f"   Available metrics: {list(metrics.keys())}")
    except Exception as e:
        print(f"❌ Metrics collection failed: {e}")
        return False
    
    # Test 6: FL Output Module
    print("\nTest 6: FL Output Module")
    try:
        output = FLOutput()
        eval_metrics = output.evaluate_model(fl.global_model, 10.0)
        print("✅ Model evaluation completed")
        print(f"   Accuracy: {eval_metrics['accuracy']:.2f}%")
        print(f"   Loss: {eval_metrics['loss']:.4f}")
    except Exception as e:
        print(f"❌ FL output failed: {e}")
        return False
    
    # Test 7: Visualization Module
    print("\nTest 7: Visualization Module")
    try:
        # Create a test results directory
        test_dir = "test_fl_results"
        os.makedirs(test_dir, exist_ok=True)
        
        # Save test metrics
        test_metrics = {
            "accuracy": eval_metrics['accuracy'],
            "loss": eval_metrics['loss'],
            "processing_time": 10.0,
            "timestamp": datetime.now().isoformat(),
            "additional_metrics": {
                "round_times": {"round_1": 3.5, "round_2": 4.2},
                "round_accuracies": [85.5, 87.2]
            }
        }
        
        metrics_file = os.path.join(test_dir, "test_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Test visualization
        viz = FLVisualization(test_dir)
        viz.visualize_from_json(metrics_file)
        print("✅ Visualization generated successfully")
        print(f"   Dashboard saved to: {test_dir}/dashboard.html")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False
    
    print("\n🎉 All FL 2025 optimization tests passed!")
    return True


if __name__ == "__main__":
    print("Testing SPACE 2025 FL Optimizations")
    print("=" * 50)
    
    success = test_fl_2025_features()
    
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
    print("\n" + "=" * 50) 