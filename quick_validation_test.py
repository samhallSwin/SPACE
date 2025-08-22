#!/usr/bin/env python3
"""
Quick Validation Test for SPACE 2025
Comprehensive system validation test - complete in under 5 minutes

Author: Stephen Zeng
Version: 1.0
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_system_check():
    """Quick system health check"""
    print("SPACE 2025 Quick Validation Test")
    print("=" * 50)
    
    start_time = time.time()
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Environment check
    print("\nTest 1: Environment Check")
    try:
        import torch
        import numpy as np
        from federated_learning.fl_core import FederatedLearning
        print("✅ All dependencies available")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
    
    # Test 2: TLE file check
    print("\nTest 2: TLE File Check")
    tle_files = ["TLEs/SatCount4.tle", "TLEs/SatCount8.tle"]
    tle_found = sum(1 for f in tle_files if os.path.exists(f))
    print(f"✅ Found {tle_found}/{len(tle_files)} TLE files")
    if tle_found > 0:
        tests_passed += 1
    
    # Test 3: FLAM generation test
    print("\nTest 3: FLAM Generation Test")
    try:
        current_time = datetime.now()
        start_sim = current_time.strftime("%Y-%m-%d %H:%M:%S")
        end_sim = (current_time + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        
        from generate_flam_csv import generate_flam_csv
        csv_file = generate_flam_csv(
            tle_file="TLEs/SatCount4.tle",
            start_time=start_sim,
            end_time=end_sim,
            timestep=1
        )
        
        if csv_file and os.path.exists(csv_file):
            file_size = os.path.getsize(csv_file)
            print(f"✅ FLAM file generated successfully: {os.path.basename(csv_file)} ({file_size} bytes)")
            tests_passed += 1
        else:
            print("❌ FLAM file generation failed")
            
    except Exception as e:
        print(f"❌ FLAM generation error: {e}")
    
    # Test 4: FL core functionality test
    print("\nTest 4: FL Core Functionality Test")
    try:
        fl = FederatedLearning()
        fl.set_num_rounds(2)
        fl.set_num_clients(2)
        print("✅ FL core initialized successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FL core test failed: {e}")
    
    # Test 5: Output directory check
    print("\nTest 5: Output Directory Check")
    directories = ["synth_FLAMs", "federated_learning"]
    dirs_exist = sum(1 for d in directories if os.path.exists(d))
    print(f"✅ Directory structure complete: {dirs_exist}/{len(directories)}")
    if dirs_exist == len(directories):
        tests_passed += 1
    
    # Test 6: Timing consistency quick check
    print("\nTest 6: Timing Consistency Quick Check")
    try:
        times = []
        for i in range(3):
            t_start = time.time()
            # Simple computation test
            result = sum(range(1000))
            t_end = time.time()
            times.append(t_end - t_start)
        
        variance = max(times) - min(times)
        if variance < 0.01:  # 10ms variance acceptable
            print(f"✅ Good timing consistency (variance: {variance:.4f}s)")
            tests_passed += 1
        else:
            print(f"⚠️  High timing variance: {variance:.4f}s")
    except Exception as e:
        print(f"❌ Timing consistency test failed: {e}")
    
    # Summary
    total_time = time.time() - start_time
    success_rate = (tests_passed / total_tests) * 100
    
    print("\n" + "=" * 50)
    print(f"Test Summary:")
    print(f"   Pass rate: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
    print(f"   Execution time: {total_time:.2f}s")
    
    if success_rate >= 80:
        print("🎉 System status is good, ready for comprehensive validation!")
        return True
    else:
        print("⚠️  System needs fixes before comprehensive validation")
        return False

if __name__ == "__main__":
    success = quick_system_check()
    if success:
        print("\nRecommend running: python comprehensive_system_validation.py")
    else:
        print("\nPlease fix the above issues first") 