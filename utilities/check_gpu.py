"""
Filename: check_gpu.py
Description: Script for checking local GPU avalibility.
Author: Connor Bett
Date: 2024-08-02
Version: 1.0
Python Version: 3.12.3

Changelog:
- 2024-08-02: Initial creation.

Usage: Run this as a standalone script.

"""
import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

if gpus:
    print("CUDA and cuDNN are properly installed.")
else:
    print("CUDA and cuDNN are not detected.")