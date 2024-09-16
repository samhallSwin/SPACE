"""
Filename: model_test.py
Description: Test file for model.py
Author: Ahmed Mubeenul Azim & Michelle Abrigo
Date: 2024-09-16
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation, setup & test case creation.

Usage: Do 'pytest model_test.py' to run the test session

"""

from model import Model
import tensorflow as tf
import numpy as np
import pytest

@pytest.fixture
def model():
    """Fixture to create a Model instance for reuse."""
    return Model()

# Test model initialisation
def test_init(model):
    assert model.get_model_type() == ""
    assert model.get_data_set() == ""