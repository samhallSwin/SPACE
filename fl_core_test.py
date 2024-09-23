"""
Filename: fl_core_test.py
Description: Test file for model.py
Author: Ahmed Mubeenul Azim & Michelle Abrigo
Date: 2024-09-23
Version: 1.0
Python Version: 

Changelog:
- 2024-08-02: Initial creation, setup & test case creation.

Usage: Do 'pytest fl_core_test.py' to run the test session

"""
from fl_core import FederatedLearning, FlowerClient
import tensorflow as tf
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture
