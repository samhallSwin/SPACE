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
import pytest
import numpy as np
import flwr as fl
from unittest.mock import MagicMock
from fl_core import FederatedLearning, FlowerClient
from model import Model

@pytest.fixture
def federated_learning():
    return FederatedLearning()

# tests setters in fl_core
def test_set_num_rounds(federated_learning):
    federated_learning.set_num_rounds(5)
    assert federated_learning.num_rounds == 5

def test_set_num_clients(federated_learning):
    federated_learning.set_num_clients(3)
    assert federated_learning.num_clients == 3

def test_set_model(federated_learning):
    model = Model()
    federated_learning.set_model(model)
    assert federated_learning.model_manager == model

# tests client initialisation with specific model
def test_flower_client_initialisation(mocker):
    mock_model_manager = MagicMock() # mock of model manager
    mock_model_manager.create_model.return_value = MagicMock()
    mock_model_manager.load_data.return_value = ([1, 2, 3], [4, 5, 6]) # dummy data for dataset

    client = FlowerClient(mock_model_manager) # calls create_model() and load_data()

    assert client.model == mock_model_manager.create_model()
    assert client.x_train == [1, 2, 3]
    assert client.y_train == [4, 5, 6]



