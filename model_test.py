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
    return Model()

# Test model initialisation
def test_init(model):
    assert model.get_model_type() == ""
    assert model.get_data_set() == ""

# Test model type setters and getters
def test_set_get_model_type(model):
    model.set_model_type("SimpleCNN")
    assert model.get_model_type() == "SimpleCNN"

    model.set_model_type("RestNet50")
    assert model.get_model_type() == "RestNet50"

# Test dataset setters and getters
def test_set_get_data_set(model):
    model.set_data_set("MNIST")
    assert model.get_data_set() == "MNIST"  

    model.set_data_set("BigEarthNet")
    assert model.get_data_set() == "BigEarthNet"   

# Test for load_data
def test_load_simplecnn_mnist(model):
    model.set_model_type("SimpleCNN")
    model.set_data_set("MNIST")
    # further elaborate wip

def test_load_simplecnn_mnist(model):
    model.set_model_type("RestNet50")
    model.set_data_set("MNIST")
    # further elaborate wip

# note: bigearthnet under models need to be implemented once it actually starts working

# Test for create_model, still a WIP
def test_create_model_simplecnn(model):
    model.set_model_type("SimpleCNN")
    simplecnn = model.create_model()
    assert simplecnn.input_shape == (None, 28, 28, 1) # None = batch_size, unsure if need to specify - need clarification

def test_create_model_resnet50(model):
    model.set_model_type("ResNet50")
    resnet50 = model.create_model()
    assert resnet50.input_shape == (None, 32, 32, 3)